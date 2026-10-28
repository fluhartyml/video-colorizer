"""Speed and picture difference: PyTorch (Apple GPU, MPS) vs Core ML, on N real frames from CLIP (starting at frame 600).

Usage: bench.py DDCOLOR_SRC CLIP N MLPACKAGE [MLPACKAGE ...]

For each .mlpackage it runs Core ML with compute units ALL (Neural Engine allowed) and CPU_AND_GPU, and prints
frames per second plus the mean and worst per-pixel difference from the PyTorch output (0-255). It also writes
coreml/sample-<package>-<units>.jpg: PyTorch on the left, Core ML on the right, for frame 50 of the sample.
"""
import sys, time
import cv2, numpy as np, torch, coremltools as ct
import torch.nn.functional as F
src_root, clip, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, src_root)
from huggingface_hub import PyTorchModelHubMixin
from ddcolor import DDColor
class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict): kwargs = {**config, **kwargs}
        super().__init__(**kwargs)

cap = cv2.VideoCapture(clip); frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
while len(frames) < n:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)

def prep(img_bgr):
    img = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    r = cv2.resize(img, (512, 512))
    l = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)[:, :, :1]
    g = cv2.cvtColor(np.concatenate((l, np.zeros_like(l), np.zeros_like(l)), -1), cv2.COLOR_LAB2RGB)
    return orig_l, g.transpose(2, 0, 1)[None].astype(np.float32)

def finish(orig_l, ab, h, w):
    ab = F.interpolate(torch.from_numpy(ab), size=(h, w))[0].numpy().transpose(1, 2, 0)
    return (cv2.cvtColor(np.concatenate((orig_l, ab), -1), cv2.COLOR_LAB2BGR) * 255).round().clip(0, 255).astype(np.uint8)

dev = torch.device("mps")
model = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope").to(dev).eval()
ref = []
with torch.no_grad():
    model(torch.from_numpy(prep(frames[0])[1]).to(dev)).cpu()  # warm-up
    t = time.time()
    for f in frames:
        ol, x = prep(f); ab = model(torch.from_numpy(x).to(dev)).cpu().numpy()
        ref.append(finish(ol, ab, *f.shape[:2]))
    print(f"PyTorch MPS: {len(frames) / (time.time() - t):.2f} fps")

for pkg in sys.argv[4:]:
    name = pkg.rstrip("/").split("/")[-1]
    for units in ("ALL", "CPU_AND_GPU"):
        m = ct.models.MLModel(pkg, compute_units=getattr(ct.ComputeUnit, units))
        m.predict({"gray_rgb": prep(frames[0])[1]})  # warm-up
        t = time.time(); out = []
        for f in frames:
            ol, x = prep(f); out.append(finish(ol, m.predict({"gray_rgb": x})["ab"], *f.shape[:2]))
        fps = len(frames) / (time.time() - t)
        diff = np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(out, ref)])
        worst = max(np.abs(a.astype(int) - b.astype(int)).max() for a, b in zip(out, ref))
        print(f"{name} on {units}: {fps:.2f} fps | mean pixel diff {diff:.2f} of 255 | worst {worst}")
        cv2.imwrite(f"coreml/sample-{name}-{units}.jpg", np.hstack([ref[50], out[50]]))
