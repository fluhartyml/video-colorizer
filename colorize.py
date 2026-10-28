"""Colorize a video with DDColor (PyTorch), frame by frame.

Usage: colorize.py DDCOLOR_SRC INPUT OUTPUT [MODEL]

  DDCOLOR_SRC  a clone of https://github.com/piddnad/DDColor (its `ddcolor` package is imported from there)
  INPUT        any video FFmpeg/OpenCV can read
  OUTPUT       an .mp4: H.264 (CRF 18), the input's audio re-encoded to AAC, the input's display aspect ratio kept
  MODEL        ddcolor_modelscope (default) | ddcolor_paper | ddcolor_artistic | ddcolor_paper_tiny

Runs on an NVIDIA GPU (CUDA), an Apple GPU (MPS), or the CPU, in that order of preference.
See README.md for speed, troubleshooting and the smoothing pass (smooth.sh) that should follow.
"""
import subprocess
import sys
import time

import cv2
import torch

src_root, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
model_name = sys.argv[4] if len(sys.argv) > 4 else "ddcolor_modelscope"
sys.path.insert(0, src_root)

from huggingface_hub import PyTorchModelHubMixin  # noqa: E402
from ddcolor import DDColor, ColorizationPipeline  # noqa: E402


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
model = DDColorHF.from_pretrained(f"piddnad/{model_name}").to(device).eval()
colorizer = ColorizationPipeline(model, input_size=512, device=device)

cap = cv2.VideoCapture(inp)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dar = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=display_aspect_ratio", "-of", "csv=p=0", inp],
                     capture_output=True, text=True).stdout.strip() or "4:3"

enc = subprocess.Popen([
    "ffmpeg", "-y", "-v", "error",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
    "-i", inp, "-map", "0:v", "-map", "1:a?",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-aspect", dar,
    "-c:a", "aac", "-b:a", "192k", "-shortest", out,
], stdin=subprocess.PIPE)

start, n = time.time(), 0
with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        enc.stdin.write(colorizer.process(frame).tobytes())
        n += 1
        if n % 50 == 0:
            el = time.time() - start
            print(f"{n}/{total} frames  {n / el:.2f} fps", flush=True)
enc.stdin.close()
enc.wait()
print(f"DONE {n} frames in {time.time() - start:.0f}s  ({n / (time.time() - start):.2f} fps)  model={model_name}  device={device}")
