"""Colorize a video with the Core ML conversion of DDColor (see coreml/convert.py), frame by frame. macOS only.

Usage: colorize-coreml.py MODEL.mlpackage INPUT OUTPUT [UNITS]

  UNITS  CPU_AND_GPU (default) | ALL | CPU_AND_NE | CPU_ONLY
         ALL lets Core ML use the Neural Engine: much faster, but with the FP16 model the colors come out
         visibly washed out (README.md, "Measured results"). CPU_AND_GPU matches the PyTorch output closely.

Pre- and post-processing are the same as DDColor's own ColorizationPipeline (ddcolor/pipeline.py): the frame's
lightness (L of Lab) is kept at full size; the model only predicts the two color channels (a, b) at 512 x 512,
which are scaled back up and joined to the original L.
"""
import subprocess
import sys
import time

import coremltools as ct
import cv2
import numpy as np

model_path, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
units = sys.argv[4] if len(sys.argv) > 4 else "CPU_AND_GPU"
SIZE = 512

model = ct.models.MLModel(model_path, compute_units=getattr(ct.ComputeUnit, units))

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


def colorize(img_bgr):
    img = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    small = cv2.resize(img, (SIZE, SIZE))
    small_l = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_rgb = cv2.cvtColor(np.concatenate((small_l, np.zeros_like(small_l), np.zeros_like(small_l)), -1),
                            cv2.COLOR_LAB2RGB)
    ab = model.predict({"gray_rgb": gray_rgb.transpose(2, 0, 1)[None].astype(np.float32)})["ab"][0]
    # Nearest-neighbour, as DDColor's pipeline does with F.interpolate's default mode.
    ab = cv2.resize(ab.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(np.concatenate((orig_l, ab), -1), cv2.COLOR_LAB2BGR)
    return (bgr * 255.0).round().clip(0, 255).astype(np.uint8)


start, n = time.time(), 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    enc.stdin.write(colorize(frame).tobytes())
    n += 1
    if n % 50 == 0:
        print(f"{n}/{total} frames  {n / (time.time() - start):.2f} fps", flush=True)
enc.stdin.close()
enc.wait()
print(f"DONE {n} frames in {time.time() - start:.0f}s  ({n / (time.time() - start):.2f} fps)  units={units}")
