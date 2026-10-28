"""Convert DDColor (the PyTorch colorizer) to a Core ML model.

Usage: convert.py DDCOLOR_SRC OUT.mlpackage [MODEL] [PRECISION]

  MODEL      ddcolor_modelscope (default) | ddcolor_paper | ddcolor_artistic | ddcolor_paper_tiny
  PRECISION  fp16 (default) | fp32

Needs Python 3.9-3.13, torch 2.7.0 and coremltools 9.0 (requirements-coreml.txt): coremltools 9.0 ships no
wheel for Python 3.14 and was tested up to torch 2.7.0. The model is traced at a fixed 1 x 3 x 512 x 512 input,
which is the size DDColor's own pipeline feeds it. The printed difference uses a random input, which exaggerates
it; coreml/bench.py measures the difference on real frames.
"""
import sys
import numpy as np
import torch
import coremltools as ct

src_root, out = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "ddcolor_modelscope"
precision = sys.argv[4] if len(sys.argv) > 4 else "fp16"
sys.path.insert(0, src_root)
from huggingface_hub import PyTorchModelHubMixin  # noqa: E402
from ddcolor import DDColor  # noqa: E402


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


model = DDColorHF.from_pretrained(f"piddnad/{model_name}").eval()
example = torch.rand(1, 3, 512, 512)
with torch.no_grad():
    traced = torch.jit.trace(model, example)
    ref = model(example).numpy()
mlmodel = ct.convert(traced,
                     inputs=[ct.TensorType(name="gray_rgb", shape=example.shape)],
                     outputs=[ct.TensorType(name="ab")],
                     minimum_deployment_target=ct.target.macOS14,
                     compute_precision=ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16,
                     compute_units=ct.ComputeUnit.ALL)
mlmodel.save(out)
got = mlmodel.predict({"gray_rgb": example.numpy()})["ab"]
print("max difference vs PyTorch on a random input:", float(np.abs(got - ref).max()), "| ab range", float(ref.min()), float(ref.max()))
