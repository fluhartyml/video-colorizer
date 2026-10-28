# video-colorizer

Colorize black-and-white video, frame by frame, with the open-source **DDColor** model. It runs on an Apple Mac (Apple GPU or Core ML), on a PC with an NVIDIA GPU, or on any CPU. A smoothing pass then steadies the color across frames.

**Support page:** <https://fluharty.me/support/colorizer.html>

---

## Who made this

**Directed by Michael Fluharty. Written by Claude.**

**All of the code in this repository, and this README, were written by Claude**, Anthropic's AI model (Claude Opus 5.5), working through Claude Code on the owner's Macs, at the direction of **Michael Fluharty**, who owns the repository.

- **Michael** set the goal (colorize the black-and-white films and shows in his home video library), chose what to colorize, and decided when to stop, keep or replace results.
- **Claude** chose DDColor, wrote every script here, ran the batches, converted the model to Core ML, measured the speed and picture differences, researched the hardware questions and wrote this document.

None of the colorization model is ours. DDColor, its weights and every library used here belong to their authors. See [Credits and licenses](#credits-and-licenses). This repository's own license covers only the scripts and documents in it.

### How it was built

| Date | What happened |
|---|---|
| 2026-10-22 | The PyTorch pipeline (`colorize.py`, the smoothing step now in `smooth.sh`, and a batch runner) was written and used on an Apple M1 Mac mini. Two 25-minute episodes were colorized at 1.77–1.78 frames per second, about 5 h 52 min each. |
| 2026-10-27 | Asked whether newer hardware would help, Claude surveyed what had been published about Apple's M5 (see [Research notes](#research-notes-apple-m5-and-pytorch)). It found no PyTorch benchmark. Then it converted DDColor to Core ML (`coreml/convert.py`), measured it against PyTorch on real frames (`coreml/bench.py`), wrote the Core ML video script (`colorize-coreml.py`), tested it end to end, and published this repository. |

---

## What it does

```
black-and-white video
   │
   ├─► colorize.py (PyTorch)  or  colorize-coreml.py (Core ML)
   │      each frame: keep its lightness (L of Lab) at full size;
   │      the model predicts the two color channels (a, b) at 512 × 512;
   │      those are scaled back up and joined to the original lightness
   │      → raw colorized video (H.264, CRF 18, audio re-encoded to AAC)
   │
   └─► smooth.sh
          brightness from the ORIGINAL video, color averaged over 7 frames
          → finished "<name> (Colorized).mp4"
```

`batch.sh` runs both steps over a list of files and puts each result next to its original. It never touches the originals.

---

## Requirements

| | PyTorch path (`colorize.py`) | Core ML path (`colorize-coreml.py`, `coreml/`) |
|---|---|---|
| Computer | Mac with Apple silicon (MPS), PC with an NVIDIA GPU (CUDA), or any CPU | Mac only |
| Python | 3.14.9 tested | **3.9 to 3.13 only**. coremltools 9.0 ships no wheel for 3.14. Tested on 3.9.6, the Mac's built-in `/usr/bin/python3`. |
| Packages | `requirements.txt` | `requirements-coreml.txt` |
| Also needed | FFmpeg and ffprobe on `PATH` (tested with FFmpeg 9.0.2 from Homebrew), and a clone of DDColor | FFmpeg and ffprobe; DDColor clone only to convert |

Disk: about 1 GB for the weights and packages. The raw colorized file is written to `work/` before smoothing, so allow room for one extra copy of the largest video.

---

## Setup

```bash
git clone https://github.com/fluhartyml/video-colorizer.git
cd video-colorizer

# DDColor itself (Apache-2.0). This repo was tested against commit 2adb63f.
git clone https://github.com/piddnad/DDColor.git ddcolor-src

# PyTorch path
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt

# Core ML path (optional, Mac only). Use a Python from 3.9 to 3.13.
/usr/bin/python3 -m venv venv-coreml
venv-coreml/bin/python -m pip install -r requirements-coreml.txt
```

The first run downloads the model weights (`piddnad/ddcolor_modelscope`, about 0.9 GB) from Hugging Face into its cache. To keep them in a folder of your choosing and run offline afterwards:

```bash
export HF_HUB_CACHE=/path/to/hf-cache     # where the weights live
export HF_HUB_OFFLINE=1                   # after the first download: never touch the network
```

---

## Use

### One video, PyTorch

```bash
venv/bin/python -I colorize.py ddcolor-src "old film.mp4" work/raw.mp4
./smooth.sh "old film.mp4" work/raw.mp4 "old film (Colorized).mp4"
```

An optional fourth argument picks the model: `ddcolor_modelscope` (default, used for every result here), `ddcolor_paper`, `ddcolor_artistic` or `ddcolor_paper_tiny`.

### One video, Core ML

```bash
# once: convert (writes a .mlpackage; fp16 is the default, fp32 is optional)
HF_HUB_CACHE=/path/to/hf-cache venv-coreml/bin/python -I coreml/convert.py ddcolor-src coreml/DDColor.mlpackage

# then
venv-coreml/bin/python -I colorize-coreml.py coreml/DDColor.mlpackage "old film.mp4" work/raw.mp4 CPU_AND_GPU
./smooth.sh "old film.mp4" work/raw.mp4 "old film (Colorized).mp4"
```

The last argument picks the hardware: `CPU_AND_GPU` (default, recommended), `ALL` (allows the Neural Engine: faster, but colors fade, see below), `CPU_AND_NE` or `CPU_ONLY`.

### A whole list

```bash
# list.txt: one path per line, relative to LIBRARY
LIBRARY="/Volumes/Videos" DDCOLOR_SRC="$PWD/ddcolor-src" ./batch.sh list.txt
ENGINE=coreml LIBRARY="/Volumes/Videos" ./batch.sh list.txt            # Core ML instead
```

Each result is written as `<name> (Colorized).mp4` beside its original. **Resumable:** a title that already has its `(Colorized)` copy is skipped, so re-running after a stop picks up where it left off. **To stop after the current title**, empty `list.txt`; the loop reads it line by line. Run it with `nohup` (or in `tmux`) for long batches, and see the `PATH` note under Troubleshooting.

### Measure your own machine

```bash
HF_HUB_CACHE=/path/to/hf-cache venv-coreml/bin/python -I coreml/bench.py ddcolor-src clip.mp4 100 coreml/DDColor.mlpackage
```

This prints frames per second for PyTorch and for each Core ML setting, the mean and worst per-pixel difference from PyTorch, and writes side-by-side sample frames to `coreml/`.

---

## Measured results

Apple M1 Mac mini (8-core GPU, 16 GB), macOS 27.0.1. The same 100 frames (704 × 576, a 1960s puppet series) were used for every row.

| Engine | Speed | Difference from PyTorch (0–255 per pixel) | Looks |
|---|---|---|---|
| PyTorch 2.7.0, Apple GPU (MPS) | **1.86 fps** | reference | reference |
| Core ML FP16, `CPU_AND_GPU` | **2.72 fps** (1.5×) | mean 3.34, worst 76 | indistinguishable |
| Core ML FP16, `ALL` (Neural Engine) | **4.58 fps** (2.5×) | mean 9.27, worst 119 | **visibly washed out**, faces grayer |
| Core ML FP32, `ALL` | 2.23 fps | mean 3.32, worst 79 | indistinguishable |
| Core ML FP32, `CPU_AND_GPU` | 2.22 fps | mean 3.44, worst 86 | indistinguishable |

End to end on the same M1, the Core ML script ran a 140-frame clip at **2.63 fps** (`CPU_AND_GPU`). Full batches with PyTorch ran at 1.77–1.78 fps.

**What that means for running time** (at 2.7 fps with Core ML on the GPU):

| Length | PyTorch on an M1 GPU | Core ML on an M1 GPU |
|---|---|---|
| 25-minute episode (≈37,500 frames) | ≈ 5 h 50 min | ≈ 3 h 50 min |
| 90-minute film at 24 fps (≈129,600 frames) | ≈ 20 h | ≈ 13 h |

**Why the Neural Engine fades the color:** it computes in 16-bit floating point. DDColor's color outputs reach about ±260 before they are turned into a picture, and somewhere in the network the reduced precision costs saturation. The GPU runs the same FP16 model with results that match PyTorch closely. This explanation is Claude's inference from the measurements above, not something Apple or the DDColor authors have stated.

---

## How it works, in detail

- **Color space.** Each frame is converted to Lab (L = lightness, a and b = color). Only L goes into the model, as a gray RGB image at 512 × 512. The model returns a and b at 512 × 512. These are enlarged to the frame's size with nearest-neighbour scaling, exactly as DDColor's own `ColorizationPipeline` does (`F.interpolate`'s default mode), and joined to the frame's **full-resolution** L. The picture's detail and brightness are always the original's. Only the color comes from the model.
- **Why 512.** It is the input size DDColor's pipeline uses, and the Core ML model is traced at that fixed size (1 × 3 × 512 × 512).
- **Encoding.** OpenCV reads frames. Colorized frames are piped as raw BGR into FFmpeg (`libx264`, CRF 18, `yuv420p`). The input's audio is mapped in, and the input's **display aspect ratio** is carried over with `-aspect`, so anamorphic standard-definition video keeps its shape.
- **Smoothing (`smooth.sh`).** A per-frame colorizer flickers, because each frame is guessed alone. The filter graph takes **Y (brightness) from the original** black-and-white video and **U and V (color) from the colorized video averaged over 7 frames** (`tmix=frames=7`), then merges them (`mergeplanes`). Brightness stays sharp and true to the original, and color settles down.
- **Device choice (`colorize.py`).** NVIDIA CUDA if present, else Apple MPS, else CPU.
- **Conversion (`coreml/convert.py`).** `torch.jit.trace` on the Hugging Face model, then `coremltools.convert` to an ML Program (`.mlpackage`), minimum target macOS 14, FP16 by default. The tracer warns about `if x.shape[1] == 3:` in DDColor's `model.py`. That branch only checks the input has 3 channels, which is always true here, so baking it in as a constant is correct.

### Design decisions, and why

| Decision | Why |
|---|---|
| DDColor | Open source (Apache-2.0), strong published results on photos (ICCV 2023), small enough to run on a laptop-class GPU, weights on Hugging Face. |
| Frame by frame, then temporal smoothing | DDColor is an image model. Averaging only the color over 7 frames removes most flicker without blurring the picture, because brightness is never averaged. |
| Original L at full size | Keeps every bit of the original's detail. The model works at 512 × 512, and only its color is enlarged. |
| `ddcolor_modelscope` weights | The weights the DDColor authors' ModelScope release uses. Every result here was made with them. |
| Core ML on `CPU_AND_GPU` as the default | 1.5× faster than PyTorch on the same Mac with no visible change. The faster Neural Engine setting fades colors. |
| A separate `venv-coreml` | coremltools 9.0 supports Python up to 3.13 and was tested up to torch 2.7.0. The PyTorch path runs on newer versions, so the two are kept apart. |
| Originals never modified; resumable batches | Long jobs get interrupted. Skipping finished titles makes a restart safe and cheap. |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERROR: No matching distribution found for coremltools` | Your Python is 3.14 or newer. Make `venv-coreml` with Python 3.9–3.13 (macOS's `/usr/bin/python3` works). |
| `Torch version 2.x has not been tested with coremltools` | A warning, not an error. Use `torch==2.7.0` (in `requirements-coreml.txt`) to match what coremltools 9.0 was tested with. |
| `TracerWarning: Converting a tensor to a Python boolean…` during conversion | Expected. See "Conversion" above. |
| `max difference vs PyTorch on a random input` looks huge (tens to ~190) | Random noise is not a picture and exaggerates it. Judge with `coreml/bench.py` on real frames. |
| Colors look washed out or gray with Core ML | You ran with `ALL` or `CPU_AND_NE` (Neural Engine) and the FP16 model. Use `CPU_AND_GPU`. |
| `ffmpeg: command not found` inside a `nohup`, `launchd` or `cron` job, though it works in Terminal | Background jobs don't load your shell's `PATH`. Add `export PATH=/opt/homebrew/bin:$PATH` (Homebrew on Apple silicon) at the top of the job, or use full paths. This bit the original batch once. |
| Hugging Face download errors, or a request for a token | Set `HF_HUB_CACHE` to a folder you can write. The warning about unauthenticated requests is harmless. After the weights are down, set `HF_HUB_OFFLINE=1`. |
| `ModuleNotFoundError: ddcolor` | The first argument must be the DDColor clone's folder (the one containing the `ddcolor/` package). |
| Output is a little shorter than the input (a few frames) | `smooth.sh` uses FFmpeg's `-shortest` with the copied audio. The last few frames, about 0.1 s, can be lost. Remove `-shortest` from `smooth.sh` if that matters. |
| Output has no sound | The input has no audio stream. `-map 1:a?` makes audio optional. |
| Squeezed or stretched picture | The input's display aspect ratio is copied with `-aspect`. If ffprobe reports none, `4:3` is used. Change the fallback in the scripts for widescreen sources. |
| Very slow (well under 1 fps) | You are on the CPU. Check that `torch.backends.mps.is_available()` (Mac) or `torch.cuda.is_available()` (NVIDIA) is `True`. |
| Out of memory on a small GPU | Close other GPU apps. The model runs one 512 × 512 frame at a time, so memory use does not grow with video length. |
| A batch title shows `FAILED colorize` | Run `colorize.py` on that file alone to see the full error. `batch.sh` only prints lines containing "error". |
| Interlaced sources (some DVD and TV captures) show combing | Deinterlace first, e.g. `ffmpeg -i in.mkv -vf bwdif -c:a copy out.mkv`, then colorize. |

---

## Research notes: Apple M5 and PyTorch

Gathered 2026-10-27 while deciding whether new hardware would speed this up. What each source reported, as reported:

- **Apple** says each M5 GPU core has a "Neural Accelerator", with over four times the peak GPU compute for AI compared with M4 and up to 30% faster graphics on the base chip, as summarized by [MacRumors' M4 vs. M5 guide](https://www.macrumors.com/guide/m4-vs-m5-chip/) and [Creative Strategies](https://creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/).
- **The accelerators must be programmed for.** They are reached through Metal 4's tensor APIs, not automatically ([BaseRT paper, arXiv](https://arxiv.org/pdf/2607.19438); Apple's [WWDC 2026 session 330](https://developer.apple.com/videos/play/wwdc2026/330/) on Metal tensors).
- **When used, they help.** An MLX test reported a 3.65× faster prompt phase for a language model ([Hacker News discussion](https://news.ycombinator.com/item?id=49441396)). An [LM Studio bug report](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040) shows llama.cpp losing 2–3× when the tensor path is off on M5.
- **PyTorch:** no confirmation found that its MPS backend uses them. The nearest issue ([pytorch/pytorch #157660](https://github.com/pytorch/pytorch/issues/157660)) asked whether Metal 4 would help and was closed.
- **Without the accelerators,** one Core ML image test went from about 7 s on M4 to 4.5 s on M5 ([tech-insider.org](https://tech-insider.org/apple-m5-chip-benchmarks-a-new-standard-for-personal-computing/)).

**Conclusion (Claude's inference from the above):** run unchanged through PyTorch, an M5 should give a normal generational gain, not the headline 4×. The Core ML path here is the one placed to benefit, if and when Core ML runs this model on the new accelerators. That has not been measured. If you have an M5, `coreml/bench.py` will show it in two minutes, and a pull request with the numbers is welcome.

---

## Credits and licenses

This repository's scripts and documents: **MIT**, see `LICENSE`. **Michael Fluharty's copyright claims none of the work below.** Each item belongs to its authors under its own license. Nothing below is copied into this repository; it is downloaded by you at setup.

| Work | Authors / owner | License | Used for |
|---|---|---|---|
| [DDColor](https://github.com/piddnad/DDColor) (code) | Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li, Xuansong Xie | Apache-2.0 | The colorization model and its pipeline, which our pre- and post-processing reproduce |
| [`piddnad/ddcolor_modelscope`](https://huggingface.co/piddnad/ddcolor_modelscope) (weights) | DDColor authors; originally released on ModelScope as [`damo/cv_ddcolor_image-colorization`](https://www.modelscope.cn/models/damo/cv_ddcolor_image-colorization/summary) (Alibaba DAMO Academy) | Apache-2.0 | The trained model |
| [PyTorch](https://pytorch.org) and torchvision | PyTorch Foundation and contributors | BSD-3-Clause | Running the model; tracing it for conversion |
| [coremltools](https://github.com/apple/coremltools) | Apple Inc. | BSD-3-Clause | Converting to Core ML; running Core ML from Python |
| [timm](https://github.com/huggingface/pytorch-image-models) | Ross Wightman and contributors | Apache-2.0 | Network building blocks DDColor uses |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub), [safetensors](https://github.com/huggingface/safetensors) | Hugging Face | Apache-2.0 | Downloading and loading the weights |
| [OpenCV](https://opencv.org) (`opencv-python-headless`) | OpenCV team and contributors | Apache-2.0 | Reading frames; color-space conversion; resizing |
| [NumPy](https://numpy.org) | NumPy developers | BSD-3-Clause | Arrays |
| [FFmpeg](https://ffmpeg.org) with [x264](https://www.videolan.org/developers/x264.html) | FFmpeg developers; VideoLAN | LGPL/GPL (x264: GPL) | Encoding, audio, smoothing filter. Installed separately, not distributed here. |

If you publish work made with DDColor, cite its authors:

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```

**A note on what you colorize:** check that you have the right to make and share a colorized copy of a film or show. Many older works are in the public domain, but not all, and some public-domain works have later restorations that are not.

---

## Repository layout

```
colorize.py           PyTorch: one video → raw colorized video
colorize-coreml.py    Core ML: one video → raw colorized video (Mac)
smooth.sh             original brightness + 7-frame color average → finished video
batch.sh              a list of videos, resumable, results beside the originals
coreml/convert.py     DDColor (PyTorch) → Core ML .mlpackage
coreml/bench.py       speed and picture difference, PyTorch vs Core ML, on real frames
requirements.txt      PyTorch path (tested on Python 3.14.9)
requirements-coreml.txt  Core ML path (Python 3.9–3.13)
```
