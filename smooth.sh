#!/bin/bash
# Steady the color: keep the ORIGINAL black-and-white picture's brightness, and average the colorized
# version's color over 7 frames, so the per-frame colorizer's flicker settles down.
# Usage: smooth.sh ORIGINAL COLORIZED OUTPUT
#   ORIGINAL   the black-and-white source           COLORIZED  the output of colorize.py / colorize-coreml.py
#   OUTPUT     the finished .mp4 (H.264 CRF 18, COLORIZED's audio copied, ORIGINAL's display aspect ratio)
set -euo pipefail
orig="$1"; raw="$2"; out="$3"
dar=$(ffprobe -v error -select_streams v:0 -show_entries stream=display_aspect_ratio -of csv=p=0 "$orig")
ffmpeg -nostdin -v error -y -i "$orig" -i "$raw" -filter_complex \
  "[0:v]format=yuv444p,extractplanes=y[y];[1:v]format=yuv444p,tmix=frames=7,extractplanes=u+v[u][v];[y][u][v]mergeplanes=0x001020:yuv444p,format=yuv420p[o]" \
  -map "[o]" -map "1:a?" -c:v libx264 -crf 18 -aspect "${dar:-4:3}" -c:a copy -shortest "$out"
