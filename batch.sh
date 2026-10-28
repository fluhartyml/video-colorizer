#!/bin/bash
# Colorize every video listed in a text file, one path per line (relative to LIBRARY), then smooth it.
# Each result is written next to its original as "<name> (Colorized).mp4". Originals are never touched.
# Resumable: a title whose "(Colorized)" copy already exists is skipped, so re-running picks up where it stopped.
#
# Usage: LIBRARY=/path/to/videos DDCOLOR_SRC=/path/to/DDColor ./batch.sh list.txt
# Optional: ENGINE=coreml MLPACKAGE=coreml/DDColor.mlpackage UNITS=CPU_AND_GPU  (default ENGINE=pytorch)
#           PYTHON=venv/bin/python (PyTorch)   PYTHON_COREML=venv-coreml/bin/python (Core ML)
set -u
list="$1"
here="$(cd "$(dirname "$0")" && pwd)"
: "${LIBRARY:?set LIBRARY to the folder that the paths in the list are relative to}"
ENGINE="${ENGINE:-pytorch}"
PYTHON="${PYTHON:-$here/venv/bin/python}"
PYTHON_COREML="${PYTHON_COREML:-$here/venv-coreml/bin/python}"
work="$here/work"; mkdir -p "$work"
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  src="$LIBRARY/$rel"; out="${src%.*} (Colorized).mp4"
  [ -f "$out" ] && continue
  [ -f "$src" ] || { echo "$(date '+%F %T') MISSING $rel"; continue; }
  echo "$(date '+%F %T') START $rel"
  rm -f "$work/raw.mp4" "$work/smooth.mp4"
  if [ "$ENGINE" = coreml ]; then
    "$PYTHON_COREML" -I "$here/colorize-coreml.py" "${MLPACKAGE:-$here/coreml/DDColor.mlpackage}" "$src" "$work/raw.mp4" "${UNITS:-CPU_AND_GPU}" 2>&1 | grep -E "^DONE|Error|error"
  else
    : "${DDCOLOR_SRC:?set DDCOLOR_SRC to your DDColor clone}"
    "$PYTHON" -I "$here/colorize.py" "$DDCOLOR_SRC" "$src" "$work/raw.mp4" 2>&1 | grep -E "^DONE|Error|error"
  fi
  [ -s "$work/raw.mp4" ] || { echo "$(date '+%F %T') FAILED colorize $rel"; continue; }
  "$here/smooth.sh" "$src" "$work/raw.mp4" "$work/smooth.mp4" \
    && mv "$work/smooth.mp4" "$out" && echo "$(date '+%F %T') DONE $rel" \
    || echo "$(date '+%F %T') FAILED smooth $rel"
done < "$list"
echo "$(date '+%F %T') BATCH FINISHED"
