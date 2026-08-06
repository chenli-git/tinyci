#!/bin/bash
# End-to-end check: generate a test scene, mosaic it, run the pipeline, score it.
# Run after any pipeline change -- PSNR is the regression test.
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${TINYCI_BUILD:-$SRC/build}"
cd "$SRC"
mkdir -p data

if [ ! -f data/scene.png ]; then
    python3 tools/make_test_scene.py -o data/scene.png
fi

python3 tools/bayer.py mosaic data/scene.png --cfa RGGB --black 512 --white 65535 >/dev/null
"$BUILD"/tinyci data/scene_bayer.pgm data/scene_out.png \
    --cfa RGGB --black 512 --white 65535 "$@"
echo
python3 tools/bayer.py psnr data/scene_out.png data/scene_gt.png
