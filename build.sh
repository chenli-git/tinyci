#!/bin/bash
# Configure + build, in-tree.
#
# This works on a Windows drive only because /etc/wsl.conf enables the DrvFs
# "metadata" automount option -- without it CMake's configure_file() fails with
# "Operation not permitted". See ../SETUP.md.
set -e
export PATH=/usr/local/cuda/bin:$PATH

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${TINYCI_BUILD:-$SRC/build}"

cmake -S "$SRC" -B "$BUILD" -G Ninja "$@"
cmake --build "$BUILD" -j
echo
echo "binaries: $BUILD/tinyci  $BUILD/bwtest"
