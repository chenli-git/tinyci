#!/usr/bin/env python3
"""Build synthetic Bayer test data, and score pipeline output against ground truth.

Synthesising a mosaic from an ordinary RGB image beats using a real RAW file here:
you keep the original as exact ground truth, so demosaic quality becomes a number
instead of an opinion. It also skips a RAW parser, which would prove nothing about
GPU or framework skill.

    python3 tools/bayer.py mosaic scene.png --cfa RGGB
    python3 tools/bayer.py psnr data/scene_out.png data/scene_gt.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# 2x2 CFA layouts as colour indices (0=R, 1=G, 2=B), [row][col] from pixel (0,0).
PATTERNS = {
    "RGGB": [[0, 1], [1, 2]],
    "BGGR": [[2, 1], [1, 0]],
    "GRBG": [[1, 0], [2, 1]],
    "GBRG": [[1, 2], [0, 1]],
}


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def cmd_mosaic(a):
    rgb = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float64) / 255.0

    # Even dimensions, so the 2x2 CFA tiles cleanly and the pattern at (0,0)
    # stays valid all the way to the bottom-right.
    h, w = rgb.shape[0] & ~1, rgb.shape[1] & ~1
    rgb = rgb[:h, :w]

    # A sensor responds linearly to light; a PNG is sRGB-encoded. Undo the
    # encoding so the synthetic mosaic behaves like real sensor data -- otherwise
    # you are demosaicking gamma-space values and every later step is subtly wrong.
    lin = srgb_to_linear(rgb)

    pat = np.array(PATTERNS[a.cfa])
    idx = pat[np.arange(h)[:, None] % 2, np.arange(w)[None, :] % 2]
    mosaic = np.take_along_axis(lin, idx[:, :, None], axis=2)[:, :, 0]

    codes = a.black + mosaic * (a.white - a.black)
    codes = np.clip(np.rint(codes), 0, 65535).astype(">u2")  # PGM is big-endian

    stem = os.path.splitext(os.path.basename(a.input))[0]
    os.makedirs(a.outdir, exist_ok=True)

    pgm = os.path.join(a.outdir, f"{stem}_bayer.pgm")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{w} {h}\n65535\n".encode())
        f.write(codes.tobytes())

    gt = os.path.join(a.outdir, f"{stem}_gt.png")
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(gt)

    print(f"mosaic : {pgm}   {w}x{h}  {a.cfa}  black={a.black} white={a.white}")
    print(f"truth  : {gt}")
    print()
    print("next:")
    print(f"  ./build/tinyci {pgm} {a.outdir}/{stem}_out.png \\")
    print(f"      --cfa {a.cfa} --black {a.black} --white {a.white}")
    print(f"  python3 tools/bayer.py psnr {a.outdir}/{stem}_out.png {gt}")


def cmd_psnr(a):
    x = np.asarray(Image.open(a.a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(a.b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        sys.exit(f"shape mismatch: {x.shape} vs {y.shape}")

    # Ignore a 4-pixel border: demosaic quality at the very edge is dominated by
    # the boundary policy, not the algorithm, and it would skew the comparison.
    b = 4
    x, y = x[b:-b, b:-b], y[b:-b, b:-b]

    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        print("PSNR: identical")
        return
    print(f"PSNR: {10 * np.log10(255.0 ** 2 / mse):.2f} dB   (MSE {mse:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mosaic", help="RGB image -> Bayer PGM + ground-truth PNG")
    m.add_argument("input")
    m.add_argument("--cfa", default="RGGB", choices=sorted(PATTERNS))
    m.add_argument("--black", type=float, default=0.0)
    m.add_argument("--white", type=float, default=65535.0)
    m.add_argument("--outdir", default="data")
    m.set_defaults(func=cmd_mosaic)

    p = sub.add_parser("psnr", help="compare two images")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_psnr)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
