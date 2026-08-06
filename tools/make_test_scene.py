#!/usr/bin/env python3
"""Generate a synthetic test scene for demosaic work.

Deliberately hostile to bilinear interpolation: high-contrast edges at many
angles, a radial zone plate that sweeps spatial frequency up to Nyquist, and
saturated primaries. Zipper artifacts and colour fringing show up here
immediately, where a photograph of a landscape would hide them.
"""

import argparse
import numpy as np
from PIL import Image


def build(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))

    # Radial zone plate: frequency rises with radius, reaching Nyquist at the
    # edges. This is where demosaic algorithms produce false colour.
    cx, cy = w * 0.30, h * 0.5
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (0.16 * w * w)
    zp = 0.5 + 0.5 * np.cos(np.pi * r2 * 28.0)
    m = xx < w * 0.55
    for c in range(3):
        img[:, :, c] = np.where(m, zp, img[:, :, c])

    # Angled colour wedges: edges at every orientation, saturated primaries.
    for i, ang in enumerate(np.linspace(0, np.pi, 9)[:-1]):
        band = ((xx - w * 0.72) * np.cos(ang) + (yy - h * 0.5) * np.sin(ang))
        stripe = (np.sin(band * 0.55) > 0).astype(np.float64)
        sel = (xx >= w * 0.55) & (yy >= h * i / 8.0) & (yy < h * (i + 1) / 8.0)
        col = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
               (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0.25, 0.75)][i]
        for c in range(3):
            img[:, :, c] = np.where(sel, stripe * col[c], img[:, :, c])

    # A flat mid-grey patch: any colour cast from a pipeline bug shows here.
    img[h - h // 8:, :w // 8, :] = 0.5
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/scene.png")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    a = ap.parse_args()

    arr = (build(a.width, a.height) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(a.out)
    print(f"wrote {a.out}  ({a.width}x{a.height})")


if __name__ == "__main__":
    main()
