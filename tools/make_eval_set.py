#!/usr/bin/env python3
"""Generate an evaluation set that isolates *why* a demosaic algorithm wins or loses.

A single test image gives you one number and no explanation. These images vary two
axes independently:

  * spatial frequency  (lo / mid / hi)  -- how close the detail sits to Nyquist
  * inter-channel correlation           -- grey, photographic, anti-correlated

That second axis is the one that matters for Malvar-He-Cutler. MHC corrects the
interpolation of one channel using the Laplacian of the channel measured at that
site, which is only valid when luminance detail is shared across R, G and B. Feed
it channels that vary independently and the "correction" injects error instead of
removing it. Bilinear has no such assumption, so it degrades gracefully.

Note that a *grey* image cannot detect a swapped horizontal/vertical kernel: the
swap exchanges the R and B estimates, and when R_true == B_true the aggregate
error is unchanged. Use `corr` (channels at different scales) for that.

    python3 tools/make_eval_set.py
"""

import argparse
import os

import numpy as np
from PIL import Image


def save(outdir, name, arr):
    path = os.path.join(outdir, f"{name}.png")
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
    return path


def band_limited(h, w, scale, rng):
    """Isotropic noise low-passed to a given spatial scale. Photographic content is
    dominated by low frequencies; synthetic charts by high ones."""
    f = np.fft.fftshift(np.fft.fft2(rng.normal(size=(h, w))))
    fy, fx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2]
    f *= np.exp(-(np.sqrt(fx ** 2 + fy ** 2) / scale) ** 2)
    out = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
    out -= out.min()
    out /= max(out.max(), 1e-12)
    return 0.15 + 0.7 * out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    h = w = a.size
    rng = np.random.default_rng(a.seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    for tag, scale in (("lo", 12), ("mid", 40), ("hi", 120)):
        L = band_limited(h, w, scale, rng)

        # Pure luminance: MHC's premise holds exactly.
        save(a.outdir, f"gray_{tag}", np.dstack([L, L, L]))

        # Shared luminance detail, slowly varying chroma -- photographic.
        ca = 0.85 + 0.25 * np.sin(xx / 120.0)
        cb = 0.85 + 0.25 * np.cos(yy / 140.0)
        save(a.outdir, f"photo_{tag}", np.dstack([L * ca, L, L * cb]))

        # Anti-correlated: MHC's premise deliberately violated.
        save(a.outdir, f"anti_{tag}", np.dstack([L, 1.0 - L, L * 0.5 + 0.25]))

    # Constant channel ratios at different scales. This is the image that detects
    # a swapped H/V kernel -- ~12 dB drop if the orientation is wrong.
    L = band_limited(h, w, 40, rng)
    save(a.outdir, "corr", np.dstack([L * 1.00, L * 0.90, L * 0.80]))

    print(f"wrote evaluation set to {a.outdir}/  ({h}x{w})")


if __name__ == "__main__":
    main()
