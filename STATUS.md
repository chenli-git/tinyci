# Status

Last updated: 2026-08-06. Milestone **M0 (host reference pipeline) — scaffold complete.**

---

## 1. Environment

Verified working end to end.

| | |
|---|---|
| Host | Windows 11, WSL2 (kernel 6.6.87.2) |
| Distro | Ubuntu 24.04 LTS — pinned, because newer Ubuntu ships GCC 15 and `nvcc` rejects host compilers it doesn't support |
| GPU | RTX 5060, Blackwell, `sm_120`, 30 SMs, 8 GB, 48 KiB shared/block |
| Driver | 581.95 (Windows-side; WSL reaches the GPU via `/dev/dxg`) |
| CUDA | 13.0.88 at `/usr/local/cuda` — toolkit only, **no Linux driver package** |
| Host compiler | GCC 13.3.0 |
| Build | CMake 3.28.3 + Ninja 1.11.1, in-tree to `./build` |
| Profiler | Nsight Compute 2025.3.1 |

Two environment details cost real time and are worth recording:

**`nvcc` is not on `PATH` for non-interactive shells.** The export lives in `~/.bashrc`, which bash
only sources when interactive. Scripts must export it themselves — `build.sh` does. A script that
"can't find nvcc" while your terminal can is this, not a broken install.

**In-tree builds on a Windows drive need the DrvFs `metadata` mount option.** Without it every file
reports mode 777 owned by root and `chmod` silently no-ops, so CMake's `configure_file()` — which
copies a file *and its mode bits* — fails with "Operation not permitted" and configure aborts. Fixed
in `/etc/wsl.conf`; see README.

## 2. Measured baseline

```
$ ./build/bwtest
GPU         : NVIDIA GeForce RTX 5060
Compute cap : sm_120
SMs         : 30
Global mem  : 7.96 GiB
Shared/block: 48 KiB

MEASURED PEAK BW : 377.5 GB/s  (2.844 ms/iter)
```

**377.5 GB/s is the denominator for every performance claim in this project** — roughly 84% of the
~448 GB/s theoretical figure (128-bit GDDR7 @ 28 Gbps). A pure copy kernel is the practical ceiling
for anything bandwidth-bound, so quoting against it is honest in a way that quoting the spec sheet
is not.

Current host pipeline, 1024×768 (0.8 MP), single-threaded:

```
  linearize + WB              2.14 ms
  demosaic (placeholder)      5.36 ms
  camera -> sRGB              0.00 ms   (not implemented)
  tone map                    0.00 ms   (not implemented)
  unsharp                     0.00 ms   (not implemented)
  encode sRGB8                9.34 ms
  ------------------------------------
  pipeline total             16.83 ms
```

These are not yet meaningful as a baseline — three stages do nothing. They become the M1 comparison
once M0 is complete.

## 3. What is implemented

**Infrastructure — done.**

- `Image<T>`: owning, interleaved 2D buffer. Rule of Zero (`std::vector` owns the memory, so copy,
  move and destruction are all correct with no hand-written special members). Interleaved rather than
  planar because that is what the GPU wants later — one coalesced 16-byte load per thread for RGBA
  float.
- `cfaColorAt()`: CFA pattern lookup for all four Bayer layouts.
- PNG I/O via stb; hand-rolled 16-bit PGM reader/writer for mosaics. PGM because it is trivially
  parseable and carries no colour-management metadata to misinterpret — sensor data is raw counts, not
  colour.
- CLI driver with per-stage wall-clock timing. Measurement is scaffolding here, not an afterthought.
- `tools/bayer.py`: RGB → synthetic Bayer mosaic + ground truth; PSNR scoring.
- `tools/make_test_scene.py`: synthetic chart, deliberately hostile to bilinear — a radial zone plate
  sweeping to Nyquist plus saturated edges at every angle. A landscape photo would hide the artifacts
  this is designed to expose.
- `bench/bwtest.cu`: peak-bandwidth benchmark.

**Pipeline stages — two of six.**

| Stage | State | Note |
|---|---|---|
| `linearizeAndWhiteBalance` | **Done** | Worked example. Read first. |
| `demosaicBilinear` | TODO | Currently a grayscale passthrough placeholder |
| `demosaicMHC` | TODO | Stretch |
| `cameraToSRGB` | TODO | XYZ→sRGB matrix constant provided |
| `toneMap` | TODO | |
| `unsharpMask` | TODO | |
| `encodeSRGB8` | **Done** | Worked example |

The two implemented stages are the first and last, so the pipeline runs end to end. Each TODO carries
a spec comment giving the algorithm *and the reasoning* — not just what to compute but why the design
is that way.

### Design decisions already baked in

**White balance happens on the mosaic, before demosaic.** Each site still holds exactly one known
colour at that point. Do it after demosaic and you have already interpolated across channels carrying
mismatched gains, baking a colour cast into every edge.

**Everything before the final encode is linear light.** Applying a colour matrix or sharpening to
gamma-encoded values is a classic bug — subtly wrong saturation and dark halos, both hard to spot by
eye and easy to ship.

**Stage signatures encode the fusion boundary.** Stages that change pixel *shape* return a new image;
stages that only transform values operate in place. The in-place ones are exactly the pointwise
stages — the ones M4 will fuse into a single GPU pass. Making that distinction visible in the header
means the fusion boundary is already obvious before any GPU work starts.

**The sRGB OETF is not gamma 1/2.2.** It has a linear segment near black (a pure power function has
infinite slope at 0, wasting code values and amplifying shadow noise) and uses 1/2.4 with an offset
above it. The composite approximates 2.2 overall, which is why the two get conflated.

## 4. Correctness

There is no unit-test framework. **PSNR against ground truth is the regression test**, printed by
`./smoke.sh` after every change.

| State | PSNR |
|---|---|
| grayscale placeholder | **11.08 dB** ← current |
| working bilinear demosaic | expect high 20s – low 30s on this chart |
| Malvar-He-Cutler | typically +2–5 dB over bilinear |

Numbers run lower than a natural photograph would score, because the test chart is adversarial by
design. That is the point: artifacts you can see are artifacts you can fix, and the bilinear-vs-MHC
comparison on this target is the clearest evidence of what the algorithm buys.

## 5. Next

Immediate — finish M0:

1. `demosaicBilinear` — replace the passthrough. Green is a quincunx (average 4 edge-adjacent);
   red/blue sit on a square lattice with three cases (horizontal pair, vertical pair, diagonal four).
2. `cameraToSRGB` — two chained 3×3 matrices, in linear light.
3. `toneMap` — exposure, then a curve. Reinhard first; highlight recovery after.
4. `unsharpMask` — separable Gaussian. At σ=2 that is 26 taps per pixel instead of 169.
5. `demosaicMHC` — gradient-corrected, if time allows.

Then M1: port each stage to a CUDA kernel, verify against these host results, and establish the naive
GPU baseline that M2–M6 improve on.

## 6. Known gaps

- `RawParams::camToXYZ` defaults to identity. Real cameras ship a measured matrix per illuminant;
  synthetic test data doesn't need one, but the field exists so the stage is honest about what it
  needs.
- No highlight recovery: when one channel clips and others don't, hue shifts (blown skies go cyan).
- Single-threaded host code. Intentional — M0 is the correctness oracle, not a performance target.
- `inotify` is unreliable on DrvFs, so CMake Tools and clangd can serve stale results. If a build
  result looks impossible, `rm -rf build && ./build.sh` before debugging the code.
