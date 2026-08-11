# Status

Last updated: 2026-08-11.

**M0 (host reference pipeline) — COMPLETE.** All six stages implemented and validated.

**M1 (CUDA) — the measurements are done.** Peak bandwidth established, kernel fusion measured, three
isolating experiments run, and `demosaicBilinear` ported to a 2-D kernel producing **bit-identical**
output at 88% of peak bandwidth. The full pipeline does not yet run on the GPU; the CUDA work lives in
standalone benchmarks under `bench/`.

**Not built:** the graph (M2), tiling (M3), shared-memory kernels (M5), streams (M6). Designed, and the
measurements motivate them — see [docs/study-plan.md](docs/study-plan.md) §5–7.

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

Complete host pipeline, 1024×768 (0.8 MP), single-threaded, σ=1.5 amount=0.8:

```
  linearize + WB              2.19 ms
  demosaic (bilinear)        11.01 ms      (MHC: 22.62 ms)
  camera -> sRGB              0.47 ms
  tone map                    1.39 ms
  unsharp                    48.54 ms   <- bottleneck
  encode sRGB8                9.81 ms
  ------------------------------------
  pipeline total             73.42 ms
```

This is the M1 baseline. Two observations already worth carrying into the GPU port:

**`cameraToSRGB` is memory-bound on the CPU.** 0.79 MP × 3ch × 4 B = 9.4 MB read and written, so
~18.9 MB in 0.47 ms ≈ **40 GB/s** — at single-core memory bandwidth. Its compute side is 9 MACs ×
0.79 M ≈ 7 MFLOP, about 15 GFLOP/s, far under what one core does with SIMD. The project's central
claim shows up before the GPU is even involved.

**Separable convolution is not scaling linearly.** Measured unsharp against σ:

| σ | N | ms |
|---|---|---|
| 1 | 7 | 29.8 |
| 2 | 13 | 70.0 |
| 4 | 25 | 229.7 |
| 8 | 49 | 490.3 |

7× the taps costs 16.5× the time, roughly N^1.4. The vertical pass is responsible: consecutive taps
are `width × 3` floats apart (~12 KB), so each is its own cache line and the working set leaves L1 as
the radius grows. The horizontal pass reads adjacent memory and does not suffer. This is exactly the
problem shared-memory tiling solves on the GPU, and it is now a measured motivation rather than a
theoretical one.

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

**Pipeline stages — six of six.**

| Stage | Validated by |
|---|---|
| `linearizeAndWhiteBalance` | Flat field exact across all four CFA layouts |
| `demosaicBilinear` | Flat field identical (4 layouts); gradient 62.4 dB; mid-frequency 48.7 dB |
| `demosaicMHC` | DC gain exactly 1; +5.0 to +8.4 dB over bilinear on luminance content; H/V kernel swap collapses `corr` 35.2 → 22.9 dB, confirming orientation |
| `cameraToSRGB` | Every PSNR unchanged — the matrix chain round-trips to identity for the synthetic sRGB camera |
| `toneMap` | W=1 is algebraically identity, PSNR unchanged; W=2 and W=4 each map themselves to exactly 1.0 |
| `unsharpMask` | Cross-validated against an independent numpy implementation: mean 0.16 LSB, max 6 LSB |
| `encodeSRGB8` | Worked example |

### MHC does not beat bilinear on every image, and that is the point

| Image | bilinear | MHC | Δ |
|---|---|---|---|
| `gray_hi` | 31.93 | 40.30 | **+8.37** |
| `corr` | 33.02 | 35.17 | **+2.15** |
| `photo_mid` | 47.94 | 46.47 | −1.47 |
| `anti` | 33.85 | 28.59 | −5.26 |
| `scene` (adversarial chart) | 18.97 | 18.28 | −0.69 |

MHC corrects one channel using the Laplacian of the channel measured at that site, which is valid
only when luminance detail is **shared across R, G and B**. Feed it channels that vary independently
— saturated synthetic charts, anti-correlated content — and the correction injects error instead of
removing it. Bilinear assumes nothing and degrades gracefully.

Real photographs are luminance-dominated, which is why MHC wins on standard sets like Kodak. The
`scene` chart here is deliberately adversarial: its colour wedges are saturated primaries with nearly
independent channels. `tools/make_eval_set.py` varies spatial frequency and inter-channel correlation
independently so the boundary is measurable rather than assumed.

One methodological note recorded there: **a grey image cannot detect a swapped H/V kernel**, because
the swap merely exchanges the R and B estimates and `R_true == B_true` leaves aggregate error
unchanged. `corr` — constant but *unequal* channel ratios — is what catches it.

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
`./smoke.sh`. Generate the full evaluation set with `python3 tools/make_eval_set.py`.

Three properties are worth testing directly, because each catches a different class of bug:

- **Flat field must be exact.** Every stage has DC gain 1, so a constant image must survive the whole
  pipeline untouched. Catches normalisation errors and wrong kernel sums.
- **Identity configurations must be identity.** `camToXYZ` = sRGB→XYZ makes the colour matrix a
  round-trip; `whitePoint` = 1 makes the tone curve algebraically the identity. Both let you verify a
  stage does real per-pixel work *and* returns exactly what it was given.
- **Cross-validate against an independent implementation.** The numpy unsharp reference agreed to
  0.16 LSB mean, which is stronger evidence than any self-consistency check.

## 5. Next — M1, CUDA

1. Port the pointwise stages to kernels (white balance, colour matrix, tone map, encode) and verify
   against the host results above.
2. **The fusion experiment.** Run those pointwise kernels separately — N round-trips to DRAM — then
   again as a single fused kernel, and measure both. This needs no graph machinery and it tests the
   project's central claim directly.
3. Neighbourhood kernels (demosaic, separable blur) with shared-memory tiling, motivated by the
   cache-locality measurement in §2.

## 6. Known gaps

- **Highlight recovery.** When one channel clips at the sensor and the others do not, hue shifts —
  blown skies drift cyan. Doing it properly requires tracking per-channel clip levels through the
  white-balance gains, which `RawParams` does not carry.
- **Gamut handling is a clip.** `cameraToSRGB` clamps negative channels to zero. Real gamut mapping
  compresses toward the gamut boundary, preserving hue and relative saturation instead of
  desaturating whatever falls outside.
- **Tone mapping can leave channels above 1.** The hue-preserving form scales RGB by one luminance
  ratio, but a saturated primary carries little luminance, so a channel can exceed 1.0 even when the
  tone-mapped luminance does not. `encodeSRGB8` clips it; the principled fix is a desaturation pass.
- **Demosaic border.** `cfaColorAt` and `atClamped` disagree about colour outside the image, so the
  outermost row and column are contaminated. Mirroring by 2 would preserve CFA parity. Measurable
  consequence: a downstream blur spreads this error inward by exactly its radius and amplifies it by
  `amount` — which is how the flat-field test lands at 61.8 dB rather than exact, with the differing
  pixels forming precisely a 6-pixel band at σ=2.
- **No local tone mapping**, and no noise reduction. NR belongs on the mosaic before demosaic, and
  must precede sharpening, which amplifies noise.
- **Single-threaded host code.** Intentional — M0 is the correctness oracle, not a performance target.
- `inotify` is unreliable on DrvFs, so CMake Tools and clangd can serve stale results. If a build
  result looks impossible, `rm -rf build && ./build.sh` before debugging the code.
