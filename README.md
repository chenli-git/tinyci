# TinyCI

A tile-based GPU image processing graph — a miniature [Core Image](https://developer.apple.com/documentation/coreimage) —
demonstrated on a RAW→display pipeline.

The goal is not "a fast image filter." It is the *architecture* that production imaging frameworks
actually have: a lazily-evaluated filter graph, backward region-of-interest propagation, tiled
evaluation under a fixed memory budget, and fusion of pointwise kernels into single GPU passes.

**Status: M0 complete.** The full host reference pipeline is implemented and validated — RAW mosaic in,
sRGB PNG out, scored against ground truth. Next milestone is the CUDA port. See [STATUS.md](STATUS.md)
for measurements and known gaps.

---

## Why a graph, not a filter

Chain seven image operations naively and you make seven round-trips to DRAM. The arithmetic is
trivial; the memory traffic is everything.

Concretely, on the development GPU (RTX 5060, measured 377.5 GB/s peak): a 24 MP RGBA fp32 image is
384 MB. One pointwise stage reads it and writes it — 768 MB of traffic, ≈ **2.0 ms**, no matter how
simple the math. A 3×3 color-matrix multiply is ~0.28 FLOP/byte, more than an order of magnitude below
the roofline ridge point. It is purely bandwidth-bound.

So five unfused pointwise stages cost ~10 ms of pure traffic; fused into one pass, ~2 ms. That
prediction — made from first principles before writing the code — is the thesis this project exists to
test. It is also why Core Image concatenates kernels rather than merely optimizing them.

## Pipeline

```
Bayer RAW (u16, CFA)
  → black-level subtract + white balance      [pointwise]
  → demosaic (bilinear → Malvar-He-Cutler)    [NEIGHBOURHOOD — fusion barrier]
  → camera RGB → XYZ → linear sRGB            [pointwise]
  → highlight recovery + tone curve           [pointwise]
  → unsharp mask (separable Gaussian)         [NEIGHBOURHOOD — fusion barrier]
  → sRGB OETF encode → u8                     [pointwise]
```

That shape is chosen deliberately: a run of pointwise stages, a neighbourhood barrier, another run of
pointwise stages. It is exactly the structure that makes fusion *measurable*.

## Documentation

| | |
|---|---|
| [docs/gpu-architecture.md](docs/gpu-architecture.md) | How a GPU works: execution model, memory hierarchy, coalescing, occupancy, roofline, CUDA→Metal |
| [docs/cuda-walkthrough.md](docs/cuda-walkthrough.md) | Every CUDA construct in this repo, explained where it appears |
| [docs/measurements.md](docs/measurements.md) | Every measured result with the reasoning behind it |
| [docs/study-plan.md](docs/study-plan.md) | Ordered learning path, each stage with a completion test |
| [STATUS.md](STATUS.md) | Current state, validation, known gaps |

## Build

Requires WSL2 Ubuntu 24.04 (or any Linux), CUDA ≥ 12.8, GCC ≤ 13, CMake ≥ 3.22, Ninja.
`CMAKE_CUDA_ARCHITECTURES` is set to `120` for Blackwell — change it for your GPU.

```bash
./build.sh          # cmake + ninja, in-tree to ./build
./smoke.sh          # generate test data → run pipeline → print PSNR
```

<details>
<summary>Building on a Windows drive under WSL</summary>

In-tree builds on `/mnt/...` require the DrvFs `metadata` automount option, or CMake's
`configure_file()` fails with *"Operation not permitted"* — DrvFs otherwise cannot store Linux
permission bits, so the `chmod` it performs silently fails.

`/etc/wsl.conf`:
```ini
[automount]
options = "metadata,uid=1000,gid=1000,umask=22,fmask=11"
```
then `wsl --shutdown`. Alternatively keep the build tree off the Windows drive:
`TINYCI_BUILD=~/tinyci-build ./build.sh`.
</details>

## Run

```bash
# synthesize Bayer input from any RGB image
python3 tools/bayer.py mosaic data/scene.png --cfa RGGB --black 512 --white 65535

# run the pipeline
./build/tinyci data/scene_bayer.pgm data/scene_out.png \
    --cfa RGGB --black 512 --white 65535 [--mhc] [--amount 0.5 --sigma 1.5]

# score against ground truth
python3 tools/bayer.py psnr data/scene_out.png data/scene_gt.png
```

Synthetic mosaics rather than real RAW files: the original image is kept as **exact ground truth**, so
demosaic quality is a number instead of an opinion. It also avoids a RAW parser, which would prove
nothing about GPU or framework design.

`bwtest` measures peak achievable memory bandwidth — the denominator for every performance claim here:

```bash
./build/bwtest
ncu -c 1 --section SpeedOfLight ./build/bwtest
```

Quote kernels as a percentage of *measured* peak, never of the spec sheet. On this hardware: 377.5 GB/s
measured against ~448 GB/s theoretical, so ~84%. The gap is DRAM refresh, read/write bus turnaround,
and imperfect access scheduling — real and unreachable.

## Layout

```
include/tinyci/image.h      Image<T>: owning interleaved 2D buffer, CFA helpers
include/tinyci/pipeline.h   Stage declarations + RawParams
include/tinyci/io.h         PNG (stb) and 16-bit PGM I/O
src/pipeline.cpp            The stages
src/main.cpp                CLI driver + per-stage timing
tools/bayer.py              Bayer synthesis and PSNR scoring
tools/make_test_scene.py    Synthetic test chart, hostile to bilinear demosaic
tools/make_eval_set.py      Evaluation set varying frequency and channel correlation
bench/bwtest.cu             Peak-bandwidth benchmark
```

## A result worth reading

Malvar-He-Cutler does **not** beat bilinear on every image, and the pattern is exact:

| Image | bilinear | MHC | Δ |
|---|---|---|---|
| luminance detail, near Nyquist | 31.93 | 40.30 | **+8.37** |
| constant channel ratios | 33.02 | 35.17 | **+2.15** |
| anti-correlated channels | 33.85 | 28.59 | −5.26 |

MHC corrects one channel using the Laplacian of the channel measured at that site — valid only when
luminance detail is shared across R, G and B. Where that assumption holds it wins by up to 8 dB; where
it fails the "correction" injects error, and assumption-free bilinear degrades more gracefully.
`tools/make_eval_set.py` varies spatial frequency and inter-channel correlation independently so the
boundary is measured rather than assumed.

## Roadmap

| | Milestone | Demonstrates |
|---|---|---|
| **M0** | Host C++ reference pipeline | Correctness oracle for everything after |
| M1 | Naive CUDA, one kernel per stage | The baseline to beat |
| M2 | Lazy graph: `Image`/`Node`/`Kernel`, DOD + ROI propagation | Framework design |
| M3 | Tiled evaluation under a memory budget | Memory-constrained execution |
| M4 | Pointwise kernel fusion | The ~5× predicted above |
| M5 | Shared-memory neighbourhood ops, fp16, vectorized access | Kernel optimization |
| M6 | Streams + double buffering | Copy/compute overlap |

## License

stb headers in `third_party/` are public domain (Sean Barrett). The rest is unlicensed personal work.
