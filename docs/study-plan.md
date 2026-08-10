# Study plan

Ordered by what unblocks the next thing. Each item has a concrete completion test — something that
either passes or does not, rather than a feeling of having read enough.

---

## Stage 1 — Understand the machine

**Read:** [gpu-architecture.md](gpu-architecture.md) §1–5.

Do not memorise the API. Build the causal chain, because everything else is a corollary:

```
a GPU hides latency instead of avoiding it
  -> it needs thousands of threads in flight
    -> 32 of them share one instruction pointer (a warp)
      -> divergent branches serialise
      -> memory coalescing is judged per warp
  -> DRAM is ~500 cycles, a register ~1
    -> every optimisation is about touching DRAM less
```

**Completion test.** Explain, without notes:
- Why a GPU needs thousands of threads when a CPU does not
- What happens when 16 threads of a warp take an `if` and 16 take the `else`
- Why `[y*width + x]` is right and `[x*height + y]` is wrong
- Why `float4` beats `float3` even though it wastes 25% of memory

---

## Stage 2 — Read the code

**Read:** [cuda-walkthrough.md](cuda-walkthrough.md) alongside the actual sources, in this order:

1. `bench/bwtest.cu` — 50 lines, one kernel
2. `include/tinyci/cuda_util.h` — the infrastructure
3. `bench/fusion.cu` — the central measurement

**Completion test.** Point at any line in `fusion.cu` and say what it does and why it is there.
Specifically:
- Why the bounds guard is mandatory rather than defensive
- Why there is a warm-up launch before timing
- Why `CudaTimer` uses CUDA events instead of `std::chrono`
- Why both paths call the same `__device__` functions
- Why both paths reach the same GB/s while one is 3× faster

---

## Stage 3 — Run the experiments

```bash
./build.sh
./build/bwtest
./build/fusion
./build/experiments
```

**Read:** [measurements.md](measurements.md), matching each number to the section of
`gpu-architecture.md` that explains it.

Pay attention to the three results that contradict the textbook:

- **The transpose costs 14%, not 10×.** L2 rescues a pattern that is scattered per warp but dense in
  aggregate. What actually costs 13.5× is losing *reuse*.
- **The first divergence benchmark measured nothing.** The compiler hoisted two `fmaf()` constants into
  a select and ran one loop. Microbenchmarks routinely measure the compiler.
- **MHC loses 5.3 dB on anti-correlated channels.** A better algorithm is only better where its
  assumption holds.

**Completion test.** For each of the six measurements, state the mechanism in one sentence — not the
number, the reason.

---

## Stage 4 — Write a kernel

Open `bench/demosaic_gpu.cu` and fill in the two TODOs.

```bash
./build.sh && ./build/demosaic_gpu
```

**Target: `max |diff| : 0`** — bit-identical to the host implementation. Same algorithm and arithmetic
on a different machine should agree exactly.

This is the step that converts "I ran a GPU benchmark" into "I ported an image processing algorithm to
the GPU and validated it against a reference." Nothing else on this list matters as much.

Two failure modes the harness diagnoses for you:
- **All pixels zero** — the kernel is not running, or a TODO is unfilled.
- **Only green wrong** — the `horiz` case was tested before the both-axes case. Green occupies a
  quincunx, so at a non-green site it lies on both axes at once.

**Completion test.** Bit-identical output, plus a real speedup number, plus an explanation of why the
output writes are not perfectly coalesced (3 interleaved floats = 12-byte stride).

---

## Stage 5 — Shared memory and tiling

The next real technique, and where the GPU and framework vocabularies converge.

A neighbourhood operator reads each input pixel many times — a 5×5 filter reads each one for 25
different outputs. Staging a tile into shared memory turns 25 DRAM loads into 1 DRAM load plus 25
shared loads.

The tile needs a **halo**: a 16×16 output tile with a radius-2 filter requires loading 20×20 input.

```
1. declare  __shared__ float tile[TILE_H + 2*R][TILE_W + 2*R];
2. cooperatively load the tile plus halo (more elements than threads -- loop)
3. __syncthreads();          // every thread must reach this
4. compute from shared memory
```

Target: the separable blur. The host measurement showed it scaling as N^1.4 rather than N because the
vertical pass strides a cache line per tap — exactly what tiling fixes.

**Completion test.** Correct output including at tile seams (`compute-sanitizer` clean), and a
measured improvement over the naive version at σ=4 or 8 where the effect is largest.

The halo arithmetic here *is* ROI propagation: to produce a valid output region you must supply input
inset by the filter radius. Stack two neighbourhood stages and the required margin grows by the sum of
their radii.

---

## Stage 6 — The graph

The architectural work this project was designed around, and the part that distinguishes an imaging
*framework* from a collection of fast kernels.

```
Image     a recipe, not a bitmap. Holds a node pointer, extent, colour space, format.
          Copying is free. Evaluation is lazy.
Node      N input Images + a Kernel + parameters.
Kernel    Pointwise      -> output(x,y) depends only on input(x,y)   -> FUSABLE
          Neighbourhood  -> depends on a window                      -> FUSION BARRIER
Region    DOD (forward):  what area does this node produce?
          ROI (backward): to produce rect R, what input rect do I need?
Context   The evaluator. Walks the graph, fuses runs of pointwise nodes, tiles under a
          memory budget, schedules.
```

Why lazy evaluation is not a style choice: **you cannot fuse what you have already executed.** The
graph must be a description until the moment it is rendered, or the optimisation opportunity is gone.

Why ROI propagates backward while DOD propagates forward: you start from the output region you want and
ask each node what input it needs, which flows from consumer to producer. The domain of definition
flows the other way, from producer to consumer.

**Completion test.** Render a tiled image where the tile seams are invisible, under a hard memory
budget, with a measured fusion speedup on the real pipeline rather than a synthetic benchmark.

---

## Stage 7 — Beyond

In rough order of value:

- **Streams and copy/compute overlap.** Double-buffer tiles so a transfer overlaps the previous tile's
  compute. Note this optimisation *disappears* on Apple silicon — unified memory means there is no copy
  to overlap.
- **Caching intermediates across parameter changes.** Dragging a slider re-renders continuously, and
  everything upstream of the changed node is reusable. This is where a graph pays for itself in a real
  application.
- **A learned stage as a graph node.** The interesting problems are not the convolutions: layout
  conversion (interleaved RGBA versus planar NCHW), precision negotiation at the boundary, and avoiding
  round-trips when the ML runtime and the imaging graph do not share an allocator. A conv stack has an
  ROI exactly like a blur does, so a learned stage should obey the same contract as every other node
  rather than being bolted on the end.
- **Metal port.** The concepts transfer almost entirely; see
  [gpu-architecture.md](gpu-architecture.md) §8.

---

## Reference material

- **CUDA C++ Programming Guide** — the authoritative source. Chapters on the execution model and memory
  hierarchy are worth reading properly; the rest is reference.
- **CUDA C++ Best Practices Guide** — coalescing, occupancy, and measurement methodology.
- **Nsight Compute** — `ncu -c 1 --section SpeedOfLight ./prog`. `-c 1` profiles only the first launch;
  `--set full` across a benchmark loop replays every launch ~20× and never finishes.
- **compute-sanitizer** — out-of-bounds device writes and races. The fastest way to find a tiled kernel
  reading past its halo.
- **Malvar, He, Cutler (2004)**, *High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned
  Color Images*, ICASSP — the source of the 5×5 kernels in `src/pipeline.cpp`.
- **Williams, Waterman, Patterson (2009)**, *Roofline: An Insightful Visual Performance Model* — the
  arithmetic-intensity framework underlying every performance argument here.

---

## What this project does not have

Being able to name the gaps is worth as much as the code:

- No graph, no lazy evaluation, no ROI propagation — designed but not built.
- No tiling and no memory budget. The GPU work runs whole images.
- No shared-memory kernels. Every GPU kernel here reads straight from global memory.
- No streams, no copy/compute overlap.
- No highlight recovery; clipped channels shift hue.
- Gamut handling is a clip, not a mapping.
- No noise reduction, which belongs on the mosaic before demosaic and must precede sharpening.
- Demosaic borders are contaminated by a clamp/parity mismatch; mirroring by 2 would fix it.
