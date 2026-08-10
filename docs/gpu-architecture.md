# How a GPU works

Reference notes for this project. Every number is either measured on the development machine
(RTX 5060, sm_120, 30 SMs, 8 GB) or a documented architectural fact.

---

## 1. The hardware

A GPU is a grid of independent processors called **SMs** (Streaming Multiprocessors). Each SM has its
own register file, its own scratchpad memory, its own schedulers, and a set of arithmetic lanes.

```
RTX 5060:  30 SMs  ×  128 FP32 lanes  =  3840 lanes
           8 GB GDDR7,  measured 377.5 GB/s
           48 KB shared memory per block
```

A CPU has 8–16 cores. So a GPU has ~300× the lanes — but each lane is far weaker. A CPU core has
branch prediction, out-of-order execution, speculative execution, and deep private caches. A GPU lane
has none of those, and does not even fetch its own instructions independently.

The design difference in one line:

> **A CPU spends transistors avoiding latency. A GPU spends them hiding it.**

A CPU tries never to wait: predict the branch, prefetch the line, reorder around the stall. A GPU
accepts that it will wait constantly, and keeps so many threads in flight that there is always
something else to run.

This is why images suit GPUs. A 24 MP frame is 24 million pixels and, for most operations, each pixel
is independent of the others. Enormous, uniform, independent work is exactly the shape a GPU wants.

---

## 2. Execution model

You write code for one thread. The hardware runs millions.

```
thread   one instance of your kernel. Usually one pixel.
warp     32 threads. The real unit of execution. Always 32 on NVIDIA.
block    a group of threads (typically 256) sharing memory and able to synchronise.
grid     all blocks in one launch.
```

You do not schedule warps. You launch a grid; the hardware partitions it into warps of 32 by
consecutive linear thread index within each block.

### SIMT

The 32 threads of a warp **share one instruction pointer**. The hardware fetches one instruction and
executes it across all 32 lanes, on different data. NVIDIA calls this SIMT — Single Instruction,
Multiple Thread.

Two consequences follow, and almost all GPU performance advice reduces to one of them.

### Consequence 1: divergence

If threads in one warp disagree about a branch, the hardware runs **both sides serially**, each with
the non-participating lanes disabled.

```cuda
if (threadIdx.x & 1) { A(); }   // 16 lanes
else                 { B(); }   // 16 lanes
```

That warp executes `A()` at half occupancy, then `B()` at half occupancy. You pay for both.

**A branch is free when the whole warp agrees.** `if (x < width)` costs nothing in the image interior
and only bites in the edge blocks. A branch keyed on pixel parity splits every warp.

Measured in `bench/experiments.cu`: **1.90×** cost for the same two code paths, changing only whether
the branch is keyed on `blockIdx` (whole warp agrees) or `threadIdx` (lanes alternate).

This is why a fixed-kernel demosaic like Malvar-He-Cutler maps well to a GPU while an adaptive one
that chooses a direction per pixel does not, even though the adaptive method scores higher on PSNR.
Choosing the slightly worse algorithm because it maps to the hardware is a routine decision in GPU
imaging work.

### Consequence 2: memory access is judged per warp

See §4.

### Why blocks exist

Threads in a block can do two things threads in different blocks cannot:

1. Share **shared memory** — a fast, software-managed scratchpad.
2. Synchronise with `__syncthreads()`.

A block runs entirely on one SM and never migrates. Blocks are otherwise independent by design, which
is how the same binary scales from 30 SMs to 300 without changes.

Typical block size for image work is 256 threads, shaped `32×8` or `16×16`. The 32 in the x dimension
is not arbitrary — it makes a warp span 32 consecutive x values, so a warp reads a contiguous run of
one image row.

---

## 3. Memory hierarchy

Approximate latencies, which govern everything else:

| Level | Scope | Latency | Size |
|---|---|---|---|
| Registers | one thread | ~1 cycle | thousands per SM |
| Shared memory | one block | ~20–30 cycles | 48 KB/block here |
| L2 cache | whole GPU | ~200 cycles | tens of MB on modern parts |
| Global memory (DRAM) | whole GPU | **~400–800 cycles** | 8 GB |

DRAM is several hundred times slower than a register. Every optimisation technique below is a way of
touching DRAM less.

### Shared memory

A manually managed cache. Worth using when threads in a block read **the same data more than once**.

A 5×5 convolution reads each input pixel for 25 different output pixels. Straight from global memory
that is 25 loads. Stage a tile into shared memory once and it becomes one global load plus 25 shared
loads — roughly a 20× reduction in DRAM traffic for that stage.

The staged tile must include a **halo**: producing a 16×16 output tile with a radius-2 filter requires
loading 20×20 input. That halo arithmetic is identical to region-of-interest propagation in an imaging
framework. The GPU term and the framework term describe the same constraint.

**Do not use shared memory when there is no reuse.** A pointwise stage reads each pixel exactly once;
staging it through shared memory adds a copy and costs occupancy for zero benefit.

---

## 4. Coalescing

When a warp issues a load, the hardware examines all 32 addresses together. If they fall within a few
32-byte sectors, it fetches those sectors in **one transaction**. If they are scattered, it issues up
to 32 transactions and you get a fraction of peak bandwidth.

```
thread 0  -> bytes   0..15   ┐
thread 1  -> bytes  16..31   ├─ 512 contiguous bytes -> a few sectors -> COALESCED
thread 31 -> bytes 496..511  ┘

thread 0  -> byte      0     ┐
thread 1  -> byte   4096     ├─ 32 unrelated sectors -> 32 transactions
thread 31 -> byte 126976     ┘
```

Practical rules for images:

- **Consecutive threads must handle consecutive pixels.** Compute
  `x = blockIdx.x * blockDim.x + threadIdx.x` and index `[y * width + x]`.
- **Never index `[x * height + y]`.** That makes each lane stride an entire column.
- **RGBA beats RGB.** `float3` is 12 bytes, so thread N begins at byte 12N — misaligned, straddling
  sector boundaries. `float4` is 16 bytes, has a single hardware load instruction, and is naturally
  aligned. You waste 25% of memory on an unused channel and go faster, because what matters on a
  bandwidth-bound kernel is transaction efficiency, not bytes requested.

### What the measurement actually showed

From `bench/experiments.cu`, all three moving identical bytes:

| Pattern | ms | GB/s | vs contiguous |
|---|---|---|---|
| contiguous `[y*W + x]` | 1.561 | 343.9 | 1.00× |
| transposed `[x*H + y]` | 1.774 | 302.7 | **1.14×** |
| permuted (no reuse) | 21.126 | 25.4 | **13.5×** |

The transpose is genuinely scattered within every warp — 32 lanes, 32 sectors — yet costs only 14%.
The sectors it touches are reused almost immediately by neighbouring warps, and L2 is large enough to
hold them. The access is scattered in the small and dense in the aggregate.

The permutation destroys reuse as well as locality, and there the penalty is real: each 32-byte sector
fetched delivers one useful 16-byte pixel.

**The textbook claim that a transpose costs 10× dates from an era of much smaller caches.** On current
hardware, losing *reuse* hurts far more than losing contiguity. Both rules still matter; their relative
weight has shifted.

---

## 5. Occupancy and latency hiding

When a warp issues a global load it stalls for hundreds of cycles. The SM does not idle — it switches
to another resident warp at zero cost, because every resident warp's registers stay live in the
register file for its entire lifetime. There is no register save/restore, which is why GPU context
switching is free and CPU context switching is not.

**Occupancy** = resident warps per SM ÷ architectural maximum. It is the latency-hiding budget.

Occupancy is capped by whichever resource runs out first:

- **Registers per thread.** The register file is fixed per SM; more registers per thread means fewer
  resident warps.
- **Shared memory per block.** Same tradeoff.
- **Block size**, and the hardware limit on blocks per SM.

**Higher occupancy is not automatically better.** Once there are enough warps to cover memory latency,
more buys nothing — and forcing the register count down to get there can spill to local memory and
make things worse. A kernel at 50% occupancy hitting 90% of peak bandwidth is finished; raising its
occupancy would be optimising a metric rather than the program.

---

## 6. Arithmetic intensity and the roofline

**Arithmetic intensity** = FLOPs performed ÷ bytes moved. It determines whether a kernel is limited by
compute or by memory.

Worked example, a 3×3 colour matrix on RGBA fp32:

```
read  16 bytes  +  write 16 bytes   =  32 bytes
9 multiply-adds                     ≈  18 FLOP
                                    →  0.56 FLOP/byte
```

The **ridge point** is where a machine stops being memory-bound and starts being compute-bound:
peak FLOP/s ÷ peak bytes/s. For this GPU that is on the order of tens of FLOP/byte.

At 0.56, the colour matrix is memory-bound by more than an order of magnitude. The ALUs sit idle
waiting on memory.

Four consequences, which between them are the design of any GPU imaging framework:

1. **Arithmetic is free.** Optimising the matrix multiply gains nothing. Adding *more* math to a
   bandwidth-bound kernel is frequently also free.
2. **N chained pointwise stages cost N round-trips to DRAM.** Fusing them into one kernel cuts traffic
   ~N× and runtime follows. Measured here: **3.03×** for three stages.
3. **Precision is a performance feature.** fp32 → fp16 halves bytes and nearly halves runtime.
   Measured: **1.96×**, with both versions reaching the same GB/s. This is why an imaging framework
   exposes pixel formats — for speed, not storage.
4. **Tiling converts DRAM traffic into cache traffic**, the only other available lever.

### The arithmetic that predicts everything

A 24 MP RGBA fp32 image is 384 MB. One pointwise stage reads and writes it — 768 MB — which at
377.5 GB/s is **~2.0 ms floor, regardless of the math**. Five unfused stages: ~10 ms. Fused: ~2 ms.

You can predict a fusion speedup from traffic alone before writing the code. Doing that, then
measuring, then explaining any gap, is what makes a performance claim credible.

---

## 7. Measurement

### Launches are asynchronous

```cuda
auto t0 = now();
kernel<<<g,b>>>(...);     // returns IMMEDIATELY, before the GPU starts
auto t1 = now();          // measures the launch call. Reports microseconds. Wrong.
```

Use CUDA events, which the GPU timestamps in its own stream order:

```cuda
cudaEvent_t a, b;
cudaEventCreate(&a); cudaEventCreate(&b);
cudaEventRecord(a);
kernel<<<g,b>>>(...);
cudaEventRecord(b);
cudaEventSynchronize(b);
float ms; cudaEventElapsedTime(&ms, a, b);
```

### Always warm up

The first launch pays JIT compilation, context creation, and page faults, and can be 10–100× slower
than steady state. Run once untimed, then time a loop and divide.

### Errors are asynchronous too

An error raised inside a kernel surfaces at a later API call, so the line the runtime blames is usually
innocent. `cudaGetLastError()` immediately after a launch catches bad launch configurations;
`cudaDeviceSynchronize()` then surfaces execution errors. **`compute-sanitizer`** catches out-of-bounds
device writes and races — the CUDA equivalent of Valgrind, and the fastest way to find a tiled kernel
reading past its halo.

### Quote percentages of measured peak

Not milliseconds, and never against the spec sheet. The copy-kernel ceiling here is 377.5 GB/s, about
84% of the ~448 GB/s theoretical figure; the missing 16% is DRAM refresh, read/write bus turnaround,
and imperfect access scheduling — real and unreachable.

A kernel at 350 GB/s is at 93% of achievable and is **done**.

`ncu -c 1 --section SpeedOfLight ./prog` reports this directly. `-c 1` profiles only the first launch;
profiling a whole benchmark loop with `--set full` replays every launch ~20× and saves/restores all
touched memory between passes, which on a 1 GB working set never finishes.

### Microbenchmarks often measure the compiler

The first version of the divergence experiment here measured nothing. Both branches were `fmaf()` with
different constants, so the compiler hoisted the constants into a select and ran a single loop —
removing the divergence entirely. The paths had to be made structurally different (`__sinf` vs
`__cosf`) before the branch survived optimisation.

Any microbenchmark result that looks too small deserves a look at the generated code
(`nvcc -ptx` or `cuobjdump -sass`) before it is believed.

---

## 8. CUDA to Metal

Apple platforms use Metal. The models map almost one to one.

| CUDA | Metal |
|---|---|
| thread | thread |
| **warp** (32) | **SIMD-group** (32 on Apple GPUs) |
| block | **threadgroup** |
| grid | grid |
| shared memory | **threadgroup memory** |
| `__syncthreads()` | `threadgroup_barrier()` |
| global memory | device memory |
| `__global__` | `kernel` |
| `__device__` | (ordinary function) |
| stream | command queue / command buffer |
| `cudaMalloc` | `MTLBuffer`, `MTLHeap` |
| texture object | `MTLTexture` |

### The one real architectural difference

**Apple silicon has unified memory.** CPU and GPU share one physical pool. There is no PCIe bus and no
host-to-device copy.

What follows from that:

- The classic "overlap the H2D copy with compute using streams" optimisation is **irrelevant** there.
  There is nothing to overlap.
- But CPU and GPU **contend for the same bandwidth**, so a bandwidth-hungry kernel steals from the CPU.
  Being frugal with bandwidth matters *more*, not less.
- Zero-copy handoff between CPU and GPU stages is cheap, which changes where a stage should live.

Apple GPUs are also tile-based deferred renderers with on-chip tile memory. For compute the model is
close to CUDA's, but that tile-memory heritage is one reason tiled evaluation runs so deep in Core
Image's design.

---

## 9. Concept self-check

Answer these out loud. Answers that feel obvious silently tend to fall apart when spoken.

**What is a warp and why does it matter?**
32 threads sharing one instruction pointer — the real unit of execution. It matters twice: divergent
branches within a warp serialise, so both paths cost; and memory coalescing is judged per warp, so 32
lanes on consecutive addresses is one transaction while 32 scattered addresses is up to 32.

**How do you speed up a memory-bound kernel?**
Move fewer bytes. In order: fuse adjacent passes so intermediates never reach DRAM; reduce precision;
ensure accesses are coalesced and aligned; use shared memory where there is reuse. Occupancy and
instruction selection come last.

**How do you know it is memory-bound?**
Compute arithmetic intensity and compare against the ridge point — a colour matrix at ~0.5 FLOP/byte
against a ridge in the tens is memory-bound by an order of magnitude. Confirm with `ncu`'s
SpeedOfLight section.

**Your kernel reaches 60% of peak bandwidth. What do you check?**
Uncoalesced or misaligned access first (`float3` instead of `float4` is the classic); then whether
there are enough resident warps to hide latency; then whether there are enough blocks to fill all SMs;
then tail effects where the final wave runs with the machine mostly idle.

**When would you not use shared memory?**
When there is no reuse. A pointwise stage reads each pixel once, so staging through shared memory adds
a copy and costs occupancy for nothing.

**Why is `float4` faster than `float3`?**
Alignment. 12 bytes puts thread N at byte 12N, straddling sector boundaries; 16 bytes is naturally
aligned with a single hardware load. Wasting 25% of memory increases throughput.

**Why does the fused kernel run 3× faster if both versions hit the same GB/s?**
Because it is not more efficient per byte — it moves 3× fewer bytes. The intermediates stay in
registers instead of round-tripping to DRAM.

**You used CUDA; we use Metal.**
The execution and memory models map nearly one to one — threadgroup for block, SIMD-group for warp,
threadgroup memory for shared. The real difference is unified memory: no host-device copies, so
copy/compute overlap becomes moot, while bandwidth frugality matters more because the GPU shares
bandwidth with the CPU.
