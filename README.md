# vkMSM

A GPU-accelerated multi-scalar multiplication (MSM) engine for elliptic curve
cryptography, implemented in native Vulkan compute shaders. Goal: implement
Pippenger's bucket method MSM over BLS12-381, validated against a CPU
reference at every stage, with a GPU vs. CPU throughput benchmark.

This is a personal/portfolio project prioritizing correctness over speed of
delivery — every stage is validated against a trusted CPU reference
([blst](https://github.com/supranational/blst)) before moving on to the next.

## Tech stack

- C++17, CMake
- Vulkan SDK (raw Vulkan compute, no wrapper library)
- GLSL compute shaders, compiled to SPIR-V via `glslc`
- Curve: BLS12-381 (base field `F_p`, ~381-bit prime)
- CPU-side oracle: [blst](https://github.com/supranational/blst) (pinned via
  CMake `FetchContent`, tag `v0.3.17`), used purely to generate known-answer
  test vectors

## Design decisions

- **Field element representation:** 12 x 32-bit limbs. GPUs generally lack
  fast native 64-bit integer arithmetic, so the CPU reference implementation
  uses the same 32-bit-limb layout it will need on GPU, rather than a more
  CPU-idiomatic 6 x 64-bit layout that would need redesigning later.
- **Field arithmetic:** Montgomery representation with CIOS
  (Coarsely Integrated Operand Scanning) multiplication — the standard
  approach, and the one that ports cleanly to shader code.
- **Pippenger window size:** 8 bits by default (256 buckets/window), exposed
  as a parameter rather than hard-coded — `window_bits` is revisited for
  real performance tuning in Stage 8.

## Build order

Each stage validates the one before it; none are skipped.

- [x] **Stage 0** — environment sanity check: a trivial Vulkan compute
  vector-add kernel, proving the instance/device/queue/shader/descriptor/
  command-buffer/sync pipeline works.
- [x] **Stage 1** — CPU finite field arithmetic (`F_p` for BLS12-381),
  validated against blst (65,000+ checks, 0 failures).
- [x] **Stage 2** — CPU elliptic curve point arithmetic (G1, Jacobian
  coordinates), validated against blst (10,500+ checks, 0 failures).
- [x] **Stage 3** — CPU MSM baselines: naive double-and-add (validated
  against blst), then Pippenger's bucket method (validated against the
  naive baseline across sizes 0-1024 and window sizes 1-16 bits).
- [x] **Stage 4** — port field arithmetic to GLSL compute shaders, validated
  against the CPU reference (3,072 checks across add/sub/mul, 0 failures).
- [x] **Stage 5** — port point arithmetic to GLSL compute shaders, validated
  against the CPU reference (1,024 checks across double/add, 0 failures).
- [x] **Stage 6** — GPU Pippenger implementation, validated against the
  Stage 3 CPU Pippenger baseline (sizes 0-1024, 0 failures). Uses a
  conflict-free bucket-assignment scheme (one GPU thread per bucket,
  scanning all points) rather than atomics, since there's no atomic
  elliptic-curve-point-addition primitive to atomically accumulate with.
- [x] **Stage 7** — benchmarking harness (GPU vs. CPU throughput). See
  [Results](#results) below.
- [x] **Stage 8** (stretch) — window-size / bucket-count tuning. See
  [Results](#results) below.

## Results

GPU vs CPU Pippenger MSM throughput, BLS12-381 G1, `window_bits=8` on both
sides, Release build. Full report with a chart: `results/stage7_benchmark.html`
(open in a browser); raw numbers: `results/stage7_benchmark.csv`.

| n | CPU time | CPU pts/s | GPU time | GPU pts/s | GPU/CPU |
|---|---|---|---|---|---|
| 1,024  | 138.9 ms | 7,375 | 556.2 ms  | 1,841 | 0.25x |
| 4,096  | 473.2 ms | 8,657 | 1.89 s    | 2,167 | 0.25x |
| 16,384 | 1.81 s   | 9,030 | 7.23 s    | 2,265 | 0.25x |
| 65,536 | 7.16 s   | 9,158 | 62.5 s    | 1,048 | 0.11x |

GPU tested: AMD Radeon(TM) Graphics (integrated). Every result above was
re-verified against the CPU reference before being trusted.

**The GPU implementation is currently slower than CPU, and the gap widens
with `n`.** This is a real, structural limitation of the Stage 6 design, not
a fluke: the bucket-fill shader runs exactly `2^window_bits = 256` threads
per dispatch (one thread per *bucket*, chosen to avoid races without atomics
— there's no atomic elliptic-curve-point-addition operation to begin with),
and each thread scans all `n` points sequentially. A GPU that can run far
more than 256 threads concurrently sits mostly idle. At `n = 2^18` a single
dispatch exceeded Windows' ~2s TDR (Timeout Detection and Recovery) limit
and the driver killed it (`VK_ERROR_DEVICE_LOST`), which is why the
benchmark stops at `2^16` instead of reaching the original `2^20` target.
Fixing this needs a bucket-assignment scheme with far more active threads
(e.g. one thread per point instead of per bucket) — real algorithm work,
left for Stage 8.

Public benchmark numbers exist for highly-tuned CUDA MSM implementations
(cuZK, Icicle) at comparable sizes, achieving very high throughput on
discrete NVIDIA GPUs. Those aren't included here: this project runs on an
integrated AMD GPU with an untuned, correctness-first implementation, and a
cross-hardware, cross-maturity comparison like that wouldn't be
apples-to-apples.

### Stage 8 — window_bits tuning

Held `n = 8,192` fixed and swept `window_bits`, since the GPU's active
thread count per dispatch is exactly `2^window_bits` (Stage 6's
one-thread-per-bucket design) — this parameter isn't just a memory/pass-count
tradeoff on the GPU side, it's a direct parallelism knob.

| window_bits | buckets | CPU time | GPU time | GPU/CPU |
|---|---|---|---|---|
| 2  | 4     | 2.70 s | 58.2 s | 0.05x |
| 4  | 16    | 1.68 s | 28.3 s | 0.06x |
| 6  | 64    | 1.19 s | 19.0 s | 0.06x |
| 8  | 256   | 0.92 s | 3.81 s | 0.24x |
| 10 | 1,024 | 0.81 s | 1.55 s | 0.53x |
| 12 | 4,096 | 0.91 s | 2.37 s | 0.39x |

GPU time drops **~37x** from `window_bits=2` to `window_bits=10` as more
active threads directly fix the underutilization Stage 7 found, then rises
again at `window_bits=12` as larger per-window bucket bookkeeping starts to
dominate. `window_bits=10` is the sweet spot for this `n`, closing GPU to
within ~2x of CPU instead of ~20x behind — but it still doesn't win
outright at any window size tried. Doing that needs the higher-parallelism
bucket-assignment redesign noted above (real algorithm work), not just
retuning this knob.

## Building

Requires the Vulkan SDK, CMake, and MSVC (Visual Studio Build Tools) on
Windows.

```
cmake -S . -B build
cmake --build build --config Debug
```

blst is fetched and built automatically as part of the CMake configure/build
step.
