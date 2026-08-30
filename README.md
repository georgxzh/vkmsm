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
- [ ] **Stage 5** — port point arithmetic to GLSL compute shaders.
- [ ] **Stage 6** — GPU Pippenger implementation.
- [ ] **Stage 7** — benchmarking harness (GPU vs. CPU throughput).
- [ ] **Stage 8** (stretch) — window-size / bucket-count tuning.

## Building

Requires the Vulkan SDK, CMake, and MSVC (Visual Studio Build Tools) on
Windows.

```
cmake -S . -B build
cmake --build build --config Debug
```

blst is fetched and built automatically as part of the CMake configure/build
step.
