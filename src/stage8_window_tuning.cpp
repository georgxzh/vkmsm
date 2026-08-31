// Stage 8 (stretch): window-size / bucket-count tuning.
//
// window_bits controls a real tradeoff on both sides:
//   - Fewer windows to process (num_windows = ceil(256/window_bits)) as
//     window_bits grows, but each window's bucket array grows as
//     2^window_bits - more memory, more per-window reduction cost.
//   - On the GPU specifically (see gpu_pippenger.cpp), this tradeoff is
//     sharper than usual: the bucket-fill shader runs exactly
//     2^window_bits threads per dispatch (one per bucket, our Stage 6
//     conflict-free design). Stage 7 found the GPU badly underutilized
//     at window_bits=8 (only 256 threads active) - so *increasing*
//     window_bits here isn't just "fewer passes," it's also more GPU
//     parallelism, right up until bucket/reduction overhead dominates.
//     This is exactly the tradeoff this stage exists to find empirically.
//
// n is held fixed so the sweep isolates window_bits as the only variable.
// Correctness (point_equal against msm_pippenger) is re-checked at every
// window_bits value before its timing is trusted.

#include <blst.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "fp.hpp"
#include "gpu_pippenger.hpp"
#include "msm.hpp"
#include "point.hpp"

using namespace vkmsm;

namespace {

const PointJacobian& generator() {
    static const PointJacobian g = [] {
        const blst_p1* bg = blst_p1_generator();
        std::array<uint32_t, kLimbs> x, y, z;
        blst_uint32_from_fp(x.data(), &bg->x);
        blst_uint32_from_fp(y.data(), &bg->y);
        blst_uint32_from_fp(z.data(), &bg->z);
        PointJacobian p;
        p.x = fp_from_plain_limbs(x);
        p.y = fp_from_plain_limbs(y);
        p.z = fp_from_plain_limbs(z);
        return p;
    }();
    return g;
}

Scalar random_scalar(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    Scalar s;
    for (auto& l : s) l = dist(rng);
    return s;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    g_checks++;
    if (!condition) {
        g_failures++;
        std::printf("  FAIL: %s\n", what);
    }
}

}  // namespace

int main() {
  try {
    const int n = 1 << 13;  // 8192, fixed across the sweep
    std::printf("Stage 8: window_bits tuning (n = %d)\n\n", n);

    std::mt19937 rng(0x8168A6E);
    std::vector<Scalar> scalars(n);
    for (auto& s : scalars) s = random_scalar(rng);

    std::vector<PointJacobian> points(n);
    points[0] = generator();
    for (int i = 1; i < n; i++) points[i] = point_add(points[i - 1], generator());

    struct Row {
        int window_bits;
        int num_buckets;
        double cpu_ms, gpu_ms;
        bool correct;
    };
    std::vector<Row> rows;

    for (int wb : {2, 4, 6, 8, 10, 12}) {
        std::printf("window_bits=%-3d (%5d buckets) ... ", wb, 1 << wb);
        std::fflush(stdout);

        double t0 = now_ms();
        PointJacobian cpu_result = msm_pippenger(points, scalars, wb);
        double t1 = now_ms();

        GpuPippengerContext* ctx = create_gpu_pippenger_context(wb);
        PointJacobian gpu_result = gpu_pippenger(*ctx, points, scalars);
        destroy_gpu_pippenger_context(ctx);
        double t2 = now_ms();

        bool correct = point_equal(cpu_result, gpu_result);
        check(correct, "GPU Pippenger matches CPU Pippenger at this window_bits");

        rows.push_back({wb, 1 << wb, t1 - t0, t2 - t1, correct});
        std::printf("CPU %.1f ms, GPU %.1f ms, %s\n", t1 - t0, t2 - t1, correct ? "correct" : "MISMATCH");
    }

    std::printf("\n=====================================\n");
    std::printf("%-14s %-12s %-14s %-14s %s\n", "window_bits", "buckets", "CPU (ms)", "GPU (ms)", "correct");
    for (const auto& r : rows) {
        std::printf("%-14d %-12d %-14.1f %-14.1f %s\n", r.window_bits, r.num_buckets, r.cpu_ms,
                     r.gpu_ms, r.correct ? "yes" : "NO");
    }

    std::ofstream csv("results/stage8_window_tuning.csv");
    csv << "window_bits,num_buckets,n,cpu_ms,gpu_ms\n";
    for (const auto& r : rows) {
        csv << r.window_bits << "," << r.num_buckets << "," << n << "," << r.cpu_ms << "," << r.gpu_ms
            << "\n";
    }
    csv.close();
    std::printf("\nResults written to results/stage8_window_tuning.csv\n");

    std::printf("\n=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
