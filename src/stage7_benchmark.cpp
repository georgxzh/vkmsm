// Stage 7: benchmarking harness. Measures wall-clock MSM time for both
// the CPU Pippenger baseline (Stage 3) and the GPU Pippenger
// implementation (Stage 6) across a range of input sizes, and reports
// throughput.
//
// Size range note: the original build order calls for n up to 2^20, but
// the current GPU Pippenger design (Stage 6: one thread per BUCKET, not
// per point - chosen for correctness/simplicity, see gpu_pippenger.cpp)
// only keeps 256 threads active per dispatch. At n=2^18 and above, each
// thread's O(n) sequential scan takes long enough that Windows' TDR
// (Timeout Detection and Recovery) kills the GPU command as a suspected
// driver hang (VK_ERROR_DEVICE_LOST), before it can finish. Benchmarking
// here is capped at n=2^16, where it comfortably completes; reaching
// 2^18/2^20 needs a bucket-assignment scheme with far more active
// threads (e.g. one thread per point, with a different conflict-
// avoidance strategy) - real algorithm work, deferred to Stage 8.
//
// Correctness is re-verified (via point_equal) at every size before its
// timing is trusted - a benchmark number for a wrong result isn't a
// benchmark, it's a coincidence.
//
// Test points are generated as 1*G, 2*G, ..., n*G (via repeated addition
// of the generator, using the already-validated Stage 2 point_add) rather
// than independently random scalar multiples - this is much cheaper to
// generate (O(n) point additions instead of O(n * 256)) and is standard
// practice for MSM benchmarking: performance depends on n and the
// scalars' bit patterns, not on which specific curve points are used.
// Scalars themselves are still genuinely random. blst is used only to
// source the trusted G1 generator constant.

#include <blst.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
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

struct BenchResult {
    int n;
    double cpu_ms;
    double gpu_ms;
    bool correct;
};

}  // namespace

int main() {
  try {
    const int window_bits = 8;
    std::printf("Stage 7: GPU vs CPU Pippenger MSM benchmark\n\n");

    GpuPippengerContext* ctx = create_gpu_pippenger_context(window_bits);
    std::printf("GPU: %s\n", gpu_pippenger_device_name(*ctx));
    std::printf("CPU Pippenger window_bits=%d (Stage 3), GPU Pippenger window_bits=%d (Stage 6)\n\n",
                 window_bits, window_bits);

    std::mt19937 rng(0xBE4C4);
    std::vector<BenchResult> results;

    for (int n : {1 << 10, 1 << 12, 1 << 14, 1 << 16}) {
        std::printf("n = 2^%d = %d ... ", static_cast<int>(std::log2(static_cast<double>(n))), n);
        std::fflush(stdout);

        std::vector<Scalar> scalars(n);
        for (auto& s : scalars) s = random_scalar(rng);

        std::vector<PointJacobian> points(n);
        points[0] = generator();
        for (int i = 1; i < n; i++) points[i] = point_add(points[i - 1], generator());

        double t0 = now_ms();
        PointJacobian cpu_result = msm_pippenger(points, scalars, window_bits);
        double t1 = now_ms();
        PointJacobian gpu_result = gpu_pippenger(*ctx, points, scalars);
        double t2 = now_ms();

        bool correct = point_equal(cpu_result, gpu_result);
        results.push_back({n, t1 - t0, t2 - t1, correct});

        std::printf("CPU %.1f ms, GPU %.1f ms, %s\n", t1 - t0, t2 - t1, correct ? "correct" : "MISMATCH");
    }

    destroy_gpu_pippenger_context(ctx);

    std::printf("\n=====================================\n");
    std::printf("%-10s %-14s %-16s %-14s %-16s %-10s %s\n", "n", "CPU (ms)", "CPU (pts/s)",
                 "GPU (ms)", "GPU (pts/s)", "speedup", "correct");
    bool all_correct = true;
    for (const auto& r : results) {
        double cpu_throughput = r.n / (r.cpu_ms / 1000.0);
        double gpu_throughput = r.n / (r.gpu_ms / 1000.0);
        double speedup = r.cpu_ms / r.gpu_ms;
        std::printf("%-10d %-14.1f %-16.0f %-14.1f %-16.0f %-10.2f %s\n", r.n, r.cpu_ms,
                     cpu_throughput, r.gpu_ms, gpu_throughput, speedup, r.correct ? "yes" : "NO");
        all_correct = all_correct && r.correct;
    }

    // Also write a CSV for the benchmark report / chart.
    std::ofstream csv("results/stage7_benchmark.csv");
    csv << "n,cpu_ms,cpu_points_per_sec,gpu_ms,gpu_points_per_sec,speedup\n";
    for (const auto& r : results) {
        double cpu_throughput = r.n / (r.cpu_ms / 1000.0);
        double gpu_throughput = r.n / (r.gpu_ms / 1000.0);
        csv << r.n << "," << r.cpu_ms << "," << cpu_throughput << "," << r.gpu_ms << ","
            << gpu_throughput << "," << (r.cpu_ms / r.gpu_ms) << "\n";
    }
    csv.close();
    std::printf("\nResults written to results/stage7_benchmark.csv\n");

    return all_correct ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
