// Stage 6 validation: GPU Pippenger MSM (src/gpu_pippenger.cpp), cross-
// checked against the already-validated CPU Pippenger implementation
// (src/msm.cpp, Stage 3).
//
// See src/gpu_pippenger.cpp for the conflict-free bucket-assignment
// design (one GPU thread per bucket, not per point) and why it was
// chosen over atomics.
//
// blst is used only to source the G1 generator (an already-trusted
// constant, per Stage 3.2's pattern) - random test points are derived
// via our own validated point_scalar_mul, not blst_p1_mult.

#include <blst.h>

#include <array>
#include <cstdio>
#include <random>
#include <vector>

#include "fp.hpp"
#include "gpu_pippenger.hpp"
#include "msm.hpp"
#include "point.hpp"

using namespace vkmsm;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    g_checks++;
    if (!condition) {
        g_failures++;
        std::printf("  FAIL: %s\n", what);
    }
}

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

PointJacobian random_point(std::mt19937& rng) { return point_scalar_mul(generator(), random_scalar(rng)); }

}  // namespace

int main() {
    const int window_bits = 8;
    std::printf("Stage 6: GPU Pippenger MSM vs CPU Pippenger (Stage 3 reference)\n\n");

    GpuPippengerContext* ctx = create_gpu_pippenger_context(window_bits);
    std::printf("Using GPU: %s\n", gpu_pippenger_device_name(*ctx));
    std::printf("window_bits=%d, num_buckets=%d\n\n", window_bits, 1 << window_bits);

    std::mt19937 rng(0xC0DE6);

    for (int n : {0, 1, 2, 3, 37, 256, 1024}) {
        std::vector<PointJacobian> points;
        std::vector<Scalar> scalars;
        for (int i = 0; i < n; i++) {
            points.push_back(random_point(rng));
            scalars.push_back(random_scalar(rng));
        }

        PointJacobian expected = msm_pippenger(points, scalars, window_bits);
        PointJacobian actual = gpu_pippenger(*ctx, points, scalars);

        // Different accumulation order than the CPU version (per-bucket
        // GPU threads vs. a sequential CPU scan), so results can be the
        // same affine point with a different Jacobian Z scaling -
        // point_equal (Stage 2) is the correct comparison here, not a
        // raw limb match.
        bool ok = point_equal(expected, actual);
        check(ok, "GPU Pippenger matches CPU Pippenger");
        std::printf("n=%-5d: %s\n", n, ok ? "PASS" : "FAIL");
    }

    destroy_gpu_pippenger_context(ctx);

    std::printf("\n=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
