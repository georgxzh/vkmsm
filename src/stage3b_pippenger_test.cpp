// Stage 3.2 validation: Pippenger's bucket method MSM, cross-checked
// against the already-validated naive MSM baseline (msm_naive), on random
// test cases of increasing size, per the project's ground rules ("validate
// its output matches the naive MSM exactly on many random test cases of
// increasing size before trusting it").
//
// This stage does NOT re-derive correctness from blst directly - msm_naive
// itself was already validated against blst in stage3a_msm_naive_test.cpp,
// so it's a trustworthy oracle for Pippenger.

#include <blst.h>

#include <cstdio>
#include <random>
#include <vector>

#include "fp.hpp"
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

Scalar random_scalar(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    Scalar s;
    for (auto& l : s) l = dist(rng);
    return s;
}

// The G1 generator's coordinates, sourced from blst rather than
// hand-transcribed (a 384-bit hex constant is exactly the kind of value
// that's easy to mistype and hard to catch - blst_p1_generator() is
// already a trusted, validated source, per Stage 2/3.1).
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

// A random point on G1, generated as (random scalar) * generator using our
// own already-validated point_scalar_mul (Stage 3.1).
PointJacobian random_point(std::mt19937& rng) {
    Scalar s = random_scalar(rng);
    return point_scalar_mul(generator(), s);
}

}  // namespace

int main() {
    std::printf("Stage 3.2: Pippenger's bucket method vs naive MSM\n\n");

    // --- Sanity: our hand-copied generator constant is actually on-curve
    // and non-infinity, so random_point() above is producing real points. ---
    {
        std::mt19937 rng(1);
        PointJacobian p = random_point(rng);
        check(point_on_curve(p), "generator-derived random point is on curve");
        check(!point_is_infinity(p), "generator-derived random point is not infinity");
    }

    std::mt19937 rng(0xB0CE7113);

    for (int n : {0, 1, 2, 3, 5, 8, 16, 64, 256, 1024}) {
        std::vector<PointJacobian> points;
        std::vector<Scalar> scalars;
        for (int i = 0; i < n; i++) {
            points.push_back(random_point(rng));
            scalars.push_back(random_scalar(rng));
        }

        PointJacobian expected = msm_naive(points, scalars);
        PointJacobian actual = msm_pippenger(points, scalars);  // default window_bits=8

        bool ok = point_equal(expected, actual);
        check(ok, "msm_pippenger matches msm_naive (window_bits=8)");
        std::printf("n=%-5d window_bits=8:  %s\n", n, ok ? "PASS" : "FAIL");
    }
    std::printf("\n");

    // Also check a couple of other window sizes at a fixed size, to make
    // sure the window_bits parameterization itself is correct, not just
    // the default.
    {
        std::vector<PointJacobian> points;
        std::vector<Scalar> scalars;
        for (int i = 0; i < 300; i++) {
            points.push_back(random_point(rng));
            scalars.push_back(random_scalar(rng));
        }
        PointJacobian expected = msm_naive(points, scalars);

        for (int wb : {1, 4, 6, 16}) {
            PointJacobian actual = msm_pippenger(points, scalars, wb);
            bool ok = point_equal(expected, actual);
            check(ok, "msm_pippenger matches msm_naive (varying window_bits)");
            std::printf("n=300   window_bits=%-2d: %s\n", wb, ok ? "PASS" : "FAIL");
        }
    }

    std::printf("\n=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
