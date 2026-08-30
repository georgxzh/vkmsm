// Stage 3.1 validation: scalar multiplication and the naive MSM baseline,
// cross-checked against blst.
//
// Test vector source: blst, https://github.com/supranational/blst, tag v0.3.17.

#include <blst.h>

#include <cstdio>
#include <cstring>
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

PointJacobian from_blst(const blst_p1& p) {
    PointJacobian r;
    std::array<uint32_t, kLimbs> x, y, z;
    blst_uint32_from_fp(x.data(), &p.x);
    blst_uint32_from_fp(y.data(), &p.y);
    blst_uint32_from_fp(z.data(), &p.z);
    r.x = fp_from_plain_limbs(x);
    r.y = fp_from_plain_limbs(y);
    r.z = fp_from_plain_limbs(z);
    return r;
}

blst_p1 to_blst(const PointJacobian& p) {
    blst_p1 r;
    blst_fp_from_uint32(&r.x, fp_to_plain_limbs(p.x).data());
    blst_fp_from_uint32(&r.y, fp_to_plain_limbs(p.y).data());
    blst_fp_from_uint32(&r.z, fp_to_plain_limbs(p.z).data());
    return r;
}

Scalar random_scalar(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    Scalar s;
    for (auto& l : s) l = dist(rng);
    return s;
}

// blst expects a little-endian byte scalar; our Scalar is already
// little-endian 32-bit words, which on x86-64 is a direct byte-for-byte
// reinterpretation.
void scalar_to_bytes(const Scalar& s, uint8_t out[32]) { std::memcpy(out, s.data(), 32); }

blst_p1 random_blst_point(std::mt19937& rng) {
    Scalar s = random_scalar(rng);
    uint8_t bytes[32];
    scalar_to_bytes(s, bytes);
    blst_p1 r;
    blst_p1_mult(&r, blst_p1_generator(), bytes, 256);
    return r;
}

}  // namespace

int main() {
    std::printf("Stage 3.1: scalar multiplication + naive MSM vs blst v0.3.17\n\n");

    // --- Scalar multiplication vs blst_p1_mult ---
    {
        std::mt19937 rng(0x5CA1AB1E);
        const int kIterations = 500;
        int ok_count = 0;
        for (int i = 0; i < kIterations; i++) {
            blst_p1 base = random_blst_point(rng);
            Scalar s = random_scalar(rng);
            uint8_t bytes[32];
            scalar_to_bytes(s, bytes);

            PointJacobian p = from_blst(base);
            PointJacobian ours = point_scalar_mul(p, s);

            blst_p1 theirs;
            blst_p1_mult(&theirs, &base, bytes, 256);

            blst_p1 ours_as_blst = to_blst(ours);
            bool ok = blst_p1_is_equal(&ours_as_blst, &theirs);
            check(ok, "scalar * P matches blst_p1_mult");
            check(point_on_curve(ours), "scalar * P result is on curve");
            if (ok) ok_count++;
        }
        std::printf("Random scalar multiplication (%d iterations): %d/%d passed\n", kIterations,
                     ok_count, kIterations);
    }

    // --- Naive MSM vs blst (sum of individually-computed blst_p1_mult results) ---
    {
        std::mt19937 rng(0xF00DBABE);
        for (int n : {1, 2, 4, 8, 16, 64, 256}) {
            std::vector<PointJacobian> points;
            std::vector<Scalar> scalars;
            blst_p1 blst_acc = *blst_p1_generator();
            // Start the blst accumulator at infinity: multiply generator by 0.
            {
                uint8_t zero[32] = {0};
                blst_p1_mult(&blst_acc, blst_p1_generator(), zero, 256);
            }

            for (int i = 0; i < n; i++) {
                blst_p1 bp = random_blst_point(rng);
                Scalar s = random_scalar(rng);
                uint8_t bytes[32];
                scalar_to_bytes(s, bytes);

                points.push_back(from_blst(bp));
                scalars.push_back(s);

                blst_p1 term;
                blst_p1_mult(&term, &bp, bytes, 256);
                blst_p1_add_or_double(&blst_acc, &blst_acc, &term);
            }

            PointJacobian ours = msm_naive(points, scalars);
            blst_p1 ours_as_blst = to_blst(ours);
            bool ok = blst_p1_is_equal(&ours_as_blst, &blst_acc);
            check(ok, "msm_naive matches blst (sum of individual scalar mults)");
            check(point_on_curve(ours), "msm_naive result is on curve");
            std::printf("msm_naive, n=%-4d: %s\n", n, ok ? "PASS" : "FAIL");
        }
        std::printf("\n");
    }

    std::printf("=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
