// Stage 2 validation: our BLS12-381 G1 Jacobian point arithmetic,
// cross-checked against blst.
//
// Test vector source: blst, https://github.com/supranational/blst, tag v0.3.17.
// Random test points are generated as (random scalar) * (blst's G1
// generator) - this is a legitimate use of the oracle library, since it
// only produces *inputs* (valid points on the curve); the add/double
// operations under test are entirely our own implementation.

#include <blst.h>

#include <array>
#include <cstdio>
#include <random>

#include "fp.hpp"
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

// A random point on G1: (random 256-bit scalar) * generator, via blst.
blst_p1 random_blst_point(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    uint8_t scalar[32];
    for (auto& b : scalar) b = static_cast<uint8_t>(dist(rng));
    blst_p1 r;
    blst_p1_mult(&r, blst_p1_generator(), scalar, 256);
    return r;
}

}  // namespace

int main() {
    std::printf("Stage 2: BLS12-381 G1 point arithmetic vs blst v0.3.17\n\n");

    // --- Generator sanity: our on-curve check agrees with blst's on a known point ---
    {
        PointJacobian gen = from_blst(*blst_p1_generator());
        check(point_on_curve(gen), "generator point is on curve (our check)");
        check(blst_p1_on_curve(blst_p1_generator()), "generator point is on curve (blst's check)");
        check(!point_is_infinity(gen), "generator is not infinity");
    }

    // --- Infinity identities ---
    {
        PointJacobian inf = point_infinity();
        check(point_is_infinity(inf), "point_infinity() reports itself as infinity");
        check(point_on_curve(inf), "infinity trivially satisfies on-curve check");

        PointJacobian gen = from_blst(*blst_p1_generator());
        check(point_equal(point_add(inf, gen), gen), "infinity + G == G");
        check(point_equal(point_add(gen, inf), gen), "G + infinity == G");
        check(point_equal(point_double(inf), inf), "double(infinity) == infinity");
    }

    // --- P + (-P) == infinity ---
    {
        PointJacobian gen = from_blst(*blst_p1_generator());
        PointJacobian neg_gen = gen;
        neg_gen.y = fp_sub(fp_zero(), gen.y);
        check(point_on_curve(neg_gen), "-G is on curve");
        PointJacobian sum = point_add(gen, neg_gen);
        check(point_is_infinity(sum), "G + (-G) == infinity");
    }

    // --- Doubling vs blst ---
    {
        std::mt19937 rng(0xD00B1E);
        const int kIterations = 2000;
        int ok_count = 0;
        for (int i = 0; i < kIterations; i++) {
            blst_p1 bp = random_blst_point(rng);
            PointJacobian p = from_blst(bp);
            check(point_on_curve(p), "random point is on curve");

            PointJacobian ours = point_double(p);
            blst_p1 theirs;
            blst_p1_double(&theirs, &bp);

            blst_p1 ours_as_blst = to_blst(ours);
            bool ok = blst_p1_is_equal(&ours_as_blst, &theirs);
            check(ok, "double(P) matches blst");
            check(point_on_curve(ours), "double(P) result is on curve");
            if (ok) ok_count++;
        }
        std::printf("Random doubling (%d iterations): %d/%d passed\n", kIterations, ok_count,
                     kIterations);
    }

    // --- Addition vs blst (distinct random points) ---
    {
        std::mt19937 rng(0xADD1710);
        const int kIterations = 2000;
        int ok_count = 0;
        for (int i = 0; i < kIterations; i++) {
            blst_p1 ba = random_blst_point(rng);
            blst_p1 bb = random_blst_point(rng);
            PointJacobian a = from_blst(ba);
            PointJacobian b = from_blst(bb);

            PointJacobian ours = point_add(a, b);
            blst_p1 theirs;
            blst_p1_add_or_double(&theirs, &ba, &bb);

            blst_p1 ours_as_blst = to_blst(ours);
            bool ok = blst_p1_is_equal(&ours_as_blst, &theirs);
            check(ok, "P + Q matches blst");
            check(point_on_curve(ours), "P + Q result is on curve");
            if (ok) ok_count++;
        }
        std::printf("Random addition (%d iterations): %d/%d passed\n", kIterations, ok_count,
                     kIterations);
    }

    // --- Addition of a point to itself (P + P) should match doubling ---
    {
        std::mt19937 rng(0x5E1F5E1F);
        const int kIterations = 500;
        int ok_count = 0;
        for (int i = 0; i < kIterations; i++) {
            blst_p1 bp = random_blst_point(rng);
            PointJacobian p = from_blst(bp);

            PointJacobian via_add = point_add(p, p);
            PointJacobian via_double = point_double(p);
            bool ok = point_equal(via_add, via_double);
            check(ok, "P + P == double(P)");
            if (ok) ok_count++;
        }
        std::printf("P + P == double(P) (%d iterations): %d/%d passed\n\n", kIterations, ok_count,
                     kIterations);
    }

    // --- Different Z representations of the same affine point compare equal ---
    {
        std::mt19937 rng(0x2A2A2A2A);
        blst_p1 bp = random_blst_point(rng);
        PointJacobian p = from_blst(bp);

        // Rescale by a random nonzero factor lambda: (X*l^2, Y*l^3, Z*l)
        // represents the same affine point for any nonzero lambda.
        std::array<uint32_t, kLimbs> lambda_plain{};
        lambda_plain[0] = 12345;
        Fp lambda = fp_from_plain_limbs(lambda_plain);
        Fp l2 = fp_mul(lambda, lambda);
        Fp l3 = fp_mul(l2, lambda);

        PointJacobian rescaled;
        rescaled.x = fp_mul(p.x, l2);
        rescaled.y = fp_mul(p.y, l3);
        rescaled.z = fp_mul(p.z, lambda);

        check(point_on_curve(rescaled), "rescaled point is still on curve");
        check(point_equal(p, rescaled), "differently-scaled Jacobian reps of same point compare equal");
    }

    std::printf("=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
