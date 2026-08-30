// Stage 1 validation: our F_p (BLS12-381 base field) implementation,
// cross-checked against blst - a well-established, independently-authored
// library - as the CPU-side oracle.
//
// Test vector source: blst, https://github.com/supranational/blst, tag v0.3.17
// (pinned in CMakeLists.txt via FetchContent for reproducibility).

#include <blst.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>

#include "fp.hpp"

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

// Lexicographic compare of plain 12-limb integers (MSB first), used only to
// reject random samples >= p when generating test inputs.
bool plain_less_than(const std::array<uint32_t, kLimbs>& a, const std::array<uint32_t, kLimbs>& b) {
    for (int i = kLimbs - 1; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;
}

std::array<uint32_t, kLimbs> random_field_element(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    std::array<uint32_t, kLimbs> x{};
    const auto& mod = fp_modulus().limb;
    do {
        for (auto& l : x) l = dist(rng);
    } while (!plain_less_than(x, mod));
    return x;
}

blst_fp to_blst(const std::array<uint32_t, kLimbs>& plain) {
    blst_fp f;
    blst_fp_from_uint32(&f, plain.data());
    return f;
}

std::array<uint32_t, kLimbs> from_blst(const blst_fp& f) {
    std::array<uint32_t, kLimbs> out{};
    blst_uint32_from_fp(out.data(), &f);
    return out;
}

bool plain_equal(const std::array<uint32_t, kLimbs>& a, const std::array<uint32_t, kLimbs>& b) {
    return a == b;
}

}  // namespace

int main() {
    std::printf("Stage 1: BLS12-381 F_p arithmetic vs blst v0.3.17\n\n");

    // --- Check 0: does our hardcoded modulus even match blst's? ---
    // If this fails, every other check below is meaningless, so it runs
    // first and on its own.
    {
        std::array<uint32_t, kLimbs> p_minus_1 = fp_modulus().limb;
        // p is odd (it's prime), so subtracting 1 from limb[0] never borrows.
        p_minus_1[0] -= 1;

        blst_fp a = to_blst(p_minus_1);
        blst_fp one;
        std::array<uint32_t, kLimbs> one_plain{};
        one_plain[0] = 1;
        one = to_blst(one_plain);

        blst_fp sum;
        blst_fp_add(&sum, &a, &one);
        auto sum_plain = from_blst(sum);

        bool modulus_matches = true;
        for (uint32_t l : sum_plain) {
            if (l != 0) modulus_matches = false;
        }
        check(modulus_matches, "hardcoded modulus matches blst's ((p-1) + 1 == 0 under blst)");
        std::printf("Modulus cross-check vs blst: %s\n\n", modulus_matches ? "PASS" : "FAIL");
        if (!modulus_matches) {
            std::printf("Hardcoded modulus is wrong - aborting further tests.\n");
            return 1;
        }
    }

    // --- Check: limb ordering sanity (limb[0] = least significant word) ---
    {
        std::array<uint32_t, kLimbs> five{};
        five[0] = 5;
        blst_fp f = to_blst(five);
        uint8_t bytes[48];
        blst_bendian_from_fp(bytes, &f);
        bool ok = true;
        for (int i = 0; i < 47; i++) {
            if (bytes[i] != 0) ok = false;
        }
        if (bytes[47] != 5) ok = false;
        check(ok, "limb[0] is the least-significant word (matches blst's convention)");
        std::printf("Limb ordering sanity check: %s\n\n", ok ? "PASS" : "FAIL");
    }

    // --- Check: round-trip conversion (plain -> Montgomery -> plain) ---
    {
        std::mt19937 rng(0xC0FFEE);
        for (int i = 0; i < 1000; i++) {
            auto plain = random_field_element(rng);
            Fp mont = fp_from_plain_limbs(plain);
            auto back = fp_to_plain_limbs(mont);
            check(plain_equal(plain, back), "round-trip plain -> Montgomery -> plain");
        }
        std::printf("Round-trip conversion (1000 random values): %d/%d passed\n\n",
                     g_checks, g_checks);
    }

    // --- Check: fp_one() / fp_zero() match plain 1 / 0 ---
    {
        auto one_plain = fp_to_plain_limbs(fp_one());
        std::array<uint32_t, kLimbs> expected_one{};
        expected_one[0] = 1;
        check(plain_equal(one_plain, expected_one), "fp_one() converts to plain value 1");

        auto zero_plain = fp_to_plain_limbs(fp_zero());
        std::array<uint32_t, kLimbs> expected_zero{};
        check(plain_equal(zero_plain, expected_zero), "fp_zero() converts to plain value 0");
    }

    // --- Random add / sub / mul vs blst ---
    {
        std::mt19937 rng(12345);
        const int kIterations = 20000;
        int add_ok = 0, sub_ok = 0, mul_ok = 0;

        for (int i = 0; i < kIterations; i++) {
            auto a_plain = random_field_element(rng);
            auto b_plain = random_field_element(rng);

            Fp a = fp_from_plain_limbs(a_plain);
            Fp b = fp_from_plain_limbs(b_plain);

            blst_fp ba = to_blst(a_plain);
            blst_fp bb = to_blst(b_plain);

            // add
            {
                Fp ours = fp_add(a, b);
                blst_fp theirs;
                blst_fp_add(&theirs, &ba, &bb);
                bool ok = plain_equal(fp_to_plain_limbs(ours), from_blst(theirs));
                check(ok, "a + b matches blst");
                if (ok) add_ok++;
            }
            // sub
            {
                Fp ours = fp_sub(a, b);
                blst_fp theirs;
                blst_fp_sub(&theirs, &ba, &bb);
                bool ok = plain_equal(fp_to_plain_limbs(ours), from_blst(theirs));
                check(ok, "a - b matches blst");
                if (ok) sub_ok++;
            }
            // mul
            {
                Fp ours = fp_mul(a, b);
                blst_fp theirs;
                blst_fp_mul(&theirs, &ba, &bb);
                bool ok = plain_equal(fp_to_plain_limbs(ours), from_blst(theirs));
                check(ok, "a * b matches blst");
                if (ok) mul_ok++;
            }
        }
        std::printf("Random add: %d/%d passed\n", add_ok, kIterations);
        std::printf("Random sub: %d/%d passed\n", sub_ok, kIterations);
        std::printf("Random mul: %d/%d passed\n\n", mul_ok, kIterations);
    }

    // --- Inversion vs blst (slower, fewer iterations) ---
    {
        std::mt19937 rng(999);
        const int kIterations = 2000;
        int inv_ok = 0, selfcheck_ok = 0;

        for (int i = 0; i < kIterations; i++) {
            std::array<uint32_t, kLimbs> a_plain;
            do {
                a_plain = random_field_element(rng);
            } while (fp_is_zero(fp_from_plain_limbs(a_plain)));

            Fp a = fp_from_plain_limbs(a_plain);
            blst_fp ba = to_blst(a_plain);

            Fp ours_inv = fp_inverse(a);
            blst_fp theirs_inv;
            blst_fp_inverse(&theirs_inv, &ba);

            bool ok = plain_equal(fp_to_plain_limbs(ours_inv), from_blst(theirs_inv));
            check(ok, "a^-1 matches blst");
            if (ok) inv_ok++;

            // a * a^-1 should be 1
            Fp product = fp_mul(a, ours_inv);
            bool self_ok = plain_equal(fp_to_plain_limbs(product), {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            check(self_ok, "a * a^-1 == 1");
            if (self_ok) selfcheck_ok++;
        }
        std::printf("Random inverse vs blst: %d/%d passed\n", inv_ok, kIterations);
        std::printf("a * a^-1 == 1 self-check: %d/%d passed\n\n", selfcheck_ok, kIterations);
    }

    // --- Edge cases: 0, 1, p-1 ---
    {
        std::array<uint32_t, kLimbs> zero{}, one{}, p_minus_1 = fp_modulus().limb;
        one[0] = 1;
        p_minus_1[0] -= 1;

        for (auto& [name, x_plain] : {std::pair{"0", zero}, std::pair{"1", one},
                                       std::pair{"p-1", p_minus_1}}) {
            Fp x = fp_from_plain_limbs(x_plain);
            blst_fp bx = to_blst(x_plain);

            Fp sum = fp_add(x, fp_one());
            blst_fp one_blst = to_blst(one);
            blst_fp bsum;
            blst_fp_add(&bsum, &bx, &one_blst);
            check(plain_equal(fp_to_plain_limbs(sum), from_blst(bsum)),
                  (std::string(name) + " + 1 matches blst").c_str());
        }
    }

    std::printf("=====================================\n");
    std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
