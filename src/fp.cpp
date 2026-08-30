#include "fp.hpp"

#include <stdexcept>

namespace vkmsm {
namespace {

// The BLS12-381 base field modulus p, little-endian 32-bit limbs.
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0
//       f6b0f6241eabfffeb153ffffb9feffffffffaaab
// This is the one hand-transcribed constant in the whole field-arithmetic
// layer; everything else (Montgomery R, R^2, n0) is *derived* from it at
// startup rather than separately transcribed, and stage1_field_test.cpp
// cross-checks this exact value against blst before trusting anything else.
const std::array<uint32_t, kLimbs>& modulus_plain() {
    static const std::array<uint32_t, kLimbs> m = {
        0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
        0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
        0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea,
    };
    return m;
}

using Limbs = std::array<uint32_t, kLimbs>;

// r = a + b, plus the final carry-out bit (which is always 0 for the field
// values we operate on here, since two values < p leave headroom below 2^384).
Limbs raw_add(const Limbs& a, const Limbs& b, uint32_t& carry_out) {
    Limbs r{};
    uint64_t carry = 0;
    for (int i = 0; i < kLimbs; i++) {
        uint64_t s = static_cast<uint64_t>(a[i]) + b[i] + carry;
        r[i] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
    carry_out = static_cast<uint32_t>(carry);
    return r;
}

// r = a - b, plus a borrow-out flag (1 if a < b).
Limbs raw_sub(const Limbs& a, const Limbs& b, uint32_t& borrow_out) {
    Limbs r{};
    int64_t borrow = 0;
    for (int i = 0; i < kLimbs; i++) {
        int64_t d = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]) - borrow;
        if (d < 0) {
            d += (static_cast<int64_t>(1) << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        r[i] = static_cast<uint32_t>(d);
    }
    borrow_out = static_cast<uint32_t>(borrow);
    return r;
}

bool raw_geq(const Limbs& a, const Limbs& b) {
    for (int i = kLimbs - 1; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] > b[i];
    }
    return true;  // equal
}

// (x * 2^n) mod p, via n repeated doublings with conditional subtraction.
// Used only to bootstrap the Montgomery constants R mod p and R^2 mod p,
// so we never have to hand-transcribe those as separate hex literals.
Limbs mod_double_n(Limbs x, int n) {
    const auto& mod = modulus_plain();
    for (int i = 0; i < n; i++) {
        uint32_t carry;
        Limbs r = raw_add(x, x, carry);
        if (raw_geq(r, mod)) {
            uint32_t borrow;
            r = raw_sub(r, mod, borrow);
        }
        x = r;
    }
    return x;
}

struct MontgomeryConstants {
    Limbs modulus;
    Limbs r_mod_p;   // R mod p == Montgomery form of the integer 1
    Limbs r2_mod_p;  // R^2 mod p, used to convert plain values into Montgomery form
    uint32_t n0;      // -p^-1 mod 2^32, the CIOS reduction constant
};

const MontgomeryConstants& constants() {
    static const MontgomeryConstants c = [] {
        MontgomeryConstants mc{};
        mc.modulus = modulus_plain();

        Limbs one_plain{};
        one_plain[0] = 1;
        mc.r_mod_p = mod_double_n(one_plain, kLimbs * 32);
        mc.r2_mod_p = mod_double_n(mc.r_mod_p, kLimbs * 32);

        // Newton-Raphson iteration for the modular inverse of an odd number
        // mod 2^32: each iteration doubles the number of correct bits, so
        // 5 iterations (1 -> 2 -> 4 -> 8 -> 16 -> 32 bits) suffices.
        uint32_t p0 = mc.modulus[0];
        uint32_t inv = 1;
        for (int i = 0; i < 5; i++) {
            inv = inv * (2u - p0 * inv);
        }
        mc.n0 = static_cast<uint32_t>(0u - inv);

        // Self-check: by construction p0 * n0 must be === -1 (mod 2^32).
        // This independently validates the derivation above at startup.
        uint32_t check = p0 * mc.n0;
        if (check != 0xFFFFFFFFu) {
            throw std::runtime_error("fp: Montgomery n0 derivation failed self-check");
        }
        return mc;
    }();
    return c;
}

inline uint32_t mac(uint32_t& carry, uint32_t a, uint32_t b, uint32_t addend) {
    uint64_t r = static_cast<uint64_t>(a) * b + addend + carry;
    carry = static_cast<uint32_t>(r >> 32);
    return static_cast<uint32_t>(r);
}

inline uint32_t adc(uint32_t& carry, uint32_t a, uint32_t b) {
    uint64_t r = static_cast<uint64_t>(a) + b + carry;
    carry = static_cast<uint32_t>(r >> 32);
    return static_cast<uint32_t>(r);
}

}  // namespace

const Fp& fp_modulus() {
    static const Fp m = [] {
        Fp f;
        f.limb = constants().modulus;
        return f;
    }();
    return m;
}

Fp fp_zero() { return Fp{}; }

Fp fp_one() {
    Fp f;
    f.limb = constants().r_mod_p;
    return f;
}

Fp fp_add(const Fp& a, const Fp& b) {
    uint32_t carry;
    Limbs r = raw_add(a.limb, b.limb, carry);
    if (raw_geq(r, fp_modulus().limb)) {
        uint32_t borrow;
        r = raw_sub(r, fp_modulus().limb, borrow);
    }
    Fp result;
    result.limb = r;
    return result;
}

Fp fp_sub(const Fp& a, const Fp& b) {
    uint32_t borrow;
    Limbs r = raw_sub(a.limb, b.limb, borrow);
    if (borrow) {
        uint32_t carry;
        r = raw_add(r, fp_modulus().limb, carry);
    }
    Fp result;
    result.limb = r;
    return result;
}

// CIOS Montgomery multiplication (Koc, Acar, Kaliski). Computes
// a * b * R^-1 mod p in a single combined multiply-and-reduce pass,
// interleaving each row of the schoolbook product with a reduction step
// so the running total never needs more than kLimbs+2 words.
Fp fp_mul(const Fp& a, const Fp& b) {
    const Limbs& mod = constants().modulus;
    uint32_t n0 = constants().n0;
    uint32_t t[kLimbs + 2] = {0};

    for (int i = 0; i < kLimbs; i++) {
        // t += a * b[i]
        uint32_t carry = 0;
        for (int j = 0; j < kLimbs; j++) {
            t[j] = mac(carry, a.limb[j], b.limb[i], t[j]);
        }
        uint32_t carry2 = 0;
        t[kLimbs] = adc(carry2, t[kLimbs], carry);
        t[kLimbs + 1] += carry2;

        // m chosen so that t + m*modulus is divisible by 2^32
        uint32_t m = t[0] * n0;

        // t += m * modulus, then shift right by one word (the low word is
        // now guaranteed zero by construction of m, so the shift and the
        // add are fused into one loop that writes directly to t[j-1]).
        uint32_t carry3 = 0;
        mac(carry3, m, mod[0], t[0]);  // low word discarded: == 0 by construction
        for (int j = 1; j < kLimbs; j++) {
            t[j - 1] = mac(carry3, m, mod[j], t[j]);
        }
        uint32_t carry4 = 0;
        t[kLimbs - 1] = adc(carry4, t[kLimbs], carry3);
        t[kLimbs] = t[kLimbs + 1] + carry4;
        t[kLimbs + 1] = 0;
    }

    Fp result;
    for (int i = 0; i < kLimbs; i++) result.limb[i] = t[i];

    // Standard CIOS bound: result < 2p here, so at most one subtraction
    // is needed to reach the canonical range [0, p). The while (rather
    // than a single if) is a cheap defensive margin.
    while (raw_geq(result.limb, mod)) {
        uint32_t borrow;
        result.limb = raw_sub(result.limb, mod, borrow);
    }
    return result;
}

Fp fp_inverse(const Fp& a) {
    // a^(p-2) mod p, via Fermat's little theorem, computed entirely in
    // Montgomery form via repeated squaring (no separate extended-Euclid
    // routine needed - it reuses the already-validated fp_mul).
    Limbs two{};
    two[0] = 2;
    uint32_t borrow;
    Limbs exponent = raw_sub(constants().modulus, two, borrow);

    Fp result = fp_one();
    for (int limb_i = kLimbs - 1; limb_i >= 0; limb_i--) {
        for (int bit = 31; bit >= 0; bit--) {
            result = fp_mul(result, result);
            if ((exponent[limb_i] >> bit) & 1u) {
                result = fp_mul(result, a);
            }
        }
    }
    return result;
}

bool fp_equal(const Fp& a, const Fp& b) { return a.limb == b.limb; }

bool fp_is_zero(const Fp& a) {
    for (uint32_t l : a.limb) {
        if (l != 0) return false;
    }
    return true;
}

Fp fp_from_plain_limbs(const std::array<uint32_t, kLimbs>& plain) {
    Fp raw;
    raw.limb = plain;
    Fp r2;
    r2.limb = constants().r2_mod_p;
    return fp_mul(raw, r2);
}

std::array<uint32_t, kLimbs> fp_to_plain_limbs(const Fp& x) {
    Fp one_plain;
    one_plain.limb[0] = 1;
    return fp_mul(x, one_plain).limb;
}

}  // namespace vkmsm
