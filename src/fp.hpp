// BLS12-381 base field (F_p) arithmetic, CPU reference implementation.
//
// Field elements are represented as 12 x 32-bit limbs (little-endian: limb[0]
// is the least significant word), matching the representation we'll later
// port to GLSL compute shaders, where 32-bit is the practical integer width.
//
// Internally, values are stored in Montgomery form (x*R mod p, where
// R = 2^384) so that multiplication can use the CIOS algorithm instead of
// a separate multiply-then-divide-by-p step. This is standard practice for
// modular arithmetic in cryptography; see fp.cpp for the derivation of the
// Montgomery constants.

#pragma once

#include <array>
#include <cstdint>

namespace vkmsm {

constexpr int kLimbs = 12;

struct Fp {
    std::array<uint32_t, kLimbs> limb{};
};

// The BLS12-381 base field modulus, as plain (non-Montgomery) limbs.
const Fp& fp_modulus();

// Additive and multiplicative identities, in Montgomery form.
Fp fp_zero();
Fp fp_one();

// Converts a plain (non-Montgomery) 12-limb integer, assumed < modulus,
// into Montgomery form.
Fp fp_from_plain_limbs(const std::array<uint32_t, kLimbs>& plain);

// Converts a Montgomery-form value back to a plain 12-limb integer.
std::array<uint32_t, kLimbs> fp_to_plain_limbs(const Fp& x);

// a + b mod p, a - b mod p. Inputs/outputs are in whatever form they share
// (Montgomery or plain) since addition and subtraction don't interact with
// the Montgomery scaling factor.
Fp fp_add(const Fp& a, const Fp& b);
Fp fp_sub(const Fp& a, const Fp& b);

// Montgomery product: if a = A*R mod p and b = B*R mod p, returns A*B*R mod p.
Fp fp_mul(const Fp& a, const Fp& b);

// Multiplicative inverse via Fermat's little theorem (a^(p-2) mod p).
// a must be nonzero.
Fp fp_inverse(const Fp& a);

bool fp_equal(const Fp& a, const Fp& b);
bool fp_is_zero(const Fp& a);

}  // namespace vkmsm
