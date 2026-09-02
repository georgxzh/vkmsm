"""BLS12-381 F_p arithmetic in JAX, batched for TPU.

Mirrors ../src/fp.cpp: Montgomery form, CIOS multiplication, 12 x 32-bit
limbs. Same algorithm and limb layout as the C++/GLSL version - ported
from explicit carry-chain ops (mac/adc, uaddCarry/usubBorrow/umulExtended)
to JAX's vectorized uint64 arithmetic. Every function operates on arrays
of shape (..., LIMBS): the leading batch dimension is what JAX/XLA
parallelizes across on TPU, playing the same role "one Vulkan thread per
element" played in the GPU shaders.

The modulus is derived independently from BLS12-381's defining seed
u = -0xd201000000010000 via p(u) = (u-1)^2*(u^4-u^2+1)/3 + u, rather than
hand-transcribed - see derive_constants.py for the cross-check (prime,
correct bit length, matches the C++ project's hardcoded constant).
"""

import jax

jax.config.update("jax_enable_x64", True)  # needed for uint64 intermediates
import jax.numpy as jnp
import numpy as np

LIMBS = 12
LIMB_MASK = np.uint64((1 << 32) - 1)


def _derive_modulus() -> int:
    u = -0xD201000000010000
    return (u - 1) ** 2 * (u**4 - u**2 + 1) // 3 + u


P = _derive_modulus()
assert P.bit_length() == 381, "derived modulus has the wrong bit length"

R = 1 << (32 * LIMBS)
R_MOD_P = R % P
R2_MOD_P = (R * R) % P
N0 = pow(-P, -1, 1 << 32)  # -p^-1 mod 2^32, exact via Python's modular inverse


def int_to_limbs(x: int, n: int = LIMBS) -> np.ndarray:
    return np.array([(x >> (32 * i)) & 0xFFFFFFFF for i in range(n)], dtype=np.uint32)


def limbs_to_int(limbs) -> int:
    x = 0
    for i, l in enumerate(np.asarray(limbs).tolist()):
        x |= int(l) << (32 * i)
    return x


# Self-check mirroring fp.cpp's runtime assertion: p0 * n0 == -1 (mod 2^32).
assert (int(int_to_limbs(P)[0]) * N0) % (1 << 32) == (1 << 32) - 1

MODULUS_LIMBS = jnp.asarray(int_to_limbs(P))
R_MOD_P_LIMBS = jnp.asarray(int_to_limbs(R_MOD_P))  # Montgomery form of 1
R2_MOD_P_LIMBS = jnp.asarray(int_to_limbs(R2_MOD_P))


def _u64(a):
    return a.astype(jnp.uint64)


def _geq(a, b):
    """a, b: (..., LIMBS) uint32. Returns (...,) bool: a >= b, MSB-first compare."""
    still_eq = jnp.ones(a.shape[:-1], dtype=jnp.bool_)
    gt = jnp.zeros(a.shape[:-1], dtype=jnp.bool_)
    for i in range(LIMBS - 1, -1, -1):
        gt_i = a[..., i] > b[..., i]
        eq_i = a[..., i] == b[..., i]
        gt = jnp.where(still_eq, gt_i, gt)
        still_eq = still_eq & eq_i
    return gt | still_eq


def _raw_add(a, b):
    a64, b64 = _u64(a), _u64(b)
    carry = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
    limbs = []
    for i in range(LIMBS):
        s = a64[..., i] + b64[..., i] + carry
        limbs.append((s & LIMB_MASK).astype(jnp.uint32))
        carry = s >> np.uint64(32)
    return jnp.stack(limbs, axis=-1), carry


def _raw_sub(a, b):
    a64, b64 = _u64(a), _u64(b)
    borrow = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
    limbs = []
    for i in range(LIMBS):
        bi = b64[..., i] + borrow
        borrow = (a64[..., i] < bi).astype(jnp.uint64)
        d = (a64[..., i] - bi) & LIMB_MASK
        limbs.append(d.astype(jnp.uint32))
    return jnp.stack(limbs, axis=-1), borrow


def _cond_sub_mod(r):
    ge = _geq(r, MODULUS_LIMBS)
    sub_r, _ = _raw_sub(r, MODULUS_LIMBS)
    return jnp.where(ge[..., None], sub_r, r)


def _reduce_final(r):
    # CIOS bound guarantees result < 2p, so one conditional subtraction
    # suffices; a second pass is a defensive margin (mirrors fp.cpp's
    # `while` loop), cheap and harmless if unneeded.
    return _cond_sub_mod(_cond_sub_mod(r))


def fp_add(a, b):
    r, _ = _raw_add(a, b)
    return _reduce_final(r)


def fp_sub(a, b):
    r, borrow = _raw_sub(a, b)
    add_r, _ = _raw_add(r, MODULUS_LIMBS)
    return jnp.where((borrow != 0)[..., None], add_r, r)


def fp_mul(a, b):
    """CIOS Montgomery multiplication - see src/fp.cpp for the derivation."""
    a64, b64, mod64 = _u64(a), _u64(b), _u64(MODULUS_LIMBS)
    batch_shape = a.shape[:-1]
    zero = jnp.zeros(batch_shape, dtype=jnp.uint64)
    t = [zero] * (LIMBS + 2)

    for i in range(LIMBS):
        carry = zero
        for j in range(LIMBS):
            prod = a64[..., j] * b64[..., i] + t[j] + carry
            t[j] = prod & LIMB_MASK
            carry = prod >> np.uint64(32)
        s = t[LIMBS] + carry
        t[LIMBS] = s & LIMB_MASK
        t[LIMBS + 1] = t[LIMBS + 1] + (s >> np.uint64(32))

        m = (t[0] * jnp.uint64(N0)) & LIMB_MASK

        carry3 = zero
        prod0 = m * mod64[0] + t[0] + carry3
        carry3 = prod0 >> np.uint64(32)  # low word discarded: == 0 by construction
        for j in range(1, LIMBS):
            prod = m * mod64[j] + t[j] + carry3
            t[j - 1] = prod & LIMB_MASK
            carry3 = prod >> np.uint64(32)
        s2 = t[LIMBS] + carry3
        t[LIMBS - 1] = s2 & LIMB_MASK
        t[LIMBS] = t[LIMBS + 1] + (s2 >> np.uint64(32))
        t[LIMBS + 1] = zero

    result = jnp.stack([t[k].astype(jnp.uint32) for k in range(LIMBS)], axis=-1)
    return _reduce_final(result)


def fp_double(a):
    return fp_add(a, a)


def to_montgomery(plain):
    """plain: (..., LIMBS) uint32, values < p. Returns Montgomery form."""
    r2 = jnp.broadcast_to(R2_MOD_P_LIMBS, plain.shape)
    return fp_mul(plain, r2)


def from_montgomery(mont):
    one = jnp.zeros(mont.shape, dtype=jnp.uint32).at[..., 0].set(1)
    return fp_mul(mont, one)


def fp_one():
    return R_MOD_P_LIMBS


def fp_zero(shape):
    return jnp.zeros(shape + (LIMBS,), dtype=jnp.uint32)
