"""BLS12-381 G1 Jacobian point arithmetic in JAX, batched for TPU.

Mirrors ../src/point.cpp exactly: dbl-2009-l doubling, add-2007-bl
addition (both a=0 specializations), point == (X, Y, Z) with Z == 0
meaning infinity. Every function operates on arrays of shape
(..., LIMBS) per coordinate - the leading batch dimension is what
JAX/XLA parallelizes across.
"""

import jax.numpy as jnp

import fp


def is_infinity(z):
    """z: (..., LIMBS) uint32. Returns (...,) bool."""
    return jnp.all(z == 0, axis=-1)


def point_double(x, y, z):
    inf = is_infinity(z)

    A = fp.fp_mul(x, x)
    B = fp.fp_mul(y, y)
    C = fp.fp_mul(B, B)
    x1_plus_b = fp.fp_add(x, B)
    D = fp.fp_double(fp.fp_sub(fp.fp_sub(fp.fp_mul(x1_plus_b, x1_plus_b), A), C))
    E = fp.fp_add(fp.fp_double(A), A)
    F = fp.fp_mul(E, E)

    x3 = fp.fp_sub(F, fp.fp_double(D))
    eight_c = fp.fp_double(fp.fp_double(fp.fp_double(C)))
    y3 = fp.fp_sub(fp.fp_mul(E, fp.fp_sub(D, x3)), eight_c)
    z3 = fp.fp_double(fp.fp_mul(y, z))

    # Infinity stays infinity.
    x3 = jnp.where(inf[..., None], x, x3)
    y3 = jnp.where(inf[..., None], y, y3)
    z3 = jnp.where(inf[..., None], z, z3)
    return x3, y3, z3


def point_add(x1, y1, z1, x2, y2, z2):
    inf1 = is_infinity(z1)
    inf2 = is_infinity(z2)

    z1z1 = fp.fp_mul(z1, z1)
    z2z2 = fp.fp_mul(z2, z2)
    u1 = fp.fp_mul(x1, z2z2)
    u2 = fp.fp_mul(x2, z1z1)
    z1_cubed = fp.fp_mul(z1z1, z1)
    z2_cubed = fp.fp_mul(z2z2, z2)
    s1 = fp.fp_mul(y1, z2_cubed)
    s2 = fp.fp_mul(y2, z1_cubed)

    same_x = jnp.all(u1 == u2, axis=-1)
    same_y = jnp.all(s1 == s2, axis=-1)

    # Generic addition (add-2007-bl); only meaningful where !same_x, but
    # computed unconditionally (branch-free, the natural style for
    # vectorized/TPU code) and then selected via jnp.where below.
    h = fp.fp_sub(u2, u1)
    two_h = fp.fp_double(h)
    i_ = fp.fp_mul(two_h, two_h)
    j_ = fp.fp_mul(h, i_)
    r = fp.fp_double(fp.fp_sub(s2, s1))
    v = fp.fp_mul(u1, i_)

    x3_add = fp.fp_sub(fp.fp_sub(fp.fp_mul(r, r), j_), fp.fp_double(v))
    y3_add = fp.fp_sub(fp.fp_mul(r, fp.fp_sub(v, x3_add)), fp.fp_double(fp.fp_mul(s1, j_)))
    z1_plus_z2 = fp.fp_add(z1, z2)
    z3_add = fp.fp_mul(fp.fp_sub(fp.fp_sub(fp.fp_mul(z1_plus_z2, z1_plus_z2), z1z1), z2z2), h)

    # Doubling result, for the same_x & same_y (P == Q) case.
    x3_dbl, y3_dbl, z3_dbl = point_double(x1, y1, z1)

    zero = fp.fp_zero(x1.shape[:-1])
    # P == -Q (same_x, !same_y) -> infinity.
    x3 = jnp.where(same_x[..., None], jnp.where(same_y[..., None], x3_dbl, zero), x3_add)
    y3 = jnp.where(same_x[..., None], jnp.where(same_y[..., None], y3_dbl, zero), y3_add)
    z3 = jnp.where(same_x[..., None], jnp.where(same_y[..., None], z3_dbl, zero), z3_add)

    # inf1 -> return P2; inf2 -> return P1 (checked last so they win).
    x3 = jnp.where(inf2[..., None], x1, jnp.where(inf1[..., None], x2, x3))
    y3 = jnp.where(inf2[..., None], y1, jnp.where(inf1[..., None], y2, y3))
    z3 = jnp.where(inf2[..., None], z1, jnp.where(inf1[..., None], z2, z3))
    return x3, y3, z3


def points_equal(x1, y1, z1, x2, y2, z2):
    """Affine-equivalence check (X1*Z2^2 == X2*Z1^2 and Y1*Z2^3 == Y2*Z1^3),
    so differently-Z-scaled Jacobian reps of the same point compare equal."""
    inf1 = is_infinity(z1)
    inf2 = is_infinity(z2)
    both_inf = inf1 & inf2

    z1z1 = fp.fp_mul(z1, z1)
    z2z2 = fp.fp_mul(z2, z2)
    lhs_x = fp.fp_mul(x1, z2z2)
    rhs_x = fp.fp_mul(x2, z1z1)
    x_eq = jnp.all(lhs_x == rhs_x, axis=-1)

    lhs_y = fp.fp_mul(y1, fp.fp_mul(z2z2, z2))
    rhs_y = fp.fp_mul(y2, fp.fp_mul(z1z1, z1))
    y_eq = jnp.all(lhs_y == rhs_y, axis=-1)

    neither_inf = ~inf1 & ~inf2
    return jnp.where(inf1 | inf2, both_inf, x_eq & y_eq & neither_inf)
