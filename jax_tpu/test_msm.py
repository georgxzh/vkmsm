"""Validates jax_tpu/msm.py: scalar multiplication against py_ecc's
multiply(), naive MSM against a sum of py_ecc multiplies, and Pippenger
against the (already-validated-in-this-file) naive MSM - same structure
as src/stage3a_msm_naive_test.cpp / stage3b_pippenger_test.cpp."""

import random

import jax.numpy as jnp
from py_ecc.optimized_bls12_381 import G1, add, multiply

import fp
import msm
import point
from test_point import to_jax_point


def from_jax_point(x, y, z):
    plain = fp.from_montgomery(jnp.stack([x, y, z]))
    return (fp.limbs_to_int(plain[0]), fp.limbs_to_int(plain[1]), fp.limbs_to_int(plain[2]))


def points_match(ours, theirs_pyecc):
    p = fp.P
    x1, y1, z1 = ours
    x2, y2, z2 = (int(c) % p for c in theirs_pyecc)
    if z1 == 0 or z2 == 0:
        return z1 == 0 and z2 == 0
    z1_inv = pow(z1, -1, p)
    ax1 = (x1 * z1_inv * z1_inv) % p
    ay1 = (y1 * z1_inv**3) % p
    z2_inv = pow(z2, -1, p)
    ax2 = (x2 * z2_inv) % p
    ay2 = (y2 * z2_inv) % p
    return ax1 == ax2 and ay1 == ay2


def main():
    random.seed(0x531)

    print("Scalar multiplication vs py_ecc multiply()...")
    n = 30
    scalars = [random.getrandbits(256) for _ in range(n)]
    base_points = [multiply(G1, random.getrandbits(256)) for _ in range(n)]
    ok = 0
    for i in range(n):
        x, y, z = to_jax_point(base_points[i])
        bits = msm.scalar_to_bits(scalars[i])
        rx, ry, rz = msm.point_scalar_mul(x, y, z, bits)
        ours = from_jax_point(rx, ry, rz)
        theirs = multiply(base_points[i], scalars[i])
        if points_match(ours, theirs):
            ok += 1
    print(f"  {ok}/{n}")

    print("Naive MSM vs sum of py_ecc multiplies...")
    for n in (1, 2, 5, 16):
        scalars = [random.getrandbits(256) for _ in range(n)]
        base_points = [multiply(G1, random.getrandbits(256)) for _ in range(n)]

        xs = jnp.stack([to_jax_point(p)[0] for p in base_points])
        ys = jnp.stack([to_jax_point(p)[1] for p in base_points])
        zs = jnp.stack([to_jax_point(p)[2] for p in base_points])
        bits_batch = jnp.stack([msm.scalar_to_bits(s) for s in scalars])

        rx, ry, rz = msm.msm_naive(xs, ys, zs, bits_batch)
        ours = from_jax_point(rx, ry, rz)

        expected = (1, 1, 0)
        for p, s in zip(base_points, scalars):
            expected = add(multiply(p, s), expected)
        match = points_match(ours, expected)
        print(f"  n={n}: {'PASS' if match else 'FAIL'}")

    print("Pippenger vs naive MSM (this module)...")
    for n in (0, 1, 2, 5, 16, 64):
        scalars = [random.getrandbits(256) for _ in range(n)]
        base_points = [multiply(G1, random.getrandbits(256)) for _ in range(n)]

        if n == 0:
            xs = jnp.zeros((0, fp.LIMBS), dtype=jnp.uint32)
            ys = jnp.zeros((0, fp.LIMBS), dtype=jnp.uint32)
            zs = jnp.zeros((0, fp.LIMBS), dtype=jnp.uint32)
            bits_batch = jnp.zeros((0, msm.SCALAR_BITS), dtype=jnp.bool_)
        else:
            xs = jnp.stack([to_jax_point(p)[0] for p in base_points])
            ys = jnp.stack([to_jax_point(p)[1] for p in base_points])
            zs = jnp.stack([to_jax_point(p)[2] for p in base_points])
            bits_batch = jnp.stack([msm.scalar_to_bits(s) for s in scalars])

        naive = msm.msm_naive(xs, ys, zs, bits_batch) if n > 0 else (
            fp.fp_zero(()).at[0].set(1), fp.fp_zero(()).at[0].set(1), fp.fp_zero(())
        )
        pip = msm.msm_pippenger(xs, ys, zs, bits_batch, window_bits=4) if n > 0 else naive

        # points_equal expects Montgomery-form inputs (like fp_mul), so
        # compare naive vs pippenger directly without converting to plain.
        nx, ny, nz = naive
        px, py, pz = pip
        matched = point.points_equal(nx[None], ny[None], nz[None], px[None], py[None], pz[None])[0]
        print(f"  n={n}: {'PASS' if bool(matched) else 'FAIL'}")


if __name__ == "__main__":
    main()
