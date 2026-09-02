"""Validates jax_tpu/point.py against py_ecc - an independent BLS12-381
implementation (not blst; extra triangulation) - mirroring how the C++
project validated src/point.cpp against blst."""

import random

import jax.numpy as jnp
import numpy as np
from py_ecc.optimized_bls12_381 import G1, add, field_modulus, is_on_curve, multiply
from py_ecc.optimized_bls12_381 import b as curve_b

import fp
import point

assert field_modulus == fp.P, "py_ecc's field modulus disagrees with our derived P"
assert is_on_curve(G1, curve_b)


def to_jax_point(p):
    """Converts a py_ecc point to Montgomery-form Jacobian limbs.

    py_ecc's optimized_bls12_381 uses standard homogeneous projective
    coordinates (x=X/Z, y=Y/Z), not Jacobian (x=X/Z^2, y=Y/Z^3) like ours
    - so its raw (X,Y,Z) can't be fed to our functions directly (that
    would silently treat it as a different point whenever Z != 1, which
    multiply() results generally have). Bridging through affine first
    (using py_ecc's own convention to get there) is the safe, convention-
    independent way across; z=1 in the affine-derived Jacobian point is
    always a valid Jacobian representation of that same affine point.
    """
    x, y, z = (int(c) % fp.P for c in p)
    if z == 0:
        ax, ay, az = 0, 0, 0  # infinity, any x/y placeholder is fine
    else:
        zi = pow(z, -1, fp.P)
        ax, ay, az = (x * zi) % fp.P, (y * zi) % fp.P, 1
    plain = jnp.stack([
        jnp.asarray(fp.int_to_limbs(ax)),
        jnp.asarray(fp.int_to_limbs(ay)),
        jnp.asarray(fp.int_to_limbs(az)),
    ])
    mont = fp.to_montgomery(plain)
    return mont[0], mont[1], mont[2]


def batch(points):
    xs = jnp.stack([to_jax_point(p)[0] for p in points])
    ys = jnp.stack([to_jax_point(p)[1] for p in points])
    zs = jnp.stack([to_jax_point(p)[2] for p in points])
    return xs, ys, zs


def from_jax_point(x, y, z):
    plain = fp.from_montgomery(jnp.stack([x, y, z]))
    return (fp.limbs_to_int(plain[0]), fp.limbs_to_int(plain[1]), fp.limbs_to_int(plain[2]))


def points_match(ours, theirs_pyecc):
    """Compare via affine coordinates - representation-independent, since
    ours is Jacobian (x=X/Z^2, y=Y/Z^3) but py_ecc's optimized_bls12_381
    uses standard homogeneous projective (x=X/Z, y=Y/Z). Converting each
    to affine with its own convention makes them directly comparable."""
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
    random.seed(0x901471)
    n = 500
    print(f"Generating {n} random G1 points via py_ecc (multiply(G1, random scalar))...")
    scalars = [random.getrandbits(256) for _ in range(n)]
    pts_a = [multiply(G1, s) for s in scalars]
    scalars_b = [random.getrandbits(256) for _ in range(n)]
    pts_b = [multiply(G1, s) for s in scalars_b]

    xa, ya, za = batch(pts_a)
    xb, yb, zb = batch(pts_b)

    print("Batched doubling vs py_ecc add(P, P)...")
    dx, dy, dz = point.point_double(xa, ya, za)
    dbl_ok = 0
    for i in range(n):
        ours = from_jax_point(dx[i], dy[i], dz[i])
        theirs = add(pts_a[i], pts_a[i])
        if points_match(ours, theirs):
            dbl_ok += 1
    print(f"  {dbl_ok}/{n}")

    print("Batched addition vs py_ecc add(P, Q)...")
    ax, ay, az = point.point_add(xa, ya, za, xb, yb, zb)
    add_ok = 0
    for i in range(n):
        ours = from_jax_point(ax[i], ay[i], az[i])
        theirs = add(pts_a[i], pts_b[i])
        if points_match(ours, theirs):
            add_ok += 1
    print(f"  {add_ok}/{n}")

    print("P + (-P) == infinity...")
    neg_ya_batch = fp.fp_sub(jnp.zeros_like(ya), ya)
    ix, iy, iz = point.point_add(xa, ya, za, xa, neg_ya_batch, za)
    inf_ok = sum(1 for i in range(n) if point.is_infinity(iz)[i].item())
    print(f"  {inf_ok}/{n}")

    total = n + n + n
    passed = dbl_ok + add_ok + inf_ok
    print(f"\n{'=' * 40}")
    print(f"Total: {passed}/{total} checks passed")
    print("PASS" if passed == total else "FAIL")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
