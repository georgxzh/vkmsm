"""Validates jax_tpu/fp.py against Python's built-in arbitrary-precision
integers - an independent, trivially-correct oracle (mirrors how the C++
project validated src/fp.cpp against blst)."""

import random

import jax.numpy as jnp
import numpy as np

import fp


def random_field_element():
    while True:
        x = random.getrandbits(384)
        if x < fp.P:
            return x


def main():
    random.seed(0xF9)
    n = 2000
    print(f"Generating {n} random field element pairs...")
    a_ints = [random_field_element() for _ in range(n)]
    b_ints = [random_field_element() for _ in range(n)]

    a_plain = jnp.asarray(np.stack([fp.int_to_limbs(x) for x in a_ints]))
    b_plain = jnp.asarray(np.stack([fp.int_to_limbs(x) for x in b_ints]))

    print("Round-trip (plain -> Montgomery -> plain)...")
    a_mont = fp.to_montgomery(a_plain)
    back = fp.from_montgomery(a_mont)
    back_ints = [fp.limbs_to_int(back[i]) for i in range(n)]
    assert back_ints == a_ints, "round-trip failed"
    print(f"  OK: {n}/{n}")

    b_mont = fp.to_montgomery(b_plain)

    print("Batched add/sub/mul vs Python bignum reference...")
    sum_mont = fp.fp_add(a_mont, b_mont)
    diff_mont = fp.fp_sub(a_mont, b_mont)
    prod_mont = fp.fp_mul(a_mont, b_mont)

    sum_plain = fp.from_montgomery(sum_mont)
    diff_plain = fp.from_montgomery(diff_mont)
    prod_plain = fp.from_montgomery(prod_mont)

    add_ok = sub_ok = mul_ok = 0
    for i in range(n):
        expected_sum = (a_ints[i] + b_ints[i]) % fp.P
        expected_diff = (a_ints[i] - b_ints[i]) % fp.P
        expected_prod = (a_ints[i] * b_ints[i]) % fp.P

        if fp.limbs_to_int(sum_plain[i]) == expected_sum:
            add_ok += 1
        if fp.limbs_to_int(diff_plain[i]) == expected_diff:
            sub_ok += 1
        if fp.limbs_to_int(prod_plain[i]) == expected_prod:
            mul_ok += 1

    print(f"  add: {add_ok}/{n}")
    print(f"  sub: {sub_ok}/{n}")
    print(f"  mul: {mul_ok}/{n}")

    print("Edge cases (0, 1, p-1)...")
    edge_ints = [0, 1, fp.P - 1]
    edge_plain = jnp.asarray(np.stack([fp.int_to_limbs(x) for x in edge_ints]))
    edge_mont = fp.to_montgomery(edge_plain)
    one_mont = jnp.broadcast_to(fp.fp_one(), edge_mont.shape)
    plus_one = fp.from_montgomery(fp.fp_add(edge_mont, one_mont))
    edge_ok = 0
    for i, x in enumerate(edge_ints):
        if fp.limbs_to_int(plus_one[i]) == (x + 1) % fp.P:
            edge_ok += 1
    print(f"  OK: {edge_ok}/{len(edge_ints)}")

    total = n + n + n + len(edge_ints) + n
    passed = add_ok + sub_ok + mul_ok + edge_ok + n
    print(f"\n{'=' * 40}")
    print(f"Total: {passed}/{total} checks passed")
    print("PASS" if passed == total else "FAIL")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
