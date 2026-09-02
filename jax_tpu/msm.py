"""BLS12-381 G1 multi-scalar multiplication in JAX, batched for TPU.

Mirrors ../src/msm.cpp: naive double-and-add (the correctness oracle and
slow baseline), then Pippenger's bucket method. Unlike the Vulkan port
(Stage 6), which had to hand-design a conflict-free bucket-assignment
scheme because GPU threads writing to the same bucket race, JAX's
`segment_sum` does exactly "sum values into buckets by index" as a single
vectorized primitive - XLA handles the parallel reduction itself. This is
the same problem Stage 6 solved by hand, solved instead by reaching for
the right built-in.
"""

import jax.numpy as jnp

import fp
import point

SCALAR_BITS = 256


def point_scalar_mul(x, y, z, scalar_bits):
    """x,y,z: (LIMBS,) Montgomery-form (a single point, not batched).
    scalar_bits: (SCALAR_BITS,) uint8/bool array, MSB first.
    Naive double-and-add - mirrors src/msm.cpp's point_scalar_mul."""
    rx, ry, rz = fp.fp_zero(()).at[0].set(1), fp.fp_zero(()).at[0].set(1), fp.fp_zero(())

    def body(i, carry):
        rx, ry, rz = carry
        rx, ry, rz = point.point_double(rx[None], ry[None], rz[None])
        rx, ry, rz = rx[0], ry[0], rz[0]
        bit = scalar_bits[i]
        ax, ay, az = point.point_add(rx[None], ry[None], rz[None], x[None], y[None], z[None])
        rx = jnp.where(bit, ax[0], rx)
        ry = jnp.where(bit, ay[0], ry)
        rz = jnp.where(bit, az[0], rz)
        return rx, ry, rz

    import jax

    rx, ry, rz = jax.lax.fori_loop(0, SCALAR_BITS, body, (rx, ry, rz))
    return rx, ry, rz


def scalar_to_bits(scalar_int):
    return jnp.array([(scalar_int >> (SCALAR_BITS - 1 - i)) & 1 for i in range(SCALAR_BITS)], dtype=jnp.bool_)


def msm_naive(xs, ys, zs, scalar_bits_batch):
    """xs,ys,zs: (N, LIMBS) Montgomery-form. scalar_bits_batch: (N, SCALAR_BITS).
    Returns the summed point via sequential accumulation of per-point
    scalar_mul results (the O(N) reduction itself, not the O(N*256)
    per-point work, is what stays sequential here)."""
    import jax

    scalar_mul_batched = jax.vmap(point_scalar_mul)
    txs, tys, tzs = scalar_mul_batched(xs, ys, zs, scalar_bits_batch)

    acc_x, acc_y, acc_z = fp.fp_zero(()).at[0].set(1), fp.fp_zero(()).at[0].set(1), fp.fp_zero(())
    n = xs.shape[0]
    for i in range(n):
        acc_x, acc_y, acc_z = point.point_add(
            acc_x[None], acc_y[None], acc_z[None], txs[i][None], tys[i][None], tzs[i][None]
        )
        acc_x, acc_y, acc_z = acc_x[0], acc_y[0], acc_z[0]
    return acc_x, acc_y, acc_z


def _scalar_window(scalar_bits_batch, window_index, window_bits):
    """scalar_bits_batch: (N, SCALAR_BITS) bool, MSB first. Returns (N,)
    uint32 digit for the given window (LSB-first window numbering, like
    src/msm.cpp's scalar_window)."""
    bit_start = window_index * window_bits
    n = scalar_bits_batch.shape[0]
    digit = jnp.zeros((n,), dtype=jnp.uint32)
    for i in range(window_bits):
        bit_pos = bit_start + i
        if bit_pos >= SCALAR_BITS:
            break
        bit = scalar_bits_batch[:, SCALAR_BITS - 1 - bit_pos].astype(jnp.uint32)
        digit = digit | (bit << i)
    return digit


def msm_pippenger(xs, ys, zs, scalar_bits_batch, window_bits=8):
    """xs,ys,zs: (N, LIMBS) Montgomery-form. scalar_bits_batch: (N, SCALAR_BITS).

    Bucket accumulation here is a straightforward `jax.lax.fori_loop`
    scatter (one point_add + `.at[k].set()` per input point). This is
    correct and simple, but - worth being honest about, in the same spirit
    as the Vulkan project's Stage 6/7 findings - it does NOT parallelize
    across points the way Stage 6's one-thread-per-bucket Vulkan shader
    did: `jax.ops.segment_sum` only handles plain numeric summation, not a
    custom group operation like EC point addition, so there's no built-in
    "segmented reduce with a custom op" to reach for here, and a
    dynamic-index scatter loop like this one typically compiles to a
    sequential loop rather than something XLA parallelizes across TPU
    lanes. A genuinely TPU-parallel bucket scheme (e.g. sorting points by
    digit, then a parallel segmented tree-reduction with point_add as the
    combiner) is real additional work, left as a further exercise -
    exactly the kind of gap Stage 8 explored on the Vulkan side.
    """
    import jax

    n = xs.shape[0]
    num_windows = (SCALAR_BITS + window_bits - 1) // window_bits
    num_buckets = 1 << window_bits

    result_x, result_y, result_z = (
        fp.fp_zero(()).at[0].set(1),
        fp.fp_zero(()).at[0].set(1),
        fp.fp_zero(()),
    )

    for w in range(num_windows - 1, -1, -1):
        if w != num_windows - 1:
            for _ in range(window_bits):
                result_x, result_y, result_z = point.point_double(
                    result_x[None], result_y[None], result_z[None]
                )
                result_x, result_y, result_z = result_x[0], result_y[0], result_z[0]

        digits = _scalar_window(scalar_bits_batch, w, window_bits)

        bucket_x = jnp.tile(fp.fp_zero(()).at[0].set(1), (num_buckets, 1))
        bucket_y = jnp.tile(fp.fp_zero(()).at[0].set(1), (num_buckets, 1))
        bucket_z = jnp.zeros((num_buckets, fp.LIMBS), dtype=jnp.uint32)

        # n is a static Python int (array shapes are always static in
        # JAX), so a plain unrolled Python loop is simpler here than
        # jax.lax.fori_loop - and avoids it: fori_loop's dynamic-index
        # .at[k].set() scatter, with k itself a value depending on
        # scatter_body's own loop-carried state, hit a MemoryError in
        # eager (non-jit) execution - not investigated further since the
        # static-loop form sidesteps it entirely and is the more natural
        # fit for a genuinely static n anyway.
        for i in range(n):
            k = digits[i]
            bxk, byk, bzk = bucket_x[k], bucket_y[k], bucket_z[k]
            nx, ny, nz = point.point_add(
                bxk[None], byk[None], bzk[None], xs[i][None], ys[i][None], zs[i][None]
            )
            bucket_x = bucket_x.at[k].set(nx[0])
            bucket_y = bucket_y.at[k].set(ny[0])
            bucket_z = bucket_z.at[k].set(nz[0])

        # Running-sum trick, same as src/msm.cpp: sum_{k=1}^{B-1} k*bucket[k]
        # with only additions.
        running_x, running_y, running_z = (
            fp.fp_zero(()).at[0].set(1),
            fp.fp_zero(()).at[0].set(1),
            fp.fp_zero(()),
        )
        window_x, window_y, window_z = (
            fp.fp_zero(()).at[0].set(1),
            fp.fp_zero(()).at[0].set(1),
            fp.fp_zero(()),
        )
        for k in range(num_buckets - 1, 0, -1):
            running_x, running_y, running_z = point.point_add(
                running_x[None], running_y[None], running_z[None],
                bucket_x[k][None], bucket_y[k][None], bucket_z[k][None],
            )
            running_x, running_y, running_z = running_x[0], running_y[0], running_z[0]
            window_x, window_y, window_z = point.point_add(
                window_x[None], window_y[None], window_z[None],
                running_x[None], running_y[None], running_z[None],
            )
            window_x, window_y, window_z = window_x[0], window_y[0], window_z[0]

        result_x, result_y, result_z = point.point_add(
            result_x[None], result_y[None], result_z[None],
            window_x[None], window_y[None], window_z[None],
        )
        result_x, result_y, result_z = result_x[0], result_y[0], result_z[0]

    return result_x, result_y, result_z
