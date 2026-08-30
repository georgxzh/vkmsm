#include "msm.hpp"

namespace vkmsm {
namespace {

constexpr int kScalarBits = 256;  // Scalar is 8 x 32-bit limbs

// Extracts window_bits bits of s starting at bit (window_index * window_bits),
// as an integer in [0, 2^window_bits). Bits beyond the scalar's width are
// treated as zero.
uint32_t scalar_window(const Scalar& s, int window_index, int window_bits) {
    int bit_start = window_index * window_bits;
    uint32_t result = 0;
    for (int i = 0; i < window_bits; i++) {
        int bit_pos = bit_start + i;
        if (bit_pos >= kScalarBits) break;
        int limb = bit_pos / 32;
        int offset = bit_pos % 32;
        uint32_t bit = (s[limb] >> offset) & 1u;
        result |= (bit << i);
    }
    return result;
}

}  // namespace

PointJacobian point_scalar_mul(const PointJacobian& p, const Scalar& s) {
    PointJacobian result = point_infinity();
    for (int limb = static_cast<int>(s.size()) - 1; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            result = point_double(result);
            if ((s[limb] >> bit) & 1u) {
                result = point_add(result, p);
            }
        }
    }
    return result;
}

PointJacobian msm_naive(const std::vector<PointJacobian>& points,
                         const std::vector<Scalar>& scalars) {
    PointJacobian acc = point_infinity();
    for (size_t i = 0; i < points.size(); i++) {
        acc = point_add(acc, point_scalar_mul(points[i], scalars[i]));
    }
    return acc;
}

PointJacobian msm_pippenger(const std::vector<PointJacobian>& points,
                             const std::vector<Scalar>& scalars, int window_bits) {
    const int num_windows = (kScalarBits + window_bits - 1) / window_bits;
    const uint32_t num_buckets = 1u << window_bits;  // indices 0..num_buckets-1; 0 unused

    PointJacobian result = point_infinity();

    for (int w = num_windows - 1; w >= 0; w--) {
        // Shift the accumulator up by one window's worth of bits (Horner's
        // method) before folding in this window's contribution. Skipped on
        // the very first (highest) window, where there's nothing yet to shift.
        if (w != num_windows - 1) {
            for (int i = 0; i < window_bits; i++) result = point_double(result);
        }

        std::vector<PointJacobian> buckets(num_buckets, point_infinity());
        for (size_t i = 0; i < points.size(); i++) {
            uint32_t digit = scalar_window(scalars[i], w, window_bits);
            if (digit != 0) {
                buckets[digit] = point_add(buckets[digit], points[i]);
            }
        }

        // Running-sum trick: sum_{k=1}^{num_buckets-1} k*buckets[k], computed
        // with only additions (no scalar multiplication needed). Processing
        // buckets from the highest index down, running_sum accumulates
        // buckets[k]+buckets[k+1]+...+buckets[num_buckets-1] and adding that
        // into window_total at every step effectively weights each bucket by
        // its index.
        PointJacobian running_sum = point_infinity();
        PointJacobian window_total = point_infinity();
        for (uint32_t k = num_buckets - 1; k >= 1; k--) {
            running_sum = point_add(running_sum, buckets[k]);
            window_total = point_add(window_total, running_sum);
        }

        result = point_add(result, window_total);
    }

    return result;
}

}  // namespace vkmsm
