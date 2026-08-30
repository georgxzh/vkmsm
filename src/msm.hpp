// BLS12-381 G1 multi-scalar multiplication (MSM), CPU reference
// implementations.

#pragma once

#include <array>
#include <vector>

#include "point.hpp"

namespace vkmsm {

// A 256-bit scalar, plain little-endian 32-bit limbs (limb[0] least
// significant). This is a bare integer for scalar multiplication, not a
// modular Fr field element - the curve arithmetic below is correct for any
// magnitude, not just values already reduced mod the group order.
using Scalar = std::array<uint32_t, 8>;

// scalar * p, via naive left-to-right (MSB -> LSB) double-and-add.
PointJacobian point_scalar_mul(const PointJacobian& p, const Scalar& s);

// sum_i (scalars[i] * points[i]), via per-point double-and-add.
// This is both the CPU correctness oracle for Pippenger's method and the
// slow baseline for the eventual GPU-vs-CPU benchmark.
PointJacobian msm_naive(const std::vector<PointJacobian>& points,
                         const std::vector<Scalar>& scalars);

// Pippenger's bucket method: processes scalars in window_bits-wide windows
// ("digits"), so each window needs 2^window_bits buckets - larger windows
// mean fewer passes over the point list but more bucket-accumulation work
// per window. window_bits=8 is a reasonable default for a correctness
// baseline; Stage 8 revisits this as an actual performance tuning knob.
// Must match msm_naive() exactly - see stage3b_pippenger_test.cpp.
PointJacobian msm_pippenger(const std::vector<PointJacobian>& points,
                             const std::vector<Scalar>& scalars, int window_bits = 8);

}  // namespace vkmsm
