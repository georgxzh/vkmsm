// BLS12-381 G1 point arithmetic, CPU reference implementation.
//
// Points are represented in Jacobian coordinates (X, Y, Z), where the
// corresponding affine point is (X/Z^2, Y/Z^3). This avoids a field
// inversion on every addition/doubling (the affine formulas need one per
// operation; Jacobian formulas need none until you actually want the
// affine result back out). Z == 0 represents the point at infinity.
//
// Curve equation: y^2 = x^3 + 4  (a = 0, b = 4)

#pragma once

#include "fp.hpp"

namespace vkmsm {

struct PointJacobian {
    Fp x, y, z;
};

PointJacobian point_infinity();
bool point_is_infinity(const PointJacobian& p);

// EFD "dbl-2009-l" (a=0 specialization).
PointJacobian point_double(const PointJacobian& p);

// EFD "add-2007-bl", with explicit handling of P==Q (delegates to
// point_double) and P==-Q (returns infinity).
PointJacobian point_add(const PointJacobian& a, const PointJacobian& b);

// y^2 == x^3 + b*z^6, the homogeneous (division-free) curve check.
bool point_on_curve(const PointJacobian& p);

// Affine-equivalence check (X1*Z2^2 == X2*Z1^2 and Y1*Z2^3 == Y2*Z1^3),
// so two Jacobian representations of the same affine point compare equal
// even with different Z.
bool point_equal(const PointJacobian& a, const PointJacobian& b);

}  // namespace vkmsm
