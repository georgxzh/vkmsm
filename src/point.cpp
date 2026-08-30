#include "point.hpp"

namespace vkmsm {
namespace {

Fp fp_double(const Fp& x) { return fp_add(x, x); }

// The BLS12-381 G1 curve constant b = 4, in Montgomery form.
const Fp& curve_b() {
    static const Fp b = [] {
        std::array<uint32_t, kLimbs> four{};
        four[0] = 4;
        return fp_from_plain_limbs(four);
    }();
    return b;
}

}  // namespace

PointJacobian point_infinity() {
    PointJacobian p;
    p.x = fp_one();
    p.y = fp_one();
    p.z = fp_zero();
    return p;
}

bool point_is_infinity(const PointJacobian& p) { return fp_is_zero(p.z); }

PointJacobian point_double(const PointJacobian& p) {
    if (point_is_infinity(p)) return p;

    // dbl-2009-l (EFD), specialized for a=0:
    //   A = X1^2, B = Y1^2, C = B^2
    //   D = 2*((X1+B)^2 - A - C)
    //   E = 3*A, F = E^2
    //   X3 = F - 2*D
    //   Y3 = E*(D - X3) - 8*C
    //   Z3 = 2*Y1*Z1
    Fp A = fp_mul(p.x, p.x);
    Fp B = fp_mul(p.y, p.y);
    Fp C = fp_mul(B, B);
    Fp X1plusB = fp_add(p.x, B);
    Fp D = fp_double(fp_sub(fp_sub(fp_mul(X1plusB, X1plusB), A), C));
    Fp E = fp_add(fp_double(A), A);
    Fp F = fp_mul(E, E);

    PointJacobian r;
    r.x = fp_sub(F, fp_double(D));
    Fp eightC = fp_double(fp_double(fp_double(C)));
    r.y = fp_sub(fp_mul(E, fp_sub(D, r.x)), eightC);
    r.z = fp_double(fp_mul(p.y, p.z));
    return r;
}

PointJacobian point_add(const PointJacobian& p1, const PointJacobian& p2) {
    if (point_is_infinity(p1)) return p2;
    if (point_is_infinity(p2)) return p1;

    Fp Z1Z1 = fp_mul(p1.z, p1.z);
    Fp Z2Z2 = fp_mul(p2.z, p2.z);
    Fp U1 = fp_mul(p1.x, Z2Z2);
    Fp U2 = fp_mul(p2.x, Z1Z1);
    Fp Z1cubed = fp_mul(Z1Z1, p1.z);
    Fp Z2cubed = fp_mul(Z2Z2, p2.z);
    Fp S1 = fp_mul(p1.y, Z2cubed);
    Fp S2 = fp_mul(p2.y, Z1cubed);

    if (fp_equal(U1, U2)) {
        // Same X: either the same point (-> double) or P == -Q (-> infinity).
        return fp_equal(S1, S2) ? point_double(p1) : point_infinity();
    }

    // add-2007-bl (EFD):
    //   H = U2-U1, I = (2H)^2, J = H*I
    //   r = 2*(S2-S1), V = U1*I
    //   X3 = r^2 - J - 2*V
    //   Y3 = r*(V-X3) - 2*S1*J
    //   Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    Fp H = fp_sub(U2, U1);
    Fp twoH = fp_double(H);
    Fp I = fp_mul(twoH, twoH);
    Fp J = fp_mul(H, I);
    Fp rr = fp_double(fp_sub(S2, S1));
    Fp V = fp_mul(U1, I);

    PointJacobian result;
    result.x = fp_sub(fp_sub(fp_mul(rr, rr), J), fp_double(V));
    result.y = fp_sub(fp_mul(rr, fp_sub(V, result.x)), fp_double(fp_mul(S1, J)));
    Fp Z1plusZ2 = fp_add(p1.z, p2.z);
    result.z = fp_mul(fp_sub(fp_sub(fp_mul(Z1plusZ2, Z1plusZ2), Z1Z1), Z2Z2), H);
    return result;
}

bool point_on_curve(const PointJacobian& p) {
    if (point_is_infinity(p)) return true;
    Fp y2 = fp_mul(p.y, p.y);
    Fp x3 = fp_mul(fp_mul(p.x, p.x), p.x);
    Fp z2 = fp_mul(p.z, p.z);
    Fp z6 = fp_mul(fp_mul(z2, z2), z2);
    Fp rhs = fp_add(x3, fp_mul(curve_b(), z6));
    return fp_equal(y2, rhs);
}

bool point_equal(const PointJacobian& a, const PointJacobian& b) {
    bool a_inf = point_is_infinity(a);
    bool b_inf = point_is_infinity(b);
    if (a_inf || b_inf) return a_inf && b_inf;

    Fp Z1Z1 = fp_mul(a.z, a.z);
    Fp Z2Z2 = fp_mul(b.z, b.z);
    if (!fp_equal(fp_mul(a.x, Z2Z2), fp_mul(b.x, Z1Z1))) return false;

    Fp Z1cubed = fp_mul(Z1Z1, a.z);
    Fp Z2cubed = fp_mul(Z2Z2, b.z);
    return fp_equal(fp_mul(a.y, Z2cubed), fp_mul(b.y, Z1cubed));
}

}  // namespace vkmsm
