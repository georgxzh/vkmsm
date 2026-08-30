// Shared G1 Jacobian point arithmetic for GLSL compute shaders, mirroring
// src/point.cpp exactly (dbl-2009-l doubling, add-2007-bl addition, a=0
// specialization). Include fp_arith.glsl first - this file uses LIMBS,
// fp_add/fp_sub/fp_mul/fp_double/fp_eq/fp_is_zero from it.

struct Point {
    uint x[LIMBS];
    uint y[LIMBS];
    uint z[LIMBS];
};

bool point_is_infinity(Point p) { return fp_is_zero(p.z); }

Point point_double(Point p) {
    if (point_is_infinity(p)) return p;

    uint A[LIMBS] = fp_mul(p.x, p.x);
    uint B[LIMBS] = fp_mul(p.y, p.y);
    uint C[LIMBS] = fp_mul(B, B);
    uint X1plusB[LIMBS] = fp_add(p.x, B);
    uint D[LIMBS] = fp_double(fp_sub(fp_sub(fp_mul(X1plusB, X1plusB), A), C));
    uint E[LIMBS] = fp_add(fp_double(A), A);
    uint F[LIMBS] = fp_mul(E, E);

    Point r;
    r.x = fp_sub(F, fp_double(D));
    uint eightC[LIMBS] = fp_double(fp_double(fp_double(C)));
    r.y = fp_sub(fp_mul(E, fp_sub(D, r.x)), eightC);
    r.z = fp_double(fp_mul(p.y, p.z));
    return r;
}

Point point_add(Point p1, Point p2) {
    if (point_is_infinity(p1)) return p2;
    if (point_is_infinity(p2)) return p1;

    uint Z1Z1[LIMBS] = fp_mul(p1.z, p1.z);
    uint Z2Z2[LIMBS] = fp_mul(p2.z, p2.z);
    uint U1[LIMBS] = fp_mul(p1.x, Z2Z2);
    uint U2[LIMBS] = fp_mul(p2.x, Z1Z1);
    uint Z1cubed[LIMBS] = fp_mul(Z1Z1, p1.z);
    uint Z2cubed[LIMBS] = fp_mul(Z2Z2, p2.z);
    uint S1[LIMBS] = fp_mul(p1.y, Z2cubed);
    uint S2[LIMBS] = fp_mul(p2.y, Z1cubed);

    if (fp_eq(U1, U2)) {
        if (fp_eq(S1, S2)) {
            return point_double(p1);
        } else {
            // P == -Q: point at infinity. Only z == 0 is load-bearing.
            Point inf;
            for (int i = 0; i < LIMBS; i++) {
                inf.x[i] = 0u;
                inf.y[i] = 0u;
                inf.z[i] = 0u;
            }
            return inf;
        }
    }

    uint H[LIMBS] = fp_sub(U2, U1);
    uint twoH[LIMBS] = fp_double(H);
    uint I[LIMBS] = fp_mul(twoH, twoH);
    uint J[LIMBS] = fp_mul(H, I);
    uint rr[LIMBS] = fp_double(fp_sub(S2, S1));
    uint V[LIMBS] = fp_mul(U1, I);

    Point result;
    result.x = fp_sub(fp_sub(fp_mul(rr, rr), J), fp_double(V));
    result.y = fp_sub(fp_mul(rr, fp_sub(V, result.x)), fp_double(fp_mul(S1, J)));
    uint Z1plusZ2[LIMBS] = fp_add(p1.z, p2.z);
    result.z = fp_mul(fp_sub(fp_sub(fp_mul(Z1plusZ2, Z1plusZ2), Z1Z1), Z2Z2), H);
    return result;
}
