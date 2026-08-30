// Shared F_p arithmetic helpers for GLSL compute shaders. Included (not
// compiled standalone) by shaders that declare their own push_constant
// block named `pc` with at least `uint n0;` and `uint modulus[LIMBS];`
// fields - see shaders/fp_ops.comp for the field-arithmetic test shader
// and shaders/point_ops.comp for the point-arithmetic shader that builds
// on these.
//
// Mirrors src/fp.cpp exactly; see that file for the algorithm derivation.

const int LIMBS = 12;

uint adc(inout uint carry, uint a, uint b) {
    uint c1, c2;
    uint s1 = uaddCarry(a, b, c1);
    uint s2 = uaddCarry(s1, carry, c2);
    carry = c1 + c2;
    return s2;
}

uint mac(inout uint carry, uint a, uint b, uint addend) {
    uint hi, lo;
    umulExtended(a, b, hi, lo);
    uint c1, c2;
    uint s1 = uaddCarry(lo, addend, c1);
    uint s2 = uaddCarry(s1, carry, c2);
    carry = hi + c1 + c2;
    return s2;
}

bool geq(uint x[LIMBS], uint y[LIMBS]) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (x[i] != y[i]) return x[i] > y[i];
    }
    return true;
}

bool fp_eq(uint x[LIMBS], uint y[LIMBS]) {
    for (int i = 0; i < LIMBS; i++) {
        if (x[i] != y[i]) return false;
    }
    return true;
}

bool fp_is_zero(uint x[LIMBS]) {
    for (int i = 0; i < LIMBS; i++) {
        if (x[i] != 0u) return false;
    }
    return true;
}

uint[LIMBS] raw_add(uint x[LIMBS], uint y[LIMBS], out uint carry_out) {
    uint r[LIMBS];
    uint carry = 0u;
    for (int i = 0; i < LIMBS; i++) {
        r[i] = adc(carry, x[i], y[i]);
    }
    carry_out = carry;
    return r;
}

uint[LIMBS] raw_sub(uint x[LIMBS], uint y[LIMBS], out uint borrow_out) {
    uint r[LIMBS];
    uint borrow = 0u;
    for (int i = 0; i < LIMBS; i++) {
        uint b1, b2;
        uint d1 = usubBorrow(x[i], y[i], b1);
        uint d2 = usubBorrow(d1, borrow, b2);
        r[i] = d2;
        borrow = b1 + b2;
    }
    borrow_out = borrow;
    return r;
}

uint[LIMBS] fp_add(uint a[LIMBS], uint b[LIMBS]) {
    uint carry;
    uint r[LIMBS] = raw_add(a, b, carry);
    if (geq(r, pc.modulus)) {
        uint borrow;
        r = raw_sub(r, pc.modulus, borrow);
    }
    return r;
}

uint[LIMBS] fp_sub(uint a[LIMBS], uint b[LIMBS]) {
    uint borrow;
    uint r[LIMBS] = raw_sub(a, b, borrow);
    if (borrow != 0u) {
        uint carry;
        r = raw_add(r, pc.modulus, carry);
    }
    return r;
}

uint[LIMBS] fp_double(uint a[LIMBS]) { return fp_add(a, a); }

// CIOS Montgomery multiplication - see src/fp.cpp for the derivation.
uint[LIMBS] fp_mul(uint a[LIMBS], uint b[LIMBS]) {
    uint t[LIMBS + 2];
    for (int i = 0; i < LIMBS + 2; i++) t[i] = 0u;

    for (int i = 0; i < LIMBS; i++) {
        uint carry = 0u;
        for (int j = 0; j < LIMBS; j++) {
            t[j] = mac(carry, a[j], b[i], t[j]);
        }
        uint carry2 = 0u;
        t[LIMBS] = adc(carry2, t[LIMBS], carry);
        t[LIMBS + 1] += carry2;

        uint m = t[0] * pc.n0;

        uint carry3 = 0u;
        mac(carry3, m, pc.modulus[0], t[0]);  // low word discarded: == 0 by construction
        for (int j = 1; j < LIMBS; j++) {
            t[j - 1] = mac(carry3, m, pc.modulus[j], t[j]);
        }
        uint carry4 = 0u;
        t[LIMBS - 1] = adc(carry4, t[LIMBS], carry3);
        t[LIMBS] = t[LIMBS + 1] + carry4;
        t[LIMBS + 1] = 0u;
    }

    uint result[LIMBS];
    for (int i = 0; i < LIMBS; i++) result[i] = t[i];

    while (geq(result, pc.modulus)) {
        uint borrow;
        result = raw_sub(result, pc.modulus, borrow);
    }
    return result;
}
