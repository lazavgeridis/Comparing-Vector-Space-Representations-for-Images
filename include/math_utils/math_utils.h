#ifndef MATH_UTILS_H
#define MATH_UTILS_H


#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>

#include "../../include/metric/metric.h"


inline int fast_mod(const int input, const int ceil) {
    return input >= ceil ? input % ceil : input;
}


inline uint64_t exp_modulo(uint32_t base, uint16_t exp, uint32_t mod) {
    uint64_t res = 1;

    while (exp > 0) {
        if (exp & 1) { res = fast_mod(res * base, mod); }
        exp >>= 1;
        base = fast_mod(base * base, mod);
    }
    return res;
}

inline int modulo(int i, int n) {
    return ((i % n) + n) % n;
}



#endif // MATH_UTILS_H