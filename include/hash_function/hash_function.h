#ifndef HASH_FUNCTION_H
#define HASH_FUNCTION_H


#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

#include "../math_utils/math_utils.h"


template <typename T>
class HashFunction {

    private:
        /* k: number of h functions that we'll create */
        const uint16_t k;
        const uint16_t d;
        const uint32_t m;
        const uint32_t M;
        const double w;
        std::vector<double> s_transformations;
        std::vector<double> a;

    public:

        HashFunction(   const uint16_t k, const uint16_t d, const uint32_t m, const uint32_t M, \
                        const double w) : k(k), d(d), m(m), M(M), w(w), s_transformations(d), a(d) {

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::uniform_real_distribution<double> distribution(0.0, w);
            
            for (size_t i = 0; i != d; ++i) {
                s_transformations[i] = distribution(generator);
            }
        
        } 
        
        ~HashFunction() = default;


        uint32_t hash_function_construction(const std::vector<T> &pixels) {
            uint32_t hash_value = 0;


            for (size_t i = 0; i != d; ++i) {
                a[i] = floor((pixels[i] - s_transformations[i]) / w);
            }

            std::reverse(a.begin(), a.end());

            for (size_t i = 0; i != d; ++i) {
                hash_value += (modulo(a[i], M) * exp_modulo(m, i, M)) % M;
            }

            return fast_mod(hash_value, M);
        }

};


template<typename T>
class AmplifiedHashFunction {

    private:

        const uint16_t k;
        const uint16_t d;
        const uint32_t m;
        const uint32_t M;
        const double w;

        std::vector<HashFunction<T>> h_i;

    public:

        AmplifiedHashFunction(  const uint16_t k, const uint16_t d, const uint32_t m, \
                                const uint32_t M, const double w): k(k), d(d), m(m), M(M), w(w) {
        
            for (size_t i = 0; i != k; ++i) {
                h_i.push_back(HashFunction<T> (k, d, m, M, w));
            }  
        }


        ~AmplifiedHashFunction() = default;


        size_t amplified_function_construction(const std::vector<T> &pixels) {
            
            std::string res = "";
            char *p_end = nullptr;
            
            for (size_t i = 0; i != k; ++i) {
                res += std::to_string(h_i[i].hash_function_construction(pixels));
            }

            return std::strtoull(res.c_str(), &p_end, 10);
        }

};


#endif // HASH_FUNCTION_H
