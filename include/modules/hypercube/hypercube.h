#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <unordered_map>
#include <vector>
#include <utility>   /* for make_pair() */
#include <limits>    /* for numeric_limits() */
#include <algorithm> /* for sort() */
#include <cassert>
#include <cstdlib>
#include <cmath>

#include "../../hash_function/hash_function.h"
#include "../../metric/metric.h"

#define MULTIPLE1 4
#define MULTIPLE2 10
#define C         1.1


template <typename T>
class Hypercube {

    private:

        uint32_t m;
        uint32_t M;

        /* hypercube method parameters */
        const uint32_t projection_dimension;
        const uint16_t max_candidates;
        const uint16_t max_probes;

        /* search methods parameters */
        const uint16_t N;
        const float    R;

        /* data parameters */
        const size_t   train_samples;
        const uint32_t D;
        const double   win;

        /* maintain a hash table for each f_i, where 1 <= i <= d'; each hash table stores pairs {h_i(x) : 0 or 1} */
        std::vector<std::unordered_multimap<uint32_t, bool>> uniform_binary_mapppings;

        /* hash table representing the hypercube; each vertex is basically a bucket */
        std::unordered_map<std::string, std::vector<size_t>> hash_table;

        /* for the hypercube method we use d' x LSH hash functions */
        std::vector<HashFunction<T>> hash_functions;


        void init_k_nearest_neighbors(std::vector<std::pair<uint32_t, size_t>> &k_nearest)
        {
            uint32_t best_dist = std::numeric_limits<uint32_t>::max();

            for (size_t i = 0; i != max_candidates; ++i) {
                k_nearest.emplace_back(best_dist, 0);
            }
        }


    public:

        Hypercube (uint32_t projdim, uint16_t cands, uint16_t probes, uint16_t nns, float r, \
                    size_t trn, uint32_t d, double meandist, const std::vector<std::vector<T>> &samples) \
                    : projection_dimension(projdim), max_candidates(cands), max_probes(probes), \
                      N(nns), R(r), train_samples(trn), D(d), win(MULTIPLE2 * meandist)
        {
            std::cout << "Window is: ";
            std::cout << win << std::endl;

            M = 1ULL << (32 / projection_dimension);
            m = (1ULL << 32) - (5);

            std::cout << "M is " << M << std::endl;
            std::cout << "m is " << m << std::endl;

            /* we do not use amplified hashes on this method, so we do not care about k (k = 0) */
            for (size_t i = 0; i != projection_dimension; ++i) {
                hash_functions.emplace_back( HashFunction<T> (0, D, m, M, win) );
                uniform_binary_mapppings.emplace_back(std::unordered_multimap<uint32_t, bool>());
            }

            hash_table.rehash(trn); // in order to keep hash table's bucket count close to the total number of points
            for (size_t i = 0; i != train_samples; ++i) {
                cube_projection_train(samples[i], i);
            }
        }


        void cube_projection_train(const std::vector<T> &point, size_t index)
        {
            uint32_t    hval;
            short       bit;
            std::string bitstring;

            for (size_t j = 0; j != projection_dimension; ++j) {
                
                hval = hash_functions[j].hash_function_construction(point);

                /* check if f_j(h_j(p)) has already been computed.
                 * positive: retrieve from data store and use that value 
                 * negative: compute f_j(h_j(x)) i.e toss a coin
                 */
                bit = retrieve_val(uniform_binary_mapppings[j], hval);

                /* this is the first time we encounter h_j, so we calculate and store f_j(h_j) */
                if (bit == -1) {
                    bit = uniform_binmap(uniform_binary_mapppings[j], hval);
                }

                bitstring += std::to_string(bit);
            }
            
            assert(bitstring.size() == projection_dimension);

            hash_table[bitstring].push_back(index);
        }


        std::string cube_projection_test(const std::vector<T> &query)
        {
            uint32_t    hval;
            short       bit;
            std::string bitstring;

            for (size_t j = 0; j != projection_dimension; ++j) {
                
                hval = hash_functions[j].hash_function_construction(query);

                /* check if f_j(h_j(p)) has already been computed.
                 * positive: retrieve value from data store and use it 
                 * negative: compute f_j(h_j(x)) i.e toss a coin
                 */
                bit = retrieve_val(uniform_binary_mapppings[j], hval);

                /* this is the first time we encounter h_j, so we calculate and store f_j(h_j) */
                if (bit == -1) {
                    bit = uniform_binmap(uniform_binary_mapppings[j], hval);
                }

                bitstring += std::to_string(bit);
            }

            assert(bitstring.size() == projection_dimension);

            return bitstring;
        }


        short retrieve_val(const std::unordered_multimap<uint32_t, bool> &map, uint32_t hval)
        {
            if ( !map.empty() ) {
                auto retrieved = map.find(hval);
                    if ( retrieved != map.end() ) 
                        return retrieved->second;
            }
            
            return -1;
        }


        short uniform_binmap(std::unordered_multimap<uint32_t, bool> &map, uint32_t hval)
        {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist(0,1); // uniform distribution in range [0, 1]
            short bit = dist(rng);  // generate 0 or 1
            map.insert( std::make_pair(hval, bit) ); // insert the pair {hval : f(hval)} into the hash table

            return bit;
        }


        std::vector<std::pair<uint32_t, size_t>> approximate_nn(const std::vector<T> &query, const std::vector<std::vector<T>> &train_samples)
        {
            /* vector to store query's nearest neighbors - M candidates */
            std::vector<std::pair<uint32_t, size_t>> candidates;
            uint32_t dist = 0;
            uint32_t cnt = projection_dimension;
            uint16_t M = max_candidates;
            uint16_t probes = max_probes;
            uint8_t  bits = 1;
            size_t   bucket_count = 0;

            init_k_nearest_neighbors(candidates);

            /* project query to a cube vertex / hash table bucket */
            const std::string key = cube_projection_test(query);
            std::string key1 = key;

            while (M > 0) {
                if (probes > 0) {
                    bucket_count = hash_table[key1].size();
                    const std::vector<size_t> &indexes = hash_table[key1]; 
                    for (size_t i = 0; (i != bucket_count) && (M > 0); ++i, --M) {
                        dist = manhattan_distance_rd<T>(query, train_samples[indexes[i]]);
                        if (dist < candidates[0].first) {
                            candidates[0] = std::make_pair(dist, indexes[i]);
                            std::sort(candidates.begin(), candidates.end(), [](const std::pair<uint32_t, size_t> &left, \
                                                                                const std::pair<uint32_t, size_t> &right) \
                                                                                { return (left.first > right.first); } );

                        }
                    }

                    /* generate a "nearby" vertex using hamming distance (hamming distance = 1, then hamming distance = 2, etc) */
                    key1 = gen_nearby_vertex(key, cnt, bits);
                    /* can't generate hamming distance = x > cube dimension */
                    if (bits > projection_dimension) break;
                    --probes;
                }
                else
                    break;
            }

            std::sort(candidates.begin(), candidates.end(), [](const std::pair<uint32_t, size_t> &left, \
                                                                const std::pair<uint32_t, size_t> &right) \
                                                                { return (left.first < right.first); } );

            return candidates;
        }


        std::vector<size_t> range_search(const std::vector<T> &query, const std::vector<std::vector<T>> &train_samples)
        {
            /* vector to store query's nearest neighbors; only store the training index this time */
            std::vector<size_t> candidates;
            uint32_t dist = 0;
            uint32_t cnt = projection_dimension;
            uint16_t M = max_candidates;
            uint16_t probes = max_probes;
            uint8_t  bits = 1;
            size_t   bucket_count = 0;

            /* project query to a cube vertex / hash table bucket */
            const std::string key = cube_projection_test(query);
            std::string key1 = key;

            while (M > 0) {
                if (probes > 0) {
                    bucket_count = hash_table[key1].size();
                    const std::vector<size_t> &indexes = hash_table[key1];
                    for (size_t i = 0; (i != bucket_count) && (M > 0); ++i, --M) {
                        dist = manhattan_distance_rd<T>(query, train_samples[indexes[i]]);
                        if (dist < C * R) {     // average distance is 20 000 - 35 000
                            candidates.push_back(indexes[i]);
                        }
                    }

                    /* generate a "nearby" vertex using hamming distance (hamming distance = 1, then hamming distance = 2, etc) */
                    key1 = gen_nearby_vertex(key, cnt, bits);
                    /* can't generate hamming distance = x > cube dimension */
                    if (bits > projection_dimension) break;
                    --probes;
                }
                else
                    break;
            }

            return candidates;
        }


        std::vector<size_t> range_search(const std::vector<T> &query, const std::vector<std::vector<T>> &train_samples, double r)
        {
            /* vector to store query's nearest neighbors; only store the training index this time */
            std::vector<size_t> candidates;
            uint32_t dist = 0;
            uint32_t cnt = projection_dimension;
            uint16_t M = max_candidates;
            uint16_t probes = max_probes;
            uint8_t  bits = 1;
            size_t   bucket_count = 0;

            /* project query to a cube vertex / hash table bucket */
            const std::string key = cube_projection_test(query);
            std::string key1 = key;

            while (M > 0) {
                if (probes > 0) {
                    bucket_count = hash_table[key1].size();
                    const std::vector<size_t> &indexes = hash_table[key1];
                    for (size_t i = 0; (i != bucket_count) && (M > 0); ++i, --M) {
                        dist = manhattan_distance_rd<T>(query, train_samples[indexes[i]]);
                        if (dist < C * r) {     // average distance is 20.000 - 35.000
                            candidates.push_back(indexes[i]);
                        }
                    }

                    /* generate a "nearby" vertex using hamming distance (hamming distance = 1, then hamming distance = 2, etc) */
                    key1 = gen_nearby_vertex(key, cnt, bits);
                    /* can't generate hamming distance = x > cube dimension */
                    if (bits > projection_dimension) break;
                    --probes;
                }
                else
                    break;
            }

            return candidates;
        }


        std::string gen_nearby_vertex(const std::string &key, uint32_t &counter, uint8_t &bits)
        {
            std::string bitstring = key;
            for (size_t i = 0; i < bits && counter != 0; ++i, --counter) {
                bitstring[counter - 1] == '0' ? bitstring[counter - 1] = '1' : bitstring[counter - 1] = '0';
                if (counter == 1 && i < bits) { // wrap-around
                    counter = projection_dimension;
                }
            }

            if(counter == 0) {
                counter = projection_dimension; // reset bit counter
                ++bits;
            }

            return bitstring;
        }
};

#endif
