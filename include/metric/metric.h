#ifndef METRIC_H
#define METRIC_H

#include <iterator>
#include <vector>
#include <cassert>
#include <iostream>
#include <limits>
#include <map>


template <typename T>
inline uint32_t manhattan_distance_rd(const std::vector<T> &dataset_vector, const std::vector<T> &query_vector) {
    assert(dataset_vector.size() == query_vector.size());
    
    uint32_t dist{};
    auto d_it = dataset_vector.cbegin();
    auto q_it = query_vector.cbegin();

    for (; d_it != dataset_vector.cend(); ++d_it, ++q_it) {
        dist += std::abs((*d_it) - (*q_it));
    }

    return dist;
}


template <typename T>
inline uint32_t mean_nn_distance(const std::vector<std::vector<T>> &dataset) {
    
    uint32_t dist{};
    uint32_t sum{};
    uint32_t best_dist = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i != dataset.size() / 600; ++i) {

        for (size_t j = 0; j != dataset.size() / 600; ++j) {

            if (i == j) continue;

            dist = manhattan_distance_rd<T>(dataset[i], dataset[j]);
            if (dist < best_dist) best_dist = dist;
        }
        sum += best_dist;
        best_dist = std::numeric_limits<uint32_t>::max();

    }

    return ( sum / ( dataset.size() / 60) );
}

#endif // METRIC_H
