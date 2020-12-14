#ifndef EXACT_NN
#define EXACT_NN

#include <vector>
#include <limits> 
#include <algorithm>

#include "../../../include/metric/metric.h"


template<typename T>
std::vector<uint32_t> exact_nn(const std::vector<std::vector<T>> &dataset, \
                                                  const std::vector<T> &query, uint16_t nns) {

    std::vector<uint32_t> closest_distances(nns); 

    for (size_t i = 0; i != nns; ++i) {
        closest_distances[i] = std::numeric_limits<uint32_t>::max();
    }

    for (size_t i = 0; i != dataset.size(); ++i) {
        uint32_t dist = manhattan_distance_rd<T> (dataset[i], query);
        if (dist < closest_distances[0]) {
            closest_distances[0] = dist;
            std::sort(closest_distances.begin(), closest_distances.end(), [](const uint32_t &left, const uint32_t &right) \
                                                                             { return (left > right); } );
        }
    }
    std::sort(closest_distances.begin(), closest_distances.end());

    return closest_distances;
}


template<typename T>
uint32_t exact_nn(const std::vector<std::vector<T>> &centroids, const std::vector<T> &point) {

    uint32_t min_dist = std::numeric_limits<uint32_t>::max();
    uint32_t dist = 0;

    for (size_t i = 0; i != centroids.size(); ++i) {
        dist = manhattan_distance_rd<T> (centroids[i], point);
        if (dist < min_dist) min_dist = dist;
    }

    return min_dist;
}

#endif // EXACT_NN
