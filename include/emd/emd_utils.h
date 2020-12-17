#ifndef EMD_UTILS_H
#define EMD_UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "../io_utils/cmd_args.h"


template<typename T>
uint32_t euclidean(T pixel1, T pixel2) {
    return (pixel1 - pixel2) * (pixel1 - pixel2);
}


template<typename T>
T center(const std::vector<T> &cluster, uint16_t dimension) {
    return cluster[ (dimension / 2) + 1 ];
}


template<typename T>
void emd(std::vector<std::vector<T>> &train_samples, std::vector<T> &query, std::vector<size_t> &nns) {

    /* make sure the dimensionality of the training set is equal to that of the query set */
    assert(train_samples[0].size() == query.size());

    /* get image resolution: dim x dim */
    uint32_t dimension_squared = query.size();
    uint16_t dimension = (uint16_t) sqrt( (double) dimension_squared );
    std::cout << "Each image has dimension " << dimension << "x" << dimension << std::endl;
    
    /* each cluster has 4x4 pixels (or 7x7 pixels) */
    const uint8_t pixels = 4;
    
    /* each image consists of ClustersxClusters clusters */
    const uint8_t clusters = dimension / pixels;
    
    std::cout << "In this case we have " << clusters * clusters << " clusters with " << pixels * pixels << " pixels each" << std::endl;

    size_t query_offset, query_index, p_offset, p_index;
    uint32_t dist;
    std::vector<T> qvector, pvector;

    //for (auto &p : train_samples) {
    auto p = train_samples[0];

        for(size_t query_cluster = 0; query_cluster < clusters * clusters; ++query_cluster) {

                //std::cout << "Query cluster " << query_cluster << std::endl;
                query_offset = (query_cluster / clusters) * dimension * pixels;
                query_offset += (query_cluster % clusters) * pixels;

            for(size_t p_cluster = 0; p_cluster < clusters * clusters; ++p_cluster) {

                //std::cout << "Training point cluster " << p_cluster << std::endl;

                p_offset = (p_cluster / clusters) * dimension * pixels;
                p_offset += (p_cluster % clusters) * pixels;

                for(size_t i = 0; i < pixels; ++i) {
                    for(size_t j = 0; j < pixels; ++j) {
                        query_index = query_offset + (i * dimension) + j;
                        p_index = p_offset + (i * dimension) + j;
                        qvector.push_back(query[query_index]);
                        pvector.push_back(p[p_index]);
                    }
                }

                /* calculate the weight of the clusters... */
                dist = euclidean(center(qvector, dimension), center(pvector, dimension));
                std::cout << "Distance between q cluster " << query_cluster \
                          << " and point cluster " << p_cluster << " is " \
                          << dist << std::endl;
                pvector.clear();
            }
            qvector.clear();
        }

    //}
}

#endif
