#ifndef EMD_UTILS_H
#define EMD_UTILS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "../io_utils/cmd_args.h"
#include "ortools/linear_solver/linear_solver.h"

using namespace operations_research;

template<typename T>
uint32_t euclidean(T pixel1, T pixel2) {
    return (pixel1 - pixel2) * (pixel1 - pixel2);
}


template<typename T>
T center(const std::vector<T> &cluster) {
    size_t size = cluster.size();

    return cluster[ (size / 2) + 1 ];
}


template<typename T>
uint32_t cluster_weight(const std::vector<T> &image, const std::vector<T> &cluster) {
    uint32_t image_sum = 0;
    uint32_t cluster_sum = 0;

    for(auto i = image.cbegin(); i != image.cend(); ++i) {
        image_sum += *i;
    }
    for(auto i = cluster.cbegin(); i != cluster.cend(); ++i) {
        cluster_sum += *i;
    }

    return cluster_sum / image_sum;
}


template<typename T>
void emd(std::vector<std::vector<T>> &train_samples, std::vector<T> &query, std::vector<size_t> &nns) {

    /* make sure the dimensionality of the training set is equal to that of the query set */
    assert(train_samples[0].size() == query.size());

    /* get image resolution: dim x dim -> in our case, 28 x 28 */
    uint32_t dimension_squared = query.size();
    uint16_t dimension = (uint16_t) sqrt( (double) dimension_squared );
    std::cout << "Each image has dimension " << dimension << "x" << dimension << std::endl;
    
    /* each cluster has 4x4 pixels or 7x7 pixels */
    const uint8_t pixels = 7;
    
    /* each image consists of (dimension / pixels) x (dimension / pixels) clusters 
     * i.e 7x7 clusters or 4x4 clusters 
     */
    const uint8_t  clusters = dimension / pixels;
    const uint16_t n = clusters * clusters;
    
    std::cout << "In this case we have " << n << " clusters with " << pixels * pixels << " pixels each" << std::endl;

    size_t query_offset, query_index, p_offset, p_index;
    uint32_t dist;
    std::vector<T> qcluster, pcluster;
    std::vector<uint32_t> distances;
    std::vector<uint32_t> q_cluster_weight(clusters * clusters, 0);
    std::vector<uint32_t> p_cluster_weight(clusters * clusters, 0);
    std::vector<MPVariable*> flows;
    std::vector<MPConstraint *> row_constraints(clusters * clusters, 0);
    std::vector<MPConstraint *> column_constraints(clusters * clusters, 0);
    T center1, center2;

    MPSolver* solver = MPSolver::CreateSolver("GLOP");
    const double infinity = solver->infinity();

    //for (auto &p : train_samples) {
    auto p = train_samples[0];

        /* create a variable vector representing flows */
        solver->MakeNumVarArray(n * n, 0.0, infinity, "flow", &flows);

        for(size_t query_cluster_index = 0; query_cluster_index < n; ++query_cluster_index) {
            
            //std::cout << "Query cluster " << query_cluster << std::endl;
            query_offset = (query_cluster_index / clusters) * dimension * pixels;
            query_offset += (query_cluster_index % clusters) * pixels;

            /* store pixels of query cluster */
            for(size_t i = 0; i < pixels; ++i) {
                for(size_t j = 0; j < pixels; ++j) {
                    query_index = query_offset + (i * dimension) + j;
                    qcluster.push_back(query[query_index]);
                }
            }

            /* (query): get cluster's "centroid" */
            center1 = center(qcluster);

            /* (query): calculate cluster's weight */
            q_cluster_weight[query_cluster_index] = cluster_weight(query, qcluster);

            for(size_t p_cluster_index = 0; p_cluster_index < n; ++p_cluster_index) {

                //std::cout << "Training point cluster " << p_cluster << std::endl;
                
                p_offset = (p_cluster_index / clusters) * dimension * pixels;
                p_offset += (p_cluster_index % clusters) * pixels;

                for(size_t i = 0; i < pixels; ++i) {
                    for(size_t j = 0; j < pixels; ++j) {
                        p_index = p_offset + (i * dimension) + j;
                        pcluster.push_back(p[p_index]);
                    }
                }

                center2 = center(pcluster);

                /* cluster's weight */
                p_cluster_weight[p_cluster_index] = cluster_weight(p, pcluster);

                /* Calculate distance between query_cluster_index (i) and p_cluster_index (j) */
                distances.push_back(euclidean(center1, center2));

                pcluster.clear();
            }
            qcluster.clear();

        }

        /* define "row" constraints */
        for(size_t i = 0; i < n; ++i) {
            row_constraints[i] = solver->MakeRowConstraint(q_cluster_weight[i], q_cluster_weight[i]);
            for(size_t j = 0; j < n; ++j) {
                row_constraints[i]->SetCoefficient(flows[i * clusters * clusters + j], 1);
            }
        }

        /* define "column" constraints */
        for(size_t j = 0; j < n; ++j) {
            column_constraints[j] = solver->MakeRowConstraint(p_cluster_weight[j], p_cluster_weight[j]);
            for(size_t i = 0; i < n; ++i) {
                column_constraints[j]->SetCoefficient(flows[i * clusters * clusters + j], 1);
            }
        }

        LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

        /* define objective function to minimize */
        MPObjective* const objective = solver->MutableObjective();
        for(size_t i = 0; i < n; ++i) {
            objective->SetCoefficient(flows[i], distances[i]);
        }
        objective->SetMinimization();

        /* invoke solver */
        const MPSolver::ResultStatus result_status = solver->Solve();
        if (result_status != MPSolver::OPTIMAL) {
          LOG(FATAL) << "The problem does not have an optimal solution!";
        }
        LOG(INFO) << "Solution:";
        LOG(INFO) << "Optimal objective value = " << objective->Value();
        
        flows.clear();
        distances.clear();
    //}
}

#endif
