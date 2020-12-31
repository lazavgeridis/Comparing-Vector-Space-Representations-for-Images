#ifndef CLUSTER_H
#define CLUSTER_H

#include <algorithm> /* for sort() */
#include <vector>
#include <string>
#include <random>   /* for rand() */
#include <chrono> 
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cmath>    /* for ceil(), abs() */
#include <cassert>

#include "../modules/exact_nn/exact_nn.h"
#include "../cluster/cluster_utils.h"

#define EPSILON 500000


template <typename T>
class Cluster {

    private:

        /* number of clusters to be created */
        const size_t         num_clusters;

        /* 
         * clusters is a vector, where each element is another 
         * vector storing the indexes of the training set points 
         * that are assigned to that cluster 
         * i.e clusters[i] is a vector storing the indexes of all
         *     the points assigned to cluster with index i
         */
        std::vector<std::vector<size_t>>  clusters;

        /* 
         * centroids is a vector storing the actual components
         * of each centroid
         */
        std::vector<std::vector<T>>       centroids;

        /* silhouette for each cluster */
        std::vector<double> avg_sk;

        /* silhouette for overall clustering */
        double stotal = 0.0;

        /* clustering objective value */
        int64_t objective;


    public:

        /* Constructor if method = Lloyds Assignment */
        Cluster(size_t nclusters): num_clusters(nclusters)
        {
            clusters.resize(num_clusters);
            avg_sk.resize(num_clusters, 0.0);
        }


        Cluster(const std::vector<std::vector<T>> &dataset, const std::vector<std::vector<size_t>> &nn_clusters) : \
        num_clusters(nn_clusters.size()), clusters(nn_clusters)
        {
            // compute the centroid of each cluster
            centroids.resize(num_clusters, std::vector<T> (dataset[0].size(), 0));
            avg_sk.resize(num_clusters, 0.0);
            median_update(dataset);
        }


        ~Cluster() = default;


        void init_plus_plus(const std::vector<std::vector<T>> &dataset, std::vector<size_t> &centroid_indexes)
        {
            std::vector<std::pair<float, size_t>>   partial_sums;
            std::vector<float>                      min_distances(dataset.size());

            /* randomly select the index of the 1st centroid from the training set */
            std::default_random_engine generator;
            srand( ( unsigned ) time(NULL) );
            size_t size = dataset.size();
            size_t index = rand() % size;

            /* emplace dataset[index] to the centroids vector */
            centroids.emplace_back(dataset[index]);
            /* add the centroid index to the centroid_indexes vector */
            centroid_indexes[0] = index;

            for (size_t t = 1; t != num_clusters; ++t) {

                for (size_t i = 0; i != size; ++i) {

                    /* if training sample with index i is one of the k centroids,
                     * don't calculate the distance with itself
                     */
                    if ( in(centroid_indexes, i) ) continue;

                    min_distances[i] = exact_nn<T> (centroids, dataset[i]);
                }

                /* normalize D(i)'s */
                normalize_distances(min_distances);

                /* calculate n - t partial sums */
                float prev_partial_sum = 0.0;
                float new_partial_sum  = 0.0;
                partial_sums.emplace_back(0.0, 0);      // P(0) = 0
                for (size_t j = 0; j != size; ++j) {

                    if ( in(centroid_indexes, j) ) continue;

                    new_partial_sum = prev_partial_sum + (min_distances[j] * min_distances[j]); // P(r) = P(r - 1) + (D(r) * D(r))
                    partial_sums.emplace_back(new_partial_sum, j);
                    prev_partial_sum = new_partial_sum;
                }

                /* generate uniformly distributed x in [0, P(n - t)]
                 * do binary search on the sorted vector containing pairs of (partial sum, index)
                 * function binary_search() returns index r of the training sample that is the next centroid
                 */
                std::uniform_real_distribution<float> distribution(0.0, new_partial_sum);
                float x = distribution(generator);
                std::sort(partial_sums.begin(), partial_sums.end(), compare);
                size_t r = binary_search(partial_sums, x);

                /* emplace train_set[r] to the centroids vector */
                centroids.emplace_back(dataset[r]);
                /* add new centroid index to the centroid_indexes vector */
                centroid_indexes[t] = r;
                /* next iteration: partial_sum's size will be decreased by one */
                partial_sums.clear();   
            }
        }


        /* this version of lloyd's assignment is used ONLY in the first iteration of k-medians++ */
        void lloyds_assignment(const std::vector<std::vector<T>> &dataset, std::vector<size_t> &centroid_indexes)
        {
            uint32_t min_dist{};
            uint32_t dist{};
            const size_t size = dataset.size();

            /* for each point compute l1 metric distance to every centroid */
            for (size_t i = 0; i != size; ++i)  {
                /* point with index i is one of k centers, so do not assign it to another center */
                if ( in(centroid_indexes, i) ) continue;
                min_dist = std::numeric_limits<uint32_t>::max();
                size_t best_centroid{};
                for (size_t j = 0; j != centroids.size(); ++j) {
                    dist = manhattan_distance_rd<T> (dataset[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                /* assign i-th training set point to cluster (center) with which it has the shortest distance */
                clusters[best_centroid].emplace_back(i);
            }
        }


        void lloyds_assignment(const std::vector<std::vector<T>> &dataset)
        {
            uint32_t min_dist{};
            uint32_t dist{};
            const size_t size = dataset.size();

            /* for each point compute l1 metric distance to every centroid */
            for (size_t i = 0; i != size; ++i)  {
                min_dist = std::numeric_limits<uint32_t>::max();
                size_t best_centroid{};
                for (size_t j = 0; j != centroids.size(); ++j) {
                    dist = manhattan_distance_rd<T> (dataset[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                /* assign i-th training set point to cluster (center) with which it has the shortest distance */
                clusters[best_centroid].emplace_back(i);
            }
        }


        void median_update(const std::vector<std::vector<T>> &dataset) 
        {
            assert(centroids.size() == clusters.size());

            const size_t        dim = centroids[0].size();
            std::vector<T>      components;

            for (size_t k = 0; k != num_clusters; ++k) {

                std::vector<T> &k_centroid = centroids[k];
                size_t cluster_size = clusters[k].size();
                components.resize(cluster_size);
                const std::vector<size_t> &cluster_indexes = clusters[k];

                for (size_t d = 0; d != dim; ++d) {

                    for (size_t t = 0; t != cluster_size; ++t) {

                        const std::vector<T> &t_vector = dataset[cluster_indexes[t]];
                        components[t] = t_vector[d];
                    }
                    std::sort(components.begin(), components.end());
                    size_t median_index = std::ceil(cluster_size / 2);
                    k_centroid[d] = components[median_index];
                }
            }
        }


        uint64_t objective_function(const std::vector<std::vector<T>> &dataset)
        {
            const size_t size = dataset.size();
            uint32_t min_dist = 0;
            uint64_t l1_norm = 0;

            for (size_t i = 0; i != size; ++i) {
                min_dist = exact_nn<T> (centroids, dataset[i]);
                l1_norm += min_dist;
            }

            return l1_norm;
        }


        void k_medians_plus_plus(const std::vector<std::vector<T>> &dataset)
        {
            std::vector<size_t> centroid_indexes(num_clusters);


            // initialization++ 
            init_plus_plus(dataset, centroid_indexes);

            /* 
             * in the first assignment operation, a point can be assigned to a center, 
             * which is actually itself; so the first iteration requires special handling.
             * after the first median update, it is not possible for the new centers
             * to be points of the training set
             */
            lloyds_assignment(dataset, centroid_indexes);

            // step 2: median update
            median_update(dataset);              

            for (auto &cluster : clusters) {
                    cluster.clear();
            }
            
            int64_t prev_objective = 0;
            int64_t new_objective  = 0;
            /* repeat steps (1) and (2) until change in cluster assignments is "small" */
            while (1) {  

                // step 1: assignment
                lloyds_assignment(dataset);

                // step 2: median update
                median_update(dataset);              

                // calculate k-medians objective function after centroids are updated
                new_objective = objective_function(dataset);

                std::cout << "\nObjective of n-1 is " << prev_objective << std::endl;
                std::cout << "Objective of n   is " << new_objective << std::endl;

                // terminating condition
                if ( std::abs(prev_objective - new_objective) < EPSILON )
                    break;
                
                /* 
                 * after the centroids are updated, each vector in clusters should be cleared;
                 * in the next iteration the points assigned to each cluster will be different
                 */
                for (auto &cluster : clusters) {
                        cluster.clear();
                }

                prev_objective = new_objective;
            }

            objective = new_objective;
        }


        void silhouette(const std::vector<std::vector<T>> &dataset)
        {
            const size_t n_vectors = dataset.size();

            std::vector<double> s(n_vectors);
            std::vector<double> a(n_vectors);
            std::vector<double> b(n_vectors);

            /* compute a[i] values */
            for (auto it = clusters.cbegin(); it != clusters.cend(); ++it) {
                const std::vector<size_t> &each_cluster_vector_indexes = *it; // reference instead of copying it
                for (size_t i = 0; i != each_cluster_vector_indexes.size(); ++i) {
                    size_t total_a_dist{};
                    for (size_t j = 0; j != each_cluster_vector_indexes.size(); ++j) {
                        if (i == j) continue;
                        total_a_dist += manhattan_distance_rd<T> (dataset[each_cluster_vector_indexes[i]], \
                                                                    dataset[each_cluster_vector_indexes[j]]);
                    }
                    if (each_cluster_vector_indexes.size() > 1) {
                        a[each_cluster_vector_indexes[i]] = (double) total_a_dist / each_cluster_vector_indexes.size(); 
                    }
                    else {
                        a[each_cluster_vector_indexes[i]] = (double) total_a_dist;  // in this case a[i] = 0
                    }
                }
            }

            /* compute closest centroid to each centroid */
            std::vector<size_t> closest_centroids(centroids.size());
            for (size_t i = 0; i != centroids.size(); ++i) {
                uint32_t min_dist = std::numeric_limits<uint32_t>::max();
                size_t closest = 0;
                for (size_t j = 0; j != centroids.size(); ++j) {
                    if (i == j) continue;
                    uint32_t dist = manhattan_distance_rd<T> (centroids[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest = j;
                    }
                }
                closest_centroids[i] = closest; // indicating that i-th centroid is closer to the j-th centroid 
            }

            /* compute b[i] values */
            for (size_t k = 0; k != clusters.size(); ++k) {
                const std::vector<size_t> &each_cluster_vector_indexes = clusters[k];
                const std::vector<size_t> &closest_cluster_vector_indexes = clusters[closest_centroids[k]];
                for (size_t i = 0; i != each_cluster_vector_indexes.size(); ++i) {
                    size_t total_b_dist{};
                    for (size_t j = 0; j != closest_cluster_vector_indexes.size(); ++j) {
                        total_b_dist += manhattan_distance_rd<T> (dataset[each_cluster_vector_indexes[i]], \
                                                                    dataset[closest_cluster_vector_indexes[j]]);
                    }
                    if (closest_cluster_vector_indexes.size() > 0) {
                        b[each_cluster_vector_indexes[i]] = (double) total_b_dist / closest_cluster_vector_indexes.size(); 
                    }
                    else {
                        b[each_cluster_vector_indexes[i]] = (double) total_b_dist;
                    }
                }
            }

            /* compute s[i] values */
            for (size_t i = 0; i != n_vectors; ++i) {
                s[i] = (b[i] - a[i]) / std::max(a[i], b[i]);
            }
            /* compute average s(p) of points in cluster i */
            for (size_t i = 0; i != centroids.size(); ++i) {
                const std::vector<size_t> &each_cluster_vector_index = clusters[i];
                size_t n_vectors = each_cluster_vector_index.size();
                for (size_t j = 0; j != n_vectors; ++j) {
                    avg_sk[i] += s[each_cluster_vector_index[j]];
                }
                if (n_vectors != 0) {
                    avg_sk[i] /= n_vectors;
                }
            }
            /* compute stotal = average s(p) of points in dataset */
            uint32_t n_centroids = centroids.size();

            for (size_t i = 0; i != n_centroids; ++i) {
                stotal += avg_sk[i];
            }
            stotal /= n_centroids;
        }


        void copy_clusters(std::vector<std::vector<size_t>> &copy) 
        {
            copy = clusters;
        }


        void write_cluster_output(const std::string &out, const std::string &header, double clustering_time, uint64_t objval = 0)
        {
            std::ofstream ofile;
            ofile.open(out, std::ios::out | std::ios::app);

            if (ofile) {
                ofile << header << std::endl;
                for (size_t i = 0; i != clusters.size(); ++i) {
                    ofile << "CLUSTER-" << i + 1 << " {size: " << clusters[i].size() << ", centroid: [";
                    for (auto &c : centroids[i]) {
                        ofile << +c << " "; 
                    }
                    ofile << "]}" << std::endl;
                }
                ofile << "clustering_time: " << clustering_time << " seconds" << std::endl;
                ofile << "Silhouette: [";
                for (auto &s : avg_sk) {
                    ofile << s << ", ";
                }
                ofile << stotal <<"]" << std::endl;
                if (!objval)
                    ofile << "Value of Objective Function: " << objective << "\n" << std::endl;
                else
                    ofile << "Value of Objective Function: " << objval << "\n" << std::endl;
                ofile.close();
            }
            else {
                std::cerr << "\nCould not open output file!\n" << std::endl;
            }
        }


        void write_cluster_output(const std::string &out, const std::string &header, uint64_t objval)
        {
            std::ofstream ofile;
            ofile.open(out, std::ios::out | std::ios::app);

            if (ofile) {
                ofile << header << std::endl;
                ofile << "Silhouette: [";
                for (auto &s : avg_sk) {
                    ofile << s << ", ";
                }
                ofile << stotal <<"]" << std::endl;
                ofile << "Value of Objective Function: " << objval << "\n" << std::endl;
                ofile.close();
            }
            else {
                std::cerr << "\nCould not open output file!\n" << std::endl;
            }
        }
};

#endif
