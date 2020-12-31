#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <chrono>
#include <cstdint>

#include "../../../include/io_utils/io_utils.h"
#include "../../../include/cluster/cluster_utils.h"
#include "../../../include/cluster/cluster.h"


static void compare_clusterings(cluster_args args, std::vector<std::vector<uint16_t>> &dataset_new, \
                                std::vector<std::vector<uint8_t>> &dataset_original, Cluster<uint16_t> *new_space, \
                                Cluster<uint8_t> *orig_space, Cluster<uint8_t> *nn) {

    /***** K-medians++ for new vector space data (10d) *****/
    std::cout << "\nNew Vector Space (10d):\nExecuting K-medians++ ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    new_space->k_medians_plus_plus(dataset_new);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Done!" << std::endl;
    std::vector<std::vector<size_t>> c;
    new_space->copy_clusters(c);
    Cluster<uint8_t> *clustering_change_vecspace = new Cluster<uint8_t> (dataset_original, c);
    std::cout << "\nComputing clustering silhouette ..." << std::endl;
    clustering_change_vecspace->silhouette(dataset_original);
    std::cout << "Done!" << std::endl;
    std::cout << "\nWriting formatted output to \"" << args.output_file << "\" ..." << std::endl;
    clustering_change_vecspace->write_cluster_output(args.output_file, "NEW SPACE", duration.count(), \
                                                clustering_change_vecspace->objective_function(dataset_original));
    std::cout << "Done!" << std::endl;

    /***** K-medians++ for original vector space data (784d) *****/
    std::cout << "\nOriginal Vector Space (784d):\nExecuting K-medians++ ..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    orig_space->k_medians_plus_plus(dataset_original);
    stop = std::chrono::high_resolution_clock::now();
    duration = stop - start;
    std::cout << "Done!" << std::endl;
    std::cout << "\nComputing clustering silhouette ..." << std::endl;
    orig_space->silhouette(dataset_original);
    std::cout << "Done!" << std::endl;
    std::cout << "\nWriting formatted output to \"" << args.output_file << "\" ..." << std::endl;
    orig_space->write_cluster_output(args.output_file, "ORIGINAL SPACE", duration.count());
    std::cout << "Done!" << std::endl;

    /***** Evaluation of nn clusters *****/
    std::cout << "\nEvaluation of NN's predicted clusters:\n" << std::endl;
    std::cout << "Computing clustering silhouette ..." << std::endl;
    nn->silhouette(dataset_original);
    std::cout << "Done!" << std::endl;
    std::cout << "\nWriting formatted output to \"" << args.output_file << "\" ..." << std::endl;
    nn->write_cluster_output(args.output_file, "CLASSES AS CLUSTERS", nn->objective_function(dataset_original));
    std::cout << "Done!" << std::endl;
}


int main(int argc, char *argv[]) {

    if (argc != 11) cluster_usage(argv[0]);
    
    cluster_args args;  
    cluster_configs configs = {};

    parse_cluster_args(argc, argv, &args);
    parse_cluster_configurations(args.config_file, &configs);

    std::vector<std::vector<uint8_t>> dataset_original;
    std::vector<std::vector<uint16_t>> dataset_new;
    std::vector<std::vector<size_t>> nn_clusters(configs.number_of_clusters);

    /* load data - original vector space, new vector space, neural network's predicted "clusters" */
    load_data(args, dataset_original, dataset_new, nn_clusters);

    Cluster<uint8_t>  *clustering_orig_space = new Cluster<uint8_t> (configs.number_of_clusters);
    Cluster<uint16_t> *clustering_new_space  = new Cluster<uint16_t> (configs.number_of_clusters);
    Cluster<uint8_t>  *clustering_nn  = new Cluster<uint8_t> (dataset_original, nn_clusters);

    compare_clusterings(args, \
                        dataset_new, \
                        dataset_original, \
                        clustering_new_space, \
                        clustering_orig_space, \
                        clustering_nn);

    delete clustering_orig_space;
    delete clustering_new_space;
    delete clustering_nn;

    return EXIT_SUCCESS;
}
