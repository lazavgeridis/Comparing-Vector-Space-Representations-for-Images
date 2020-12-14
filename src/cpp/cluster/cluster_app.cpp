#include <iostream>
#include <vector>
#include <string>
#include <utility>

#include "../../include/io_utils/io_utils.h"
#include "../../include/cluster/cluster_utils.h"
#include "../../include/cluster/cluster.h"


int main(int argc, char *argv[])  {

    cluster_args args;  
    cluster_configs configs = {};
    Cluster<uint8_t> *cluster = nullptr;

    if (argc != 10 && argc != 9) cluster_usage(argv[0]);
    
    parse_cluster_args(argc, argv, &args);
    parse_cluster_configurations(args.config_file, &configs);


    /* read training set file and store the training data */
    std::vector<std::vector<uint8_t>> train_data;
    std::cout << "\nReading training set from \"" << args.input_file << "\"..." << std::endl;
    read_file<uint8_t> (args.input_file, train_data);
    std::cout << "Done!" << std::endl;

    /* based on the assignment method specified by the user, use the appropriate constructor for Cluster */
    if (args.method == "Classic") {
        cluster = new Cluster<uint8_t> (configs.number_of_clusters);
    }
    else if (args.method == "LSH") {
        double r = mean_nn_distance<uint8_t> (train_data);
        cluster = new Cluster<uint8_t> (configs.number_of_clusters, configs.number_of_hash_tables, 0,  \
                                            configs.number_of_hash_functions, r, train_data);
    }
    else {
        double r = mean_nn_distance<uint8_t> (train_data);
        cluster = new Cluster<uint8_t> (    configs.number_of_clusters, configs.hypercube_dimensions, \
                                            configs.max_number_M_hypercube, configs.number_of_probes, 0, 0.0, \
                                            train_data.size(), train_data[0].size(), r, train_data);
    }

    std::cout << "\nK-Medians++ is executing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cluster->k_medians_plus_plus(train_data, args.method);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Done!" << std::endl;

    std::cout << "\nCalculating clustering silhouette..." << std::endl;
    cluster->silhouette(train_data);
    std::cout << "Done!" << std::endl;

    std::cout << "\nWriting formatted output to \"" << args.output_file << "\"..." << std::endl;
    cluster->write_cluster_output(args.output_file , args.method, args.complete, duration);
    std::cout << "Done!" << std::endl;

    delete cluster;

    return EXIT_SUCCESS;
}
