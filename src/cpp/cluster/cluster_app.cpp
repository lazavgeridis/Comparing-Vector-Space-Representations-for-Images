#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

#include "../../../include/io_utils/io_utils.h"
#include "../../../include/cluster/cluster_utils.h"
#include "../../../include/cluster/cluster.h"


int main(int argc, char *argv[])  {

    if (argc != 11) cluster_usage(argv[0]);
    
    cluster_args args;  
    cluster_configs configs = {};

    parse_cluster_args(argc, argv, &args);
    parse_cluster_configurations(args.config_file, &configs);

    /* load data - original vector space, new vector space, neural network clusters */
    std::vector<std::vector<uint8_t>> dataset_original;
    std::cout << "\nReading input dataset (original vector space) from \"" << args.input_file_original << "\" ..." << std::endl;
    read_dataset<uint8_t> (args.input_file_original, dataset_original);
    std::cout << "Done!" << std::endl;

    std::vector<std::vector<uint16_t>> dataset_new;
    std::cout << "\nReading input dataset (new vector space) from \"" << args.input_file_new << "\" ..." << std::endl;
    read_dataset<uint16_t> (args.input_file_new, dataset_new);
    std::cout << "Done!" << std::endl;

    std::cout << "\nReading nn clusters file from \"" << args.nn_clusters_file << "\" ..." << std::endl;
    std::cout << "Done!" << std::endl;

    /* based on the assignment method specified by the user, use the appropriate constructor for Cluster */
    //Cluster<uint8_t> *cluster = new Cluster<uint8_t> (configs.number_of_clusters);

    //std::cout << "\nK-Medians++ is executing..." << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    //cluster->k_medians_plus_plus(train_data, args.method);
    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    //std::cout << "Done!" << std::endl;

    //std::cout << "\nComputing clustering silhouette..." << std::endl;
    //cluster->silhouette(train_data);
    //std::cout << "Done!" << std::endl;

    //std::cout << "\nWriting formatted output to \"" << args.output_file << "\" ..." << std::endl;
    //cluster->write_cluster_output(args.output_file , args.method, args.complete, duration);
    //std::cout << "Done!" << std::endl;

    //delete cluster;

    return EXIT_SUCCESS;
}
