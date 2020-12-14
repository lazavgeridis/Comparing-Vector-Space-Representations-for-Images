#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <sys/time.h>

#include "../../include/io_utils/cmd_args.h"
#include "../../include/io_utils/io_utils.h"
#include "../../include/modules/lsh/lsh.h"
#include "../../include/modules/exact_nn/exact_nn.h"


#define C 1.2


static void start_lsh_simulation(Lsh_args *args) {

    /* Create LSH structure */
    uint16_t L = args->get_hash_tables_num();
    uint16_t N = args->get_nearest_neighbors_num();
    uint32_t K = args->get_k();
    double R = args->get_radius();

    /* read training set and store each data sample */
    std::cout << "\nReading training set..." << std::endl;
    std::vector<std::vector<uint8_t>> dataset;
    read_file<uint8_t> (args->get_input_file_path(), dataset);
    std::cout << "Done!" << std::endl;

    std::cout << "\nComputing mean nearest neighbor distance..." << std::endl;
    double r = mean_nn_distance<uint8_t> (dataset); 
    std::cout << "Done!" << std::endl;

    std::cout << "\nCreating LSH structure..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    LSH<uint8_t> lsh = LSH<uint8_t> (L, N, K, r, dataset);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start); 
    std::cout << "LSH Structure Creation lasted " << duration.count() << " seconds" << std::endl;

    while (1) {   
        /* read test set and store each query point */
        std::vector<std::vector<uint8_t>> queries;
        std::cout << "\nReading test set from \"" << args->get_query_file_path() << "\"..." << std::endl;
        read_file<uint8_t> (args->get_query_file_path(), queries);
        std::cout << "Done!" << std::endl;

        std::cout << "\nQuery file contains " << queries.size() << " queries" << std::endl;
        std::cout << "\nStart Executing ANN / ENN / Range-Search" << std::endl;
        std::cout << "..." << std::endl;

        /********** Start ANN / ENN / Range search **********/
        std::vector<std::vector<std::pair<uint32_t, size_t>>>  ann_results(queries.size(), \
                                                                    std::vector<std::pair<uint32_t, size_t>> (args->get_nearest_neighbors_num()));

        std::vector<std::vector<uint32_t>>                     enn_distances(queries.size(), \
                                                                    std::vector<uint32_t> (args->get_nearest_neighbors_num()));

        std::vector<std::vector<size_t>>                       range_results(queries.size());
        std::vector<std::chrono::microseconds>                 ann_query_times(queries.size());
        std::vector<std::chrono::microseconds>                 enn_query_times(queries.size());

        for (size_t i = 0; i != queries.size(); ++i) {

            /* Approximate K-NN calculation */
            start = std::chrono::high_resolution_clock::now();
            ann_results[i] = lsh.approximate_k_nn(queries[i]);
            stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            ann_query_times[i] = duration;

            /* Exact NN calculation */
            start = std::chrono::high_resolution_clock::now();
            enn_distances[i] = exact_nn<uint8_t> (dataset, queries[i], args->get_nearest_neighbors_num());
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            enn_query_times[i] = duration;

            /* Range Search */
           range_results[i] = lsh.approximate_range_search(C, R, queries[i]);
        }

        print_statistics(args->get_nearest_neighbors_num(), queries.size(), ann_results, ann_query_times, \
                            enn_distances, enn_query_times);

        std::cout << "\nWriting formatted output to \"" << args->get_output_file_path() << "\"..."<< std::endl;
        write_output(args->get_output_file_path(), args->get_nearest_neighbors_num(), queries.size(), \
                                ann_results, ann_query_times, enn_distances, enn_query_times, range_results, "LSH");
        std::cout << "Done!" << std::endl;

        std::cout << "You can now open the output file and see its contents" << std::endl;
        std::string option = user_prompt_exit("\nDo you want to continue the simulation and repeat the search process?: [Y/N]: ");
        if (option == "Y") continue;
        else break;
        option = user_prompt_exit("\nDo you want to use the same query file?: [Y/N]: ");
        if (option != "Y") {
            args->set_query_file_path(user_prompt_file("\nEnter the path to the new query file: "));
        }
    }
}


int main (int argc, char *argv[]) {
    
    Lsh_args *args = nullptr;

    /* user executed the program using command line arguments */
    if (argc == 15) {
        lsh_parse_args(argc, argv, &args);
        start_lsh_simulation(args);
    }
    /* user executed the program with no command line arguments i.e ./lsh */
    else if (argc == 1) {
        user_interface(&args);
        start_lsh_simulation(args);
    }
    else lsh_usage(argv[0]);

    
    delete args;

    return EXIT_SUCCESS;
}
