#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sys/time.h>

#include "../../include/modules/hypercube/hypercube.h" 
#include "../../include/io_utils/cmd_args.h"
#include "../../include/io_utils/io_utils.h"
#include "../../include/modules/exact_nn/exact_nn.h"


static void start_hypercube_simulation(Cube_args *args)
{
    /* read training set and store each data sample */
    std::vector<std::vector<uint8_t>> training_samples;
    std::cout << "\nReading training set from \"" << args->get_input_file_path() << "\"..." << std::endl;
    read_file<uint8_t> (args->get_input_file_path(), training_samples);
    std::cout << "Done!" << std::endl;

    std::cout << "\nComputing mean nearest neighbor distance..." << std::endl;
    double r = mean_nn_distance<uint8_t> (training_samples);  
    std::cout << "Done!" << std::endl;

    std::cout << "\nCreating Hypercube structure..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Hypercube<uint8_t> cube(args->get_k(), args->get_max_candidates(), args->get_max_probes(), \
                                                    args->get_nearest_neighbors_num(), args->get_radius(), training_samples.size(), \
                                                    training_samples[0].size(), r, training_samples);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Done!" << std::endl;
    std::cout << "Hypercube Structure Creation lasted " << duration.count() << " seconds" << std::endl;

    while (1) {
        /* read test set and store each query point */
        std::vector<std::vector<uint8_t>> test_samples;
        std::cout << "\nReading test set from \"" << args->get_query_file_path() << "\"..." << std::endl;
        read_file<uint8_t> (args->get_query_file_path(), test_samples);
        std::cout << "Done!" << std::endl;

        std::cout << "\nQuery file contains " << test_samples.size() << " queries" << std::endl;
        std::cout << "\nStart Executing ANN / ENN / Range-Search" << std::endl;
        std::cout << "..." << std::endl;
        /********** Start ANN / ENN / Range search **********/
        std::vector<std::vector<std::pair<uint32_t, size_t>>> ann_results(test_samples.size(), \
                                                                std::vector<std::pair<uint32_t, size_t>> (args->get_nearest_neighbors_num()));

        std::vector<std::vector<uint32_t>>                    enn_distances(test_samples.size(), \
                                                                    std::vector<uint32_t> (args->get_nearest_neighbors_num()));

        std::vector<std::vector<size_t>>                      range_results(test_samples.size());
        std::vector<std::chrono::microseconds>                ann_query_times(test_samples.size());
        std::vector<std::chrono::microseconds>                enn_query_times(test_samples.size());

        for (size_t i = 0; i != test_samples.size(); ++i) {

            /* Approximate K-NN calculation */
            start = std::chrono::high_resolution_clock::now();
            ann_results[i] = cube.approximate_nn(test_samples[i], training_samples);
            stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            ann_query_times[i] = duration;

            /* Exact NN calculation */
            start = std::chrono::high_resolution_clock::now();
            enn_distances[i] = exact_nn<uint8_t> (training_samples, test_samples[i], args->get_nearest_neighbors_num());
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            enn_query_times[i] = duration;

            /* Range Search */
            range_results[i] = cube.range_search(test_samples[i], training_samples);
        }

        print_statistics(args->get_nearest_neighbors_num(), test_samples.size(), ann_results, ann_query_times, \
                            enn_distances, enn_query_times);

        std::cout << "\nWriting formatted output to \"" << args->get_output_file_path() << "\"..."<< std::endl;
        write_output(args->get_output_file_path(), args->get_nearest_neighbors_num(), test_samples.size(),
                                ann_results, ann_query_times, enn_distances, enn_query_times, range_results, "Hypercube");
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


int main(int argc, char *argv[])
{

    Cube_args *args = nullptr;

    if (argc == 17) {
        cube_parse_args(argc, argv, &args);     // user executed the program with command line options
        start_hypercube_simulation(args);
    }
    else if (argc == 1) {                       // user executed the program without command line options i.e ./cube
        user_interface(&args);
        start_hypercube_simulation(args);
    }
    else {                                      // invalid execution
        cube_usage(argv[0]);
    }

    delete args;

    return EXIT_SUCCESS;
}
