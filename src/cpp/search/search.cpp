#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <sys/time.h>


#include "../../../include/io_utils/io_utils.h"
#include "../../../include/modules/lsh/lsh.h"
#include "../../../include/modules/exact_nn/exact_nn.h"
#include "../../../include/metric/metric.h"


static void start_search_simulation(search_cmd_args *args) {

    uint32_t N = 1;
    std::vector<std::vector<uint8_t>> dataset_initial, queryset_initial;
    std::vector<std::vector<uint16_t>> dataset_reduced, queryset_reduced;

    std::cout << "\nReading input files and query files (initial space and reduced space)..." << std::endl;
    read_dataset<uint8_t> (args->input_file_initial, dataset_initial);
    read_dataset<uint16_t> (args->input_file_reduced, dataset_reduced);
    read_dataset<uint8_t> (args->query_file_initial, queryset_initial);
    read_dataset<uint16_t> (args->query_file_reduced, queryset_reduced);
    std::cout << "Done!" << std::endl;


    std::cout << "\nComputing mean nearest neighbor distance..." << std::endl;
    double r = mean_nn_distance<uint8_t> (dataset_initial); 
    std::cout << "Done!" << std::endl;


    std::cout << "\nCreating LSH structure..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    LSH<uint8_t> lsh = LSH<uint8_t> (args->L, 1, args->k, r, dataset_initial);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start); 
    std::cout << "LSH Structure Creation Lasted " << duration.count() << " seconds" << std::endl;


    std::cout << "\nStart Executing ANN / ENN" << std::endl;
    std::cout << "..." << std::endl;

    /********** Start ANN / ENN / Range search **********/
    std::vector<std::vector<std::pair<uint32_t, size_t>>>   ann_results(queryset_initial.size(), \
                                                                std::vector<std::pair<uint32_t, size_t>> (N));

    std::vector<std::pair<uint32_t, size_t>>                enn_distances(queryset_initial.size());
                                                                
    std::vector<std::pair<uint32_t, size_t>>                enn_reduced_distances(queryset_reduced.size());

    std::vector<std::chrono::microseconds>                  search_times(3);


    /* Exact NN calculation in reduced space */
    for (size_t i = 0; i != queryset_reduced.size(); ++i) {
        start = std::chrono::high_resolution_clock::now();
        enn_reduced_distances[i] = search_exact_nn<uint16_t> (dataset_reduced, queryset_reduced[i]);
        stop = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //enn_reduced_distances[i].first = manhattan_distance_rd<uint8_t> (queryset_initial[i], dataset_initial[enn_reduced_distances[i].second]);
        search_times[0] += dur;
    }


    /* Approximate K-NN calculation */
    for (size_t i = 0; i != queryset_initial.size(); ++i) {
        start = std::chrono::high_resolution_clock::now();
        ann_results[i] = lsh.approximate_k_nn(queryset_initial[i]);
        stop = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        search_times[1] += dur;
    }


    /* Exact NN calculation */
    for (size_t i = 0; i != queryset_initial.size(); ++i) {
        start = std::chrono::high_resolution_clock::now();
        enn_distances[i] = search_exact_nn<uint8_t> (dataset_initial, queryset_initial[i]);
        stop = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        search_times[2] += dur;
    }
    

    std::cout << "\nWriting formatted output to \"" << args->output_file << "\"..."<< std::endl;
    write_search_output(args->output_file, queryset_initial.size(), \
                            ann_results, enn_distances, enn_reduced_distances, search_times);
    std::cout << "Done!" << std::endl;

}



int main(int argc, char *argv[]) {

    search_cmd_args args;

    if (argc != 15) search_usage(argv[0]);

    search_parse_args(argc, argv, &args);
    start_search_simulation(&args);


    return EXIT_SUCCESS;
}
