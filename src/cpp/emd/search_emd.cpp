#include <iostream>
#include <vector>

#include "../../include/emd/emd_utils.h"
#include "../../include/io_utils/cmd_args.h"
#include "../../include/io_utils/io_utils.h"


void earth_movers(Prj3_args *args) {
    std::vector<std::vector<uint8_t>> train_samples;
    std::vector<std::vector<uint8_t>> query_samples;
    std::vector<uint8_t> train_labels;
    std::vector<uint8_t> query_labels;

    /* load the 2 datasets (trainset, testset) and their labels respectively */
    std::cout << "\nReading training set from \"" << args->get_trainset_original_path() << "\"..." <<  std::endl;
    read_dataset<uint8_t> (args->get_trainset_original_path(), train_samples);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Train set has " << train_samples.size() << " samples" << std::endl;

    std::cout << "\nReading training set labels from \"" << args->get_train_labels_path() << "\"..." <<  std::endl;
    read_labels<uint8_t> (args->get_train_labels_path(), train_labels);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Train labels has " << train_labels.size() << " labels" << std::endl;

    std::cout << "\nReading query set from \"" << args->get_queryset_original_path() << "\"..." <<  std::endl;
    read_dataset<uint8_t> (args->get_queryset_original_path(), query_samples);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Query set has " << query_samples.size() << " samples" << std::endl;

    std::cout << "\nReading query set labels from \"" << args->get_query_labels_path() << "\"..." <<  std::endl;
    read_labels<uint8_t> (args->get_query_labels_path(), query_labels);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Query labels has " << query_labels.size() << " labels" << std::endl;

    /* for each query sample, find its 10 nns using emd */
    std::vector<size_t> nns(10, 0);
    size_t size = query_samples.size();
    for(size_t i = 0; i < 1; ++i) {
        emd(train_samples, query_samples[i], nns);

    }

}


int main(int argc, char *argv[]) {

    Prj3_args *args = nullptr;

    if (argc == 12) {
        emd_parse_args(argc, argv, &args);
        earth_movers(args);
    }
    else {
        emd_usage(argv[0]);
    }

    delete args;

    return EXIT_SUCCESS;
}
