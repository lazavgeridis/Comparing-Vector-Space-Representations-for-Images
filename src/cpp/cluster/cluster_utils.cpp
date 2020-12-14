#include <iostream>
#include <string>
#include <utility>
#include <unistd.h>
#include <getopt.h>
#include <fstream>

#include "../../include/io_utils/io_utils.h"
#include "../../include/cluster/cluster_utils.h"


void cluster_usage(const char *exec) {
    fprintf(stderr, "\nUsage: %s \n\n"
                        "[+] -i [input_file]\n"
                        "[+] -c [configuration_file]\n"
                        "[+] -o [output_file]\n"
                        "[+] --complete [optional]\n"
                        "[+] -m [assignment method]\n"
                        "\nProvide all the above arguments\n", exec); 

    exit(EXIT_FAILURE);
}


void parse_cluster_args(int argc, char * const argv[], cluster_args *args) {
    
    int opt;
    std::string input, output, config;

    int option_index = 0;
    const struct option longopts[] = {
        {"complete", no_argument, 0, 'f'},
        {0, 0, 0, 0}
    };

    args->complete = false;
    while ((opt = getopt_long(argc, argv, "i:c:o:m:f", longopts, &option_index)) != -1) {
        switch(opt) {
            case 'i':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                args->input_file = optarg;
                break;
                        
            case 'c':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Configuration file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                args->config_file = optarg; 
                break;

            case 'o':
                // convention: if the output file does not exist, create one on the working directory
                if( file_exists(optarg) ) 
                    args->output_file = optarg;
                else {
                    std::ofstream out("./output");
                    args->output_file = "output";
                }
                break;

            case 'm':
                args->method = optarg;
                break;
            
            case 'f':
                args->complete = true;
                break;

            default: 
                // one or more of the "-x" options did not appear
                cluster_usage(argv[0]);
                break;
        }
    }
}


static void evaluate_configuration_values(cluster_configs *configs) {
    
    if (configs->number_of_hash_tables == 0) configs->number_of_hash_tables = 3;
    if (configs->number_of_hash_functions == 0) configs->number_of_hash_functions = 4;
    if (configs->max_number_M_hypercube == 0) configs->max_number_M_hypercube = 10;
    if (configs->hypercube_dimensions == 0) configs->hypercube_dimensions = 3;
    if (configs->number_of_probes == 0) configs->number_of_probes = 2;
}


void parse_cluster_configurations(std::string config_file, cluster_configs *configs) {

    std::string delimiter = ": ";
    std::string token;
    size_t pos = 0;

    std::ifstream file(config_file);
    std::string line;
    while (std::getline(file, line)) {
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            line.erase(0, pos + delimiter.length());
        }
        if (token == "number_of_clusters") {
            configs->number_of_clusters = stoi(line);
        }
        else if (token == "number_of_vector_hash_tables") {
            configs->number_of_hash_tables = stoi(line);
        }
        else if (token == "number_of_vector_hash_functions") {
            configs->number_of_hash_functions = stoi(line);
        }
        else if (token == "max_number_M_hypercube") {
            configs->max_number_M_hypercube = stoi(line);
        }
        else if (token == "number_of_hypercube_dimensions") {
            configs->hypercube_dimensions = stoi(line);
        }
        else if (token == "number_of_probes") {
            configs->number_of_probes = stoi(line);
        }
    }

    evaluate_configuration_values(configs); 
}


size_t binary_search(const std::vector<std::pair<float, size_t>> &partial_sums, float val)
{

    size_t middle = 0, begin = 0, end = partial_sums.size();
    const std::pair<float, size_t> *less_than = &partial_sums[0];
    //const std::pair<float, size_t> *greater_than = &partial_sums[0];

    while (begin <= end) {

        middle = begin + (end - begin) / 2;
        if (val == partial_sums[middle].first) {
            return partial_sums[middle].second;
        }
        if(val < partial_sums[middle].first) {
            less_than = &partial_sums[middle];
            end = middle - 1;

        }
        else {
            //greater_than = &partial_sums[middle];
            begin = middle + 1;

        }
    }

    //std::cout << "P(r-1) = " << greater_than->first << " < " << val << " <= P(r) = " << less_than->first << std::endl;

    return less_than->second;
}


float find_max(const std::vector<float> &min_distances)
{
    float max_dist = std::numeric_limits<float>::min();

    for (float dist : min_distances) {
        if (dist > max_dist) max_dist = dist;
    }

    return max_dist;
}


void normalize_distances(std::vector<float> &min_distances)
{
    float dmax = find_max(min_distances);

    for (float &d : min_distances)
        d /= dmax;
}


bool in(const std::vector<size_t> &centroid_indexes, size_t index)
{
    for (size_t j = 0; j != centroid_indexes.size(); ++j) {
        if (centroid_indexes[j] == index)
            return true;
    }

    return false;
}


bool compare(const std::pair<float, size_t> &p1, const std::pair<float, size_t> &p2) 
{
    return p1.first < p2.first;
}
