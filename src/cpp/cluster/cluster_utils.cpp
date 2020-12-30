#include <iostream>
#include <string>
#include <utility>
#include <unistd.h>
#include <getopt.h>
#include <fstream>

#include "../../../include/io_utils/io_utils.h"
#include "../../../include/cluster/cluster_utils.h"


void cluster_usage(const char *exec) {
    fprintf(stderr, "\nUsage: %s \n\n"
                        "[+] -d [input file original space]\n"
                        "[+] -i [input file new space]\n"
                        "[+] -n [classes from Neural Network as clusters file]\n"
                        "[+] -c [configuration file]\n"
                        "[+] -o [output file]\n"
                        "\nProvide all the above arguments\n", exec); 

    exit(EXIT_FAILURE);
}


void parse_cluster_args(int argc, char * const argv[], cluster_args *args) {
    
    int opt;
    std::string input_original, input_new, nn_clusters, config, output;

    while ((opt = getopt(argc, argv, "d:i:n:c:o")) != -1) {
        switch(opt) {
            case 'd':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file (original space) does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                args->input_file_original = optarg;
                break;
                        
            case 'i':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file (new space) does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                args->input_file_new = optarg;
                break;

            case 'n':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: NN clusters file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                args->nn_clusters_file = optarg;
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

            default: 
                // one or more of the "-x" options did not appear
                cluster_usage(argv[0]);
                break;
        }
    }
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
    }
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
