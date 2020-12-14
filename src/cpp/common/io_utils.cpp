#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <utility>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>

#include "../../include/io_utils/cmd_args.h"


void lsh_usage(const char *exec) {
    fprintf(stderr, "\nUsage: %s \n"
                        "[+] -d [input_file]\n"
                        "[+] -q [query_file]\n"
                        "[+] -k [hash_functions_number]\n"
                        "[+] -L hash_tables_number\n"
                        "[+] -o [output_file]\n"
                        "[+] -N [nearest_neighbors_number]\n"
                        "[+] -R [radius]\n"
                        "\nProvide all the above arguments\n",exec); 

    exit(EXIT_FAILURE);
}


void cube_usage(const char *exec) {
    fprintf(stderr, "\nUsage: %s \n"
                        "[+] -d [input_file]\n"
                        "[+] -q [query_file]\n"
                        "[+] -k [projection_dimension]\n"
                        "[+] -M [max_candidates]\n"
                        "[+] -probes [max_probes]\n"
                        "[+] -o [output_file]\n"
                        "[+] -N [nearest_neighbors_number]\n"
                        "[+] -R [radius]\n"
                        "\nProvide all the above arguments\n",exec); 

    exit(EXIT_FAILURE);
}


bool file_exists(const char *filepath) {
    struct stat buf;

    return ( stat(filepath, &buf) == 0 );

}


std::string user_prompt_exit(const std::string &message) {

    std::string exit;

    std::cout << message ;
    std::cin >> exit;

    return exit;
}


std::string user_prompt_file(const std::string &message) {
    std::string file_path;

    std::cout << message ;
    std::cin >> file_path; 
    if ( !file_exists(file_path.c_str()) ) {
        std::cerr << "\nFile does not exist!" << std::endl;
        exit(EXIT_FAILURE);
    }

    return file_path;
}


size_t user_prompt_query_index(const std::string &message, long lower, long upper)
{
    long index = 0;

    std::cout << message;
    std::cin >> index;
    if (index < lower || index > upper) {
        std::cerr << "\nQuery index is out of bounds!" << std::endl;
        exit(EXIT_FAILURE);
    }

    return index;
}


void lsh_parse_args(int argc, char * const argv[], Lsh_args **args) {
    
    int opt{};
    uint32_t hfunc_num{};
    uint16_t htabl_num{}, nn_num{};
    std::string dataset_file, query_file, output_file;
    float rad{};

    while ( (opt = getopt(argc, argv, "d:q:k:L:o:N:R:")) != -1 ) {
        switch (opt) {
            case 'd':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                dataset_file = optarg;
                break;
                        
            case 'q':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Query file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                query_file = optarg; 
                break;

            case 'k':
                hfunc_num = atoi(optarg);
                if (hfunc_num < 1) {
                    std::cerr << "\n[+]Error: -k must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'L':
                htabl_num = atoi(optarg);
                if (htabl_num < 1) {
                    std::cerr << "\n[+]Error: -L muste be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'o':
                // convention: if the output file does not exist, create one on the working directory
                if( file_exists(optarg) ) 
                    output_file = optarg;
                else {
                    std::ofstream out(optarg);
                    output_file = optarg;
                }
                break;

            case 'N':
                nn_num = atoi(optarg);
                if (nn_num < 1) {
                    std::cerr << "\n[+]Error: -N must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'R':
                rad = atof(optarg);
                break;

            default: 
                // one or more of the "-x" options did not appear
                lsh_usage(argv[0]);
                break;
        }
    }
    *args = new Lsh_args(dataset_file, query_file, output_file, nn_num, rad, hfunc_num, htabl_num); 
}


void cube_parse_args(int argc, char * const argv[], Cube_args **args) {
    int opt = 0, projection_dimension = 0, max_candidates = 0, max_probes = 0, nn_num = 0;
    float rad = 0.0;
    std::string dataset_file, query_file, output_file;

    while ( (opt = getopt(argc, argv, "d:q:k:M:p:o:N:R:")) != -1 ) {
        switch (opt) {
            case 'd':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                dataset_file = optarg;
                break;
                        
            case 'q':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Query file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                query_file = optarg; 
                break;

            case 'k':
                projection_dimension = atoi(optarg);
                if (projection_dimension < 1) {
                    std::cerr << "\n[+]Error: -k must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'M':
                max_candidates = atoi(optarg);
                if (max_candidates < 1) {
                    std::cerr << "\n[+]Error: -M must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'p':
                max_probes = atoi(argv[optind]);
                if (max_probes < 1) {
                    std::cerr << "\n[+]Error: -probes must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'o':
                // convention: if the output file does not exist, create one on the working directory
                if( file_exists(optarg) ) 
                    output_file = optarg;
                else {
                    std::ofstream out(optarg);
                    output_file = optarg;
                }
                break;

            case 'N':
                nn_num = atoi(optarg);
                if (nn_num < 1) {
                    std::cerr << "\n[+]Error: -N must be >= 1\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            case 'R':
                rad = atof(optarg);
                break;

            default: 
                // one or more of the "-x" options did not appear
                cube_usage(argv[0]);
                break;
        }
    }

    *args = new Cube_args(dataset_file, query_file, output_file, nn_num, rad, projection_dimension, max_candidates, max_probes); 
}


void user_interface(Cube_args **args) {

    std::string input_file, query_file, output_file; 

    if (*args == nullptr) {
        input_file = user_prompt_file("Enter path to input file: ");
        query_file = user_prompt_file("Enter path to query file: ");
        output_file = user_prompt_file("Enter path to output file: ");

        *args = new Cube_args(input_file, query_file, output_file);
    }
}

void user_interface(Lsh_args **args) {

    std::string input_file, query_file, output_file; 

    if (*args == nullptr) {
        input_file = user_prompt_file("Enter path to input file: ");
        query_file = user_prompt_file("Enter path to query file: ");
        output_file = user_prompt_file("Enter path to output file: ");

        *args = new Lsh_args(input_file, query_file, output_file);
    }
}


uint32_t bigend_to_littlend(uint32_t big_endian) {
    uint32_t b0, b1, b2, b3;
    uint32_t little_endian;

    b0 = (big_endian & 0x000000ff) << 24U;
    b1 = (big_endian & 0x0000ff00) << 8U;
    b2 = (big_endian & 0x00ff0000) >> 8U;
    b3 = (big_endian & 0xff000000) >> 24U;

    little_endian = b0 | b1 | b2 | b3;

    return little_endian;
}



void print_statistics(const uint16_t nns, const size_t size, const std::vector<std::vector<std::pair<uint32_t, size_t>>> &ann_res, \
                            const std::vector<std::chrono::microseconds> &ann_query_times, \
                            const std::vector<std::vector<uint32_t>> &enn_dists, \
                            const std::vector<std::chrono::microseconds> &enn_query_times) {

    std::vector<std::pair<uint32_t, size_t>> approx_nearest;
    std::vector<uint32_t> exact_nearest;
    std::vector<double> approx_factor;

    size_t wrong_dists{};
    size_t not_found{};


    for (size_t i = 0; i != size; ++i) {
        approx_nearest = ann_res[i];
        exact_nearest  = enn_dists[i];
        for (size_t j = 0; j != nns; ++j) {
            uint32_t dist = approx_nearest[j].first;
            if (dist == std::numeric_limits<uint32_t>::max()) {
                // that means that we didnt find any neighbor (nearest neighbor-1)
                if (j == 0) ++not_found;
            }
            else {
                if 
                    (dist < exact_nearest[j]) wrong_dists++;
                else
                    approx_factor.emplace_back((double) (dist / exact_nearest[j]));
            }
        }
    }
    std::cout << "\t\tPRINTING STATISTICS" << std::endl;
    std::cout << "\nWrong Distances (distanceLSH < distanceTrue): " << wrong_dists << std::endl;
    std::cout << "Not Found: " << not_found << std::endl;

    double mean_af{};
    double max_af = std::numeric_limits<double>::min();

    for (auto const &af: approx_factor) {
        if (af > max_af) max_af = af;
        mean_af += af;
    }
    std::cout << "\nMax-Approximation-Factor: " << (double) max_af << std::endl;
    std::cout << "Mean-Approximation-Factor: " << mean_af / approx_factor.size() << std::endl;

    size_t lsh_mean_time{};
    for (auto const &time: ann_query_times) {
        lsh_mean_time += time.count();
    }
    std::cout << "Mean-Time-Search-LSH: " << lsh_mean_time / size << std::endl;

    size_t exact_mean_time{};
    for (auto const &time: enn_query_times) {
        exact_mean_time += time.count();
    }
    std::cout << "Mean-Time-Search-Exact: " << exact_mean_time / size << std::endl;
}



void write_output(const std::string &out, const uint16_t nns, const size_t size, \
                            const std::vector<std::vector<std::pair<uint32_t, size_t>>> &ann_res, \
                            const std::vector<std::chrono::microseconds> &ann_query_times, \
                            const std::vector<std::vector<uint32_t>> &enn_dists, const std::vector<std::chrono::microseconds> &enn_query_times, \
                            const std::vector<std::vector<size_t>> &range_res, const std::string &structure) {
    
    std::vector<std::pair<uint32_t, size_t>> approx_nearest;
    std::vector<uint32_t> exact_nearest;
    std::ofstream ofile;
    ofile.open(out, std::ios::out | std::ios::trunc);

    for (size_t i = 0; i != size; ++i) {
        approx_nearest = ann_res[i];
        exact_nearest  = enn_dists[i];
        ofile << "Query: " << i << std::endl;
        for (size_t j = 0; j != nns; ++j) {
            uint32_t dist = approx_nearest[j].first;
            size_t ith_vec = approx_nearest[j].second;
            if (dist == std::numeric_limits<uint32_t>::max()) {
                // that means that we didnt find any neighbor (nearest neighbor-1)
                ofile << "Nearest neighbor-" << j + 1 << ": " << "Not Found" << std::endl;
                ofile << "distance" << structure << ": " << "None" << std::endl;
            }
            else {
                ofile << "Nearest neighbor-" << j + 1 << ": " << ith_vec << std::endl;
                ofile << "distance" << structure << ": " << dist << std::endl;
            }
            ofile << "distanceTrue: " << exact_nearest[j] << std::endl;
        }

        ofile << "t" << structure << ": " << ann_query_times[i].count() << std::endl;
        ofile << "tTrue: " << (double) enn_query_times[i].count() << std::endl;

        ofile << "R-near neighbors:" << std::endl;
        if (range_res[i].empty()) {
            ofile << "Not Found" << std::endl;
            continue;
        }
        for (auto &c : range_res[i]) {
            ofile << c << std::endl;
        }
    }
    ofile.close();
}
