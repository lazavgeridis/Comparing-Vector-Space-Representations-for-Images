#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <utility>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>

#include "../../../include/io_utils/cmd_args.h"
#include "../../../include/io_utils/io_utils.h"
#include "../../../include/metric/metric.h"
#include "ortools/linear_solver/linear_solver.h"


void emd_usage(const char *exec) {
    fprintf(stderr, "\nUsage: %s \n"
                        "[+] -d    [train_file]\n"
                        "[+] -q    [query_file]\n"
                        "[+] -l1   [train_labels]\n"
                        "[+] -l2   [query_labels]\n"
                        "[+] -o    [output_file]\n"
                        "[+] -EMD\n"
                        "\nProvide all the above arguments\n",exec); 

    exit(EXIT_FAILURE);
}


bool file_exists(const char *filepath) {
    struct stat buf;

    return ( stat(filepath, &buf) == 0 );

}


void emd_parse_args(int argc, char * const argv[], Prj3_args **args) {
    int opt = 0;
    std::string train_file, query_file, train_labels, query_labels, output_file, labels, MD;

    while ( (opt = getopt(argc, argv, "d:q:l:o:E:")) != -1 ) {
        switch (opt) {
            case 'd':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Input file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                train_file = optarg;
                break;
                        
            case 'q':
                if ( !file_exists(optarg) ) {
                    std::cerr << "\n[+]Error: Query file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                query_file = optarg; 
                break;

            case 'l':
                if ( !file_exists(argv[optind]) ) {
                    std::cerr << "\n[+]Error: Input labels file does not exist!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }

                labels = optarg;

                if(labels == "1") {
                    train_labels = argv[optind];
                }
                else if (labels == "2") {
                    query_labels = argv[optind];
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

            case 'E':
                MD = optarg;
                if (MD != "MD") {
                    std::cerr << "\n[+]Error: Valid use of the option is -EMD!\n" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;

            default: 
                // one or more of the "-x" options did not appear
                emd_usage(argv[0]);
                break;
        }
    }

    *args = new Prj3_args(train_file, query_file, train_labels, query_labels, output_file); 
}


void write_output(const std::string &outpath, double emd_acc, double manhattan_acc) {
    std::ofstream ofile;
    ofile.open(outpath, std::ios::out | std::ios::trunc);

    ofile << "Average Correct Search Results EMD: " << emd_acc << std::endl;
    ofile << "Average Correct Search Results MANHATTAN: " << manhattan_acc << std::endl;
    ofile.close();
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


std::pair<uint8_t, uint8_t> center(std::pair<uint8_t, uint8_t> first_center_crds, uint8_t nclusters, uint8_t npixels, size_t index) {
    uint8_t x = first_center_crds.first  + (index / nclusters) * npixels;
    uint8_t y = first_center_crds.second + (index % nclusters) * npixels; 
    std::pair<uint8_t, uint8_t> c = {x, y};

    return c;
}


double euclidean(std::pair<uint8_t, uint8_t> center1, std::pair<uint8_t, uint8_t> center2) {
    uint16_t x_diff = (center1.first - center2.first) * (center1.first - center2.first);
    uint16_t y_diff = (center1.second - center2.second) * (center1.second - center2.second);

    return sqrt(x_diff + y_diff);
}


template <typename T>
void init_k_nearest(std::vector<std::pair<T, size_t>> &nns) {

    for(size_t i = 0; i != 10; ++i) 
        nns.emplace_back(std::numeric_limits<T>::max(), 0);
}


template<typename T>
uint32_t image_weight(const std::vector<T> &image) {
    uint32_t image_sum = 0;

    for(auto i = image.cbegin(); i != image.cend(); ++i) {
        image_sum += *i;
    }

    return image_sum;
}


template<typename T>
double cluster_weight(uint32_t image_weight, const std::vector<T> &cluster) {
    uint32_t cluster_sum = 0;

    for(auto i = cluster.cbegin(); i != cluster.cend(); ++i) {
        cluster_sum += *i;
    }

    return cluster_sum / (double)image_weight;
}


namespace operations_research {

template<typename T>
void emd(std::vector<std::vector<T>> &train_samples, std::vector<T> &query, std::vector<std::pair<double, size_t>> &nns) {

    /* make sure the dimensionality of the training set is equal to that of the query set */
    assert(train_samples[0].size() == query.size());

    /* get image resolution: dim x dim -> in our case, 28 x 28 */
    const uint32_t dimension_squared = query.size();
    const uint16_t dimension = (uint16_t) sqrt( (double) dimension_squared );
    
    /* each cluster has 4x4 pixels or 7x7 pixels */
    const uint8_t pixels = 7;
    const std::pair<uint8_t, uint8_t> center_crds = {pixels / 2, pixels / 2};
    
    /* each image consists of (dimension / pixels) x (dimension / pixels) clusters 
     * i.e 7x7 clusters or 4x4 clusters 
     */
    const uint8_t  clusters = dimension / pixels;
    const uint16_t n = clusters * clusters;
    
    double infinity;
    size_t query_offset, query_index, p_offset, p_index;
    std::vector<T> qcluster, pcluster;
    std::vector<double> q_cluster_weight(n, 0.0);
    std::vector<double> p_cluster_weight(n, 0.0);
    std::vector<MPConstraint *> row_constraints(n, 0);
    std::vector<MPConstraint *> column_constraints(n, 0);
    std::vector<MPVariable*> flows;
    std::vector<double> distances;
    std::pair<uint8_t, uint8_t> qcenter, pcenter;
    uint32_t point_weight;
    const size_t size = train_samples.size();
    const uint32_t query_weight = image_weight<T> (query);

    init_k_nearest<double> (nns);

    for (size_t i = 0; i != size; ++i) {

        MPSolver* solver = MPSolver::CreateSolver("GLOP");
        infinity = solver->infinity();

        /* create a variable vector representing flows - n squared variables total */
        solver->MakeNumVarArray(n * n, 0.0, infinity, "flow", &flows);
        const std::vector<T> &p = train_samples[i];
        point_weight = image_weight<T> (p);

        for(size_t query_cluster_index = 0; query_cluster_index != n; ++query_cluster_index) {
            
            query_offset = (query_cluster_index / clusters) * dimension * pixels;
            query_offset += (query_cluster_index % clusters) * pixels;

            /* store pixels of query cluster */
            for(size_t i = 0; i != pixels; ++i) {
                for(size_t j = 0; j != pixels; ++j) {
                    query_index = query_offset + (i * dimension) + j;
                    qcluster.push_back(query[query_index]);
                }
            }

            /* (query): get cluster's "centroid" coordinates */
            qcenter = center(center_crds, clusters, pixels, query_cluster_index);
            /* (query): calculate cluster's weight */
            q_cluster_weight[query_cluster_index] = cluster_weight<T> (query_weight, qcluster);

            for(size_t p_cluster_index = 0; p_cluster_index != n; ++p_cluster_index) {

                p_offset = (p_cluster_index / clusters) * dimension * pixels;
                p_offset += (p_cluster_index % clusters) * pixels;

                for(size_t i = 0; i != pixels; ++i) {
                    for(size_t j = 0; j != pixels; ++j) {
                        p_index = p_offset + (i * dimension) + j;
                        pcluster.push_back(p[p_index]);
                    }
                }

                pcenter = center(center_crds, clusters, pixels, p_cluster_index);
                p_cluster_weight[p_cluster_index] = cluster_weight<T> (point_weight, pcluster);

                /* Calculate distance between query_cluster_index (i) and p_cluster_index (j) */
                distances.push_back( euclidean(qcenter, pcenter) );

                pcluster.clear();
            }
            qcluster.clear();
        }

        /* define "row" constraints */
        for (size_t i = 0; i != n; ++i) {
            row_constraints[i] = solver->MakeRowConstraint(q_cluster_weight[i], q_cluster_weight[i]);
            for(size_t j = 0; j != n; ++j) {
                row_constraints[i]->SetCoefficient(flows[i * n + j], 1);
            }
        }
        /* define "column" constraints */
        for (size_t j = 0; j != n; ++j) {
            column_constraints[j] = solver->MakeRowConstraint(p_cluster_weight[j], p_cluster_weight[j]);
            for(size_t i = 0; i != n; ++i) {
                column_constraints[j]->SetCoefficient(flows[i * n + j], 1);
            }
        }

        //LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

        /* define objective function to minimize */
        MPObjective* const objective = solver->MutableObjective();
        for(size_t i = 0; i != n * n; ++i) {
            objective->SetCoefficient(flows[i], distances[i]);
        }
        objective->SetMinimization();

        /* invoke solver */
        const MPSolver::ResultStatus result_status = solver->Solve();
        if (result_status != MPSolver::OPTIMAL) {
          LOG(FATAL) << "The problem does not have an optimal solution!";
        }

        if (objective->Value() < nns[0].first) {
            nns[0] = std::make_pair(objective->Value(), i);
            std::sort(nns.begin(), nns.end(), [](const std::pair<double, size_t> &left, \
                                                 const std::pair<double, size_t> &right) \
                                                 { return (left.first > right.first); } );
        }
        
        distances.clear();
        flows.clear();

        delete solver;
    }

    std::sort(nns.begin(), nns.end(), [](const std::pair<double, size_t> &left, \
                                         const std::pair<double, size_t> &right) \
                                        { return (left.first < right.first); } );
}
}   // namespace operations_research


template<typename T>
void manhattan(std::vector<std::vector<T>> &train_samples, std::vector<T> &query, std::vector<std::pair<uint32_t, size_t>> &nns) {

    const size_t size = train_samples.size();
    uint32_t dist;
    init_k_nearest<uint32_t> (nns);

    for (size_t i = 0; i != size; ++i) {
        dist = manhattan_distance_rd<T> (train_samples[i], query);
        if (dist < nns[0].first) {
            nns[0] = std::make_pair(dist, i);
            std::sort(nns.begin(), nns.end(), [](const std::pair<uint32_t, size_t> &left, \
                                                 const std::pair<uint32_t, size_t> &right) \
                                                 { return (left.first > right.first); } );
        }
    }

    std::sort(nns.begin(), nns.end(), [](const std::pair<double, size_t> &left, \
                                         const std::pair<double, size_t> &right) \
                                        { return (left.first < right.first); } );
}


template <typename T>
double evaluate(uint8_t query_label, std::vector<uint8_t> &train_labels, std::vector<std::pair<T, size_t>> &nns) {
    uint8_t count = 0;
    for(size_t i = 0; i < nns.size(); ++i) {
        if (query_label == train_labels[nns[i].second]) 
            ++count;
    }

    return count / (double)nns.size();
}


template <typename T>
void load_data(Prj3_args *args, std::vector<std::vector<T>> &trainset, std::vector<uint8_t> &trainlabels, \
                std::vector<std::vector<T>> &queryset, std::vector<uint8_t> &querylabels) {

    /* load the 2 datasets (trainset, testset) and their labels respectively */
    std::cout << "\nReading training set from \"" << args->get_trainset_original_path() << "\" ..." <<  std::endl;
    read_dataset<T> (args->get_trainset_original_path(), trainset);
    std::cout << "Done!\n" << std::endl;

    std::cout << "\nReading training set labels from \"" << args->get_train_labels_path() << "\" ..." <<  std::endl;
    read_labels<uint8_t> (args->get_train_labels_path(), trainlabels);
    std::cout << "Done!\n" << std::endl;

    std::cout << "\nReading query set from \"" << args->get_queryset_original_path() << "\" ..." <<  std::endl;
    read_dataset<T> (args->get_queryset_original_path(), queryset);
    std::cout << "Done!\n" << std::endl;

    std::cout << "\nReading query set labels from \"" << args->get_query_labels_path() << "\" ..." <<  std::endl;
    read_labels<uint8_t> (args->get_query_labels_path(), querylabels);
    std::cout << "Done!\n" << std::endl;
}


void nn_search(Prj3_args *args) {
    std::vector<std::vector<uint8_t>> train_samples, query_samples;
    std::vector<uint8_t> train_labels, query_labels;

    load_data<uint8_t> (args, train_samples, train_labels, query_samples, query_labels);

    /* for each query, find its 10 nns using:
     * a. earh mover's distance
     * b. manhattan distance
     */
    const size_t size = query_samples.size() / 500; 
    double emd_sum = 0.0, manhattan_sum = 0.0, emd_dur = 0.0, manhattan_dur = 0.0;

    for(size_t i = 0; i != size; ++i) {
        std::vector<std::pair<double, size_t>> emd_nns;
        std::vector<std::pair<uint32_t, size_t>> manhattan_nns;

        auto start = std::chrono::high_resolution_clock::now();
        operations_research::emd<uint8_t> (train_samples, query_samples[i], emd_nns);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        emd_dur += elapsed.count();
        emd_sum += evaluate<double> (query_labels[i], train_labels, emd_nns);

        start = std::chrono::high_resolution_clock::now();
        manhattan<uint8_t> (train_samples, query_samples[i], manhattan_nns);
        stop = std::chrono::high_resolution_clock::now();
        elapsed = stop - start;
        manhattan_dur += elapsed.count();
        manhattan_sum += evaluate<uint32_t> (query_labels[i], train_labels, manhattan_nns);
    }

    std::cout << "NN search using EMD lasted " << emd_dur << "secs" << std::endl;
    std::cout << "NN search using MANHATTAN lasted " << manhattan_dur << "secs" << std::endl;
    write_output(args->get_output_file_path(), emd_sum / size, manhattan_sum / size);
}


int main(int argc, char *argv[]) {

    Prj3_args *args = nullptr;

    if (argc == 12) {
        emd_parse_args(argc, argv, &args);
        nn_search(args);
    }
    else {
        emd_usage(argv[0]);
    }

    delete args;

    return EXIT_SUCCESS;
}
