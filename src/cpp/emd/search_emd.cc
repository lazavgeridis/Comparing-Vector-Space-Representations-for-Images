#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>

#include "../../../include/io_utils/cmd_args.h"
#include "../../../include/io_utils/io_utils.h"
#include "ortools/linear_solver/linear_solver.h"



void init_k_nearest(std::vector<std::pair<double, size_t>> &nns) {

    for(size_t i = 0; i != 10; ++i) 
        nns.emplace_back(std::numeric_limits<double>::max(), 0);

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


template<typename T>
uint32_t euclidean(T pixel1, T pixel2) {
    return (pixel1 - pixel2) * (pixel1 - pixel2);
}


template<typename T>
T center(const std::vector<T> &cluster) {
    size_t size = cluster.size();

    return cluster[ (size / 2) + 1 ];
}


template<typename T>
double cluster_weight(const std::vector<T> &image, const std::vector<T> &cluster) {
    uint32_t image_sum = 0;
    uint32_t cluster_sum = 0;

    for(auto i = image.cbegin(); i != image.cend(); ++i) {
        image_sum += *i;
    }
    for(auto i = cluster.cbegin(); i != cluster.cend(); ++i) {
        cluster_sum += *i;
    }

    return cluster_sum / (double)image_sum;
}


namespace operations_research {

template<typename T>
void emd(std::vector<std::vector<T>> &train_samples, std::vector<T> &query, std::vector<std::pair<double, size_t>> &nns) {

    /* make sure the dimensionality of the training set is equal to that of the query set */
    assert(train_samples[0].size() == query.size());

    /* get image resolution: dim x dim -> in our case, 28 x 28 */
    const uint32_t dimension_squared = query.size();
    const uint16_t dimension = (uint16_t) sqrt( (double) dimension_squared );
    //std::cout << "Each image has dimension " << dimension << "x" << dimension << std::endl;
    
    /* each cluster has 4x4 pixels or 7x7 pixels */
    const uint8_t pixels = 7; // should be faster
    
    /* each image consists of (dimension / pixels) x (dimension / pixels) clusters 
     * i.e 7x7 clusters or 4x4 clusters 
     */
    const uint8_t  clusters = dimension / pixels;
    const uint16_t n = clusters * clusters;
    
    std::cout << "In this case we have " << n << " clusters with " << pixels * pixels << " pixels each" << std::endl;

    uint32_t dist;
    double infinity;
    size_t query_offset, query_index, p_offset, p_index;
    std::vector<T> qcluster, pcluster;
    std::vector<double> q_cluster_weight(n, 0.0);
    std::vector<double> p_cluster_weight(n, 0.0);
    std::vector<MPConstraint *> row_constraints(n, 0);
    std::vector<MPConstraint *> column_constraints(n, 0);
    std::vector<uint32_t> distances;
    std::vector<MPVariable*> flows;
    T center1, center2;
    const size_t size = train_samples.size();

    init_k_nearest(nns);

    for (size_t i = 0; i != size; ++i) {

        MPSolver* solver = MPSolver::CreateSolver("GLOP");
        infinity = solver->infinity();

        /* create a variable vector representing flows - n squared variables total */
        solver->MakeNumVarArray(n * n, 0.0, infinity, "flow", &flows);

        const std::vector<T> &p = train_samples[i];

        for(size_t query_cluster_index = 0; query_cluster_index < n; ++query_cluster_index) {
            
            query_offset = (query_cluster_index / clusters) * dimension * pixels;
            query_offset += (query_cluster_index % clusters) * pixels;

            /* store pixels of query cluster */
            for(size_t i = 0; i < pixels; ++i) {
                for(size_t j = 0; j < pixels; ++j) {
                    query_index = query_offset + (i * dimension) + j;
                    qcluster.push_back(query[query_index]);
                }
            }

            /* (query): get cluster's "centroid" */
            center1 = center(qcluster);

            /* (query): calculate cluster's weight */
            q_cluster_weight[query_cluster_index] = cluster_weight(query, qcluster);

            for(size_t p_cluster_index = 0; p_cluster_index < n; ++p_cluster_index) {

                p_offset = (p_cluster_index / clusters) * dimension * pixels;
                p_offset += (p_cluster_index % clusters) * pixels;

                for(size_t i = 0; i < pixels; ++i) {
                    for(size_t j = 0; j < pixels; ++j) {
                        p_index = p_offset + (i * dimension) + j;
                        pcluster.push_back(p[p_index]);
                    }
                }

                center2 = center(pcluster);

                /* cluster's weight */
                p_cluster_weight[p_cluster_index] = cluster_weight(p, pcluster);

                /* Calculate distance between query_cluster_index (i) and p_cluster_index (j) */
                dist = euclidean(center1, center2);

                //std::cout << query_cluster_index << "-" << p_cluster_index \
                //    << " Distance: " << dist << std::endl;
                
                distances.push_back(dist);

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
        for(size_t i = 0; i < n * n; ++i) {
            objective->SetCoefficient(flows[i], distances[i]);
        }
        objective->SetMinimization();

        /* invoke solver */
        const MPSolver::ResultStatus result_status = solver->Solve();
        if (result_status != MPSolver::OPTIMAL) {
          LOG(FATAL) << "The problem does not have an optimal solution!";
        }
        //LOG(INFO) << "Solution:";
        //LOG(INFO) << "Optimal objective value = " << objective->Value();

        if (objective->Value() < nns[0].first) {
            nns[0] = std::make_pair(objective->Value(), i);
            std::sort(nns.begin(), nns.end(), [](const std::pair<double, size_t> &left, \
                                                 const std::pair<double, size_t> &right) \
                                                 { return (left.first > right.first); } );
        }
        
        distances.clear();
        flows.clear();

        delete solver;
        //std::cout << "Solver deallocated" << std::endl;
    }

    std::sort(nns.begin(), nns.end(), [](const std::pair<double, size_t> &left, \
                                         const std::pair<double, size_t> &right) \
                                        { return (left.first < right.first); } );
}
}   // namespace operations_research


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


double evaluate(uint8_t query_label, std::vector<uint8_t> &train_labels, std::vector<std::pair<double, size_t>> &nns) {
    uint8_t count = 0;
    for(size_t i = 0; i < nns.size(); ++i) {
        if (query_label == train_labels[nns[i].second]) 
            ++count;
    }

    return count / (double)nns.size();
}


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
    const size_t size = query_samples.size();
    double sum = 0.0;

    for(size_t i = 0; i < size; ++i) {

        std::vector<std::pair<double, size_t>> nns;
        operations_research::emd<uint8_t>(train_samples, query_samples[i], nns);
        //std::cout << "Query's label: " << +query_labels[i] << std::endl;

        //for (auto pair : nns)
        //    std::cout << "Distance: " << pair.first \
        //              << "    Index: " << pair.second \
        //              << "    Label: " << +train_labels[pair.second] << std::endl;

        sum += evaluate(query_labels[i], train_labels, nns);

        //std::cout << "Correct labels for query " << i << ": " << sum << std::endl;
        /* for each query compute:
         * percentage = # same label neighbors / 10
         * sum = sum + percentage
         */
    }
    std::cout << "Average correct EMD results = " << sum / (double)size << std::endl;
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
