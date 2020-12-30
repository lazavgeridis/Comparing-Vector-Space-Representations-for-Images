#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include <string>
#include <vector>
#include <utility>


typedef struct cluster_args {
    std::string input_file_original;
    std::string input_file_new;
    std::string nn_clusters_file;
    std::string config_file;
    std::string output_file;
} cluster_args;


typedef struct cluster_configs {
    int number_of_clusters;
} cluster_configs;


void cluster_usage(const char *);

void parse_cluster_args(int , char * const *, cluster_args *);

void parse_cluster_configurations(std::string, cluster_configs *);

void normalize_distances(std::vector<float> &);

size_t binary_search(const std::vector<std::pair<float, size_t>> &, float);

float find_max(const std::vector<float> &);

bool compare(const std::pair<float, size_t> &, const std::pair<float, size_t> &);

bool in(const std::vector<size_t> &, size_t);

#endif // CLUSTER_UTILS_H
