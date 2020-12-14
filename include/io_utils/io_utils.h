#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include "cmd_args.h"

std::string user_prompt_exit(const std::string &);

std::string user_prompt_file(const std::string&);

size_t user_prompt_query_index(const std::string&, long, long);

uint32_t bigend_to_littlend(uint32_t);

bool file_exists(const char *);

void lsh_parse_args(int, char * const [], Lsh_args **);

void cube_parse_args(int, char * const [], Cube_args **);

void lsh_usage(const char *);

void cube_usage(const char *);

void user_interface(Cube_args **);

void user_interface(Lsh_args **);

void print_statistics(const uint16_t , const size_t , 
                            const std::vector<std::vector<std::pair<uint32_t, size_t>>> &, \
                            const std::vector<std::chrono::microseconds> &, \
                            const std::vector<std::vector<uint32_t>> &, 
                            const std::vector<std::chrono::microseconds> &);
                            

void write_output(const std::string &, const uint16_t , const size_t , \
                        const std::vector<std::vector<std::pair<uint32_t, size_t>>> &, \
                        const std::vector<std::chrono::microseconds> &, const std::vector<std::vector<uint32_t>> &, \
                        const std::vector<std::chrono::microseconds> &, const std::vector<std::vector<size_t>> &, \
                        const std::string &);


template <typename T>
void read_file(const std::string& filepath, std::vector<std::vector<T>> &data) {
    std::ifstream data_file;
    data_file.open(filepath, std::ios::in | std::ios::binary);

    if(data_file) {
        uint32_t magic_num, images_number, rows, cols;

        // read magic number
        data_file.read( (char *) &magic_num, sizeof(magic_num) );
        magic_num = bigend_to_littlend(magic_num);

        // read number of images
        data_file.read( (char *) &images_number, sizeof(images_number) );
        images_number = bigend_to_littlend(images_number);

        // read number of rows
        data_file.read( (char *) &rows, sizeof(rows) );
        rows = bigend_to_littlend(rows);

        // read number of columns
        data_file.read( (char *) &cols, sizeof(cols) );
        cols = bigend_to_littlend(cols);

        /* images_number images, and each image is of dimension rows * columns
         * each pixel takes values from [0, 255]
         * "flatten" and store i-th image to the i-th element of the vector
         */
        data.resize(images_number, std::vector<T>(rows * cols, 0));
        for(size_t i = 0; i != images_number; ++i) {
            std::vector<T> &vec = data[i];
            for(size_t j = 0; j != rows * cols; ++j) {
                data_file.read( (char *) &vec[j], sizeof(vec[j]) );
            }
        }

        data_file.close();
    }
    else {
        std::cerr << "Could not open the file!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
