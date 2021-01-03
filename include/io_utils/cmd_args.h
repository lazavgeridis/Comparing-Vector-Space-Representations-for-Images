#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>


typedef struct search_cmd_args {
    int k;
    int L;
    std::string input_file_initial;
    std::string input_file_reduced;
    std::string query_file_initial;
    std::string query_file_reduced;
    std::string output_file;
} search_cmd_args;


class Prog_args // abstract class
{
    private:
        const std::string input_file_path;
        std::string query_file_path;
        const std::string output_file_path;
        uint16_t nearest_neighbors_num = 0;
        float radius = 0.0;

    public:
        Prog_args(const std::string &, const std::string &, const std::string &, uint16_t, float);
        virtual ~Prog_args() = default;
        void set_query_file_path(const std::string &);
        void set_nearest_neighbors_num(uint16_t);
        void set_radius(float);
        std::string get_input_file_path() const;
        std::string get_query_file_path() const;
        std::string get_output_file_path() const;
        uint16_t get_nearest_neighbors_num() const;
        float get_radius() const;
        
        virtual uint32_t get_k() const = 0; // pure virtual function
};


class Lsh_args: public Prog_args
{
    private:
        uint32_t hash_functions_num = 0; // k
        uint16_t hash_tables_num = 0;   // L

    public:
        Lsh_args(const std::string &, std::string &, std::string &, uint16_t, float, uint32_t, uint16_t);
        Lsh_args(const std::string &, const std::string &, const std::string &);
        /* Constructor with default values */
        //Lsh_args(const string &, string &, string &);
        void set_hash_functions_num(uint16_t);
        void set_hash_tables_num(uint16_t);
        uint32_t get_k() const;
        uint16_t get_hash_tables_num() const;
};


class Prj3_args
{
    private:
        const std::string trainset_original_path;
        const std::string queryset_original_path;
        const std::string train_labels_path;
        const std::string query_labels_path;
        const std::string output_file_path;

    public:
        Prj3_args(const std::string &, const std::string &, const std::string &, const std::string &, const std::string &);
        std::string get_trainset_original_path() const;
        std::string get_queryset_original_path() const;
        std::string get_train_labels_path() const;
        std::string get_query_labels_path() const;
        std::string get_output_file_path() const;
};


inline Prog_args::Prog_args::Prog_args(const std::string &ipath, const std::string &qpath, const std::string &opath, \
                                        uint16_t nn_num, float rad) : \
                                        input_file_path(ipath), query_file_path(qpath), output_file_path(opath), \
                                        nearest_neighbors_num(nn_num), radius(rad)
{ }

inline void Prog_args::set_query_file_path(const std::string &qfile)
{
    query_file_path = qfile;
}

inline void Prog_args::set_nearest_neighbors_num(uint16_t nns)
{
    nearest_neighbors_num = nns;
}

inline void Prog_args::set_radius(float rad)
{
    radius = rad;
}

inline std::string Prog_args::get_input_file_path() const
{
    return input_file_path;
}

inline std::string Prog_args::get_query_file_path() const
{
    return query_file_path;
}

inline std::string Prog_args::get_output_file_path() const
{
    return output_file_path;
}

inline uint16_t Prog_args::get_nearest_neighbors_num() const
{
    return nearest_neighbors_num;
}

inline float Prog_args::get_radius() const
{
    return radius;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline Lsh_args::Lsh_args(const std::string &ipath, std::string &qpath, std::string &opath, \
                            uint16_t nn_num, float rad, uint32_t hfunc_num, uint16_t htabl_num) : \
                        Prog_args(ipath, qpath, opath, nn_num, rad), hash_functions_num(hfunc_num), \
                        hash_tables_num(htabl_num)
{}

inline Lsh_args::Lsh_args(const std::string &ipath, const std::string &qpath, const std::string &opath)
    : Prog_args(ipath, qpath, opath, 1, 10000.0), hash_functions_num(4), hash_tables_num(5)

{}


inline void Lsh_args::set_hash_functions_num(uint16_t hfunc_num)
{
    hash_functions_num = hfunc_num;
}


inline void Lsh_args::set_hash_tables_num(uint16_t htabl_num)
{
    hash_tables_num = htabl_num;
}


inline uint32_t Lsh_args::get_k() const
{
    return hash_functions_num;
}


inline uint16_t Lsh_args::get_hash_tables_num() const
{
    return hash_tables_num;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline Prj3_args::Prj3_args(const std::string &train_file, const std::string &query_file, \
                            const std::string &train_labels, const std::string &query_labels, \
                            const std::string &output_file) : \
                                                trainset_original_path(train_file), \
                                                queryset_original_path(query_file), \
                                                train_labels_path(train_labels), \
                                                query_labels_path(query_labels), \
                                                output_file_path(output_file)
{}


inline std::string Prj3_args::get_trainset_original_path() const
{
    return trainset_original_path;
}


inline std::string Prj3_args::get_queryset_original_path() const
{
    return queryset_original_path;
}


inline std::string Prj3_args::get_train_labels_path() const
{
    return train_labels_path;
}


inline std::string Prj3_args::get_query_labels_path() const
{
    return query_labels_path;
}


inline std::string Prj3_args::get_output_file_path() const
{
    return output_file_path;
}
        

#endif
