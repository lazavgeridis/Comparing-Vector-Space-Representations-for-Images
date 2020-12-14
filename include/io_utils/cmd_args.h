#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>


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


class Cube_args: public Prog_args
{
    private:
        const uint32_t projection_dim = 0; // d' is specified as the -k option in the command line
        const uint16_t max_candidates = 0; // M
        const uint16_t max_probes = 0;    // probes

    public:
        Cube_args(const std::string &, const std::string &, const std::string &, uint16_t, float, uint32_t, uint16_t, uint16_t);
        Cube_args(const std::string &, const std::string &, const std::string &);
        uint32_t get_k() const;
        uint16_t get_max_candidates() const;
        uint16_t get_max_probes() const;
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

inline Cube_args::Cube_args(const std::string &ipath, const std::string &qpath, const std::string &opath, uint16_t nn_num, float rad, \
                            uint32_t proj, uint16_t max_cands, uint16_t max_probes) : \
        Prog_args(ipath, qpath, opath, nn_num, rad), projection_dim(proj), max_candidates(max_cands), max_probes(max_probes)  
{}


inline Cube_args::Cube_args(const std::string &ipath, const std::string &qpath, const std::string &opath) : \
        Prog_args(ipath, qpath, opath, 1, 10000.0), projection_dim(14), max_candidates(10), max_probes(2)
{}


inline uint32_t Cube_args::get_k() const
{
    return projection_dim;
}


inline uint16_t Cube_args::get_max_candidates() const
{
    return max_candidates;
}


inline uint16_t Cube_args::get_max_probes() const
{
    return max_probes;
}

#endif
