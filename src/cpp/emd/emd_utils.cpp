#include <iostream>
#include <unistd.h>

#include "../../../include/io_utils/cmd_args.h"
#include "../../../include/io_utils/io_utils.h"


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
