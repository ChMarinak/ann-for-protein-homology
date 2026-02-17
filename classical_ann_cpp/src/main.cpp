#include "utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    Config cfg;
    parse_arguments(argc, argv, cfg);

    if (cfg.dpath.empty() || cfg.outpath.empty()) {
        cerr << "Usage:\n"
             << "  ./search -d <input> -q <query> -o <output> -t <mnist|sift|bio>\n"
             << "  [ -lsh | -hypercube | -ivfflat | -ivfpq | -brute ] [params...]\n";
        return 1;
    }


    run_search(cfg);

    return 0;
}
