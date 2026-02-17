#pragma once

#include <string>
#include <vector>
#include <cstdint>

using namespace std;

// Forward declarations
template <typename T>
struct Dataset;

struct SearchResult;


// Configuration Struct
struct Config {
    // General
    string algo = "brute";
    string type = "mnist";
    string dpath, qpath, outpath, evalpath;
    bool range_search = false;

    // Common
    int N = 1;
    float R = 2000.0f;
    int seed = 1;
    int max_queries = -1;

    // LSH
    int k = 4;
    int L = 5;
    double w = 4.0;

    // Hypercube
    int kproj = 14;
    int M = 10;
    int probes = 2;

    // IVFFlat
    int kclusters = 50;
    int nprobe = 5;

    // IVFPQ
    int nbits = 8;
};


// Public API
void parse_arguments(int argc, char** argv, Config& cfg);

void run_search(const Config& cfg);


// Explicit template instantiations
template <typename T>
void run_algorithm(const Dataset<T>& data,
                   const Dataset<T>& queries,
                   const Config& cfg);
