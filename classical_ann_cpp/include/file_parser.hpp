#pragma once
#include <vector>
#include <string>
#include <cstdint>

using namespace std;

// Dataset Struct
template <typename T>
struct Dataset {
    vector<vector<T>> data;
    int dim;
};

// SearchResult Struct
struct SearchResult {
    vector<int> indices;
    vector<float> dists;
    vector<int> r_neighbors;
    double time = 0.0;
};

// Dataset readers
Dataset<uint8_t> read_mnist(const string& path);
Dataset<float> read_sift(const string& path);
Dataset<float> read_bio(const string& path, int dim = 320);

// Ground truth loader
vector<SearchResult> load_ground_truth(
    const string& filename,
    double& overall_tTrue,
    int max_queries = -1
);
