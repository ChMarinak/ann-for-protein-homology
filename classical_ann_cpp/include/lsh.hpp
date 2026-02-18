#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include "utils.hpp"

using namespace std;

template <typename T>
class LSH {
public:
    // Constructor
    LSH(int dim_, int L_ = 5, int k_ = 4, float w_ = 4.0f, unsigned int seed_ = 1);

    // Build functions
    void build(const vector<vector<T>>& dataset);

    // Find nearest neighbors (k-NN)
    vector<pair<int, double>> find_nearest(const vector<T>& query, int N) const;

    // Range search: returns indices within radius R
    vector<int> range_search(const vector<T>& query, float R) const;

private:
    // Core parameters
    int L;
    int k ;
    int dim;
    float w ;
    unsigned int seed;

    // Pointer to the data to be used for build/query; may be null until build()
    const vector<vector<T>>* data_ptr = nullptr;

    // Random projection parameters
    vector<vector<vector<float>>> a;  // L × k × dim random vectors
    vector<vector<float>> b;               // L × k random offsets
    vector<vector<uint32_t>> r;            // L × k random integer coefficients

    // Hash tables: L tables, each maps an integer hash key to indices
    vector<unordered_map<uint64_t, vector<int>>> hashTables;

    bool built = false;

    // Compute a single hash value for a vector
    uint64_t hash(int table, const vector<T>& v) const;
};