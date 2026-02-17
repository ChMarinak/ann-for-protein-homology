#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include "utils.hpp"


template <typename T>
class LSH {
public:
    // Constructor
    LSH(int dim_, int L_ = 5, int k_ = 4, float w_ = 4.0f, unsigned int seed_ = 1);

    // Build functions
    void build(const std::vector<std::vector<T>>& dataset);

    // Find nearest neighbors (k-NN)
    std::vector<std::pair<int, double>> find_nearest(const std::vector<T>& query, int N) const;

    // Range search: returns indices within radius R
    std::vector<int> range_search(const std::vector<T>& query, float R) const;

private:
    // Core parameters
    int L;
    int k ;
    int dim;
    float w ;
    unsigned int seed;

    // Pointer to the data to be used for build/query; may be null until build()
    const std::vector<std::vector<T>>* data_ptr = nullptr;

    // Random projection parameters
    std::vector<std::vector<std::vector<float>>> a;  // L × k × dim random vectors
    std::vector<std::vector<float>> b;               // L × k random offsets
    std::vector<std::vector<uint32_t>> r;            // L × k random integer coefficients

    // Hash tables: L tables, each maps an integer hash key to indices
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> hashTables;

    bool built = false;

    // Compute a single hash value for a vector
    uint64_t hash(int table, const std::vector<T>& v) const;
};