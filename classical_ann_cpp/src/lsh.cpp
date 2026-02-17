#include "lsh.hpp"
#include <sstream>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <cmath>

using namespace std;

static constexpr uint64_t PRIME_M = 4294967291ULL;

template <typename T>
LSH<T>::LSH(int dim_, int L_, int k_, float w_, unsigned int seed_)
    : L(L_), k(k_), dim(dim_), w(w_), seed(seed_)
{
    mt19937 gen(seed);
    normal_distribution<float> normal(0.0f, 1.0f);
    uniform_real_distribution<float> uniform(0.0f, w);
    uniform_int_distribution<uint32_t> randint(1, PRIME_M - 1);

    a.resize(L, vector<vector<float>>(k, vector<float>(dim)));
    b.resize(L, vector<float>(k));
    r.resize(L, vector<uint32_t>(k));
    hashTables.resize(L);

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int d = 0; d < dim; ++d)
                a[i][j][d] = normal(gen);
            b[i][j] = uniform(gen);
            r[i][j] = randint(gen);
        }
    }
}


// Compute LSH hash
template <typename T>
uint64_t LSH<T>::hash(int table, const vector<T>& v) const
{
    long double g = 0.0L; // accumulate in high precision

    for (int j = 0; j < k; ++j) {
        long double dot = 0.0L;
        for (int d = 0; d < dim; ++d)
            dot += static_cast<long double>(a[table][j][d]) * static_cast<long double>(v[d]);

        // Compute h_j = floor((a_j · v + b_j) / w)
        long double h = floor((dot + b[table][j]) / w);

        // Ensure positive modular arithmetic
        uint64_t hmod = static_cast<uint64_t>(fmod(fmod(h, (long double)PRIME_M) + (long double)PRIME_M, (long double)PRIME_M));

        // Combine hashes with random coefficients
        g = fmod(g + (r[table][j] * hmod), (long double)PRIME_M);
    }

    return static_cast<uint64_t>(g);
}



// Build L hash tables using
template <typename T>
void LSH<T>::build(const vector<vector<T>>& dataset)
{
    if (built) return;
    data_ptr = &dataset;

    size_t tableSize = dataset.size() / 4 + 1; // heuristic bucket scaling

    for (int idx = 0; idx < (int)dataset.size(); ++idx) {
        for (int l = 0; l < L; ++l) {
            uint64_t g = hash(l, dataset[idx]) % tableSize;
            hashTables[l][g].push_back(idx);
        }
    }

    built = true;
}


// Query for approximate kNN or range search
template <typename T>
vector<pair<int, double>> LSH<T>::find_nearest(const vector<T>& query, int N) const
{
    if (!built)
        throw runtime_error("LSH index not built! Call build() first.");
    if (!data_ptr) throw runtime_error("LSH::find_nearest(): index not built or no data available.");

    float R2 = 0.0f; (void)R2; // unused here
    unordered_set<int> visited;
    vector<pair<float, int>> dist_idx;

    size_t tableSize = data_ptr->size() / 4 + 1;
    vector<int> candidates;

    // Candidate collection
    for (int l = 0; l < L; ++l) {
        uint64_t g = hash(l, query) % tableSize;
        auto it = hashTables[l].find(g);
        if (it != hashTables[l].end()) {
            for (int idx : it->second)
                if (visited.insert(idx).second)
                    candidates.push_back(idx);
        }
    }

    // Distance computation
    dist_idx.reserve(candidates.size());
    for (int idx : candidates) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = query[d] - (*data_ptr)[idx][d];
            dist2 += diff * diff;
        }
        dist_idx.emplace_back(dist2, idx);
    }

    // Sort and take top-N
    sort(dist_idx.begin(), dist_idx.end(),
         [](auto& a, auto& b) { return a.first < b.first; });

    int limit = min(N, (int)dist_idx.size());
    vector<pair<int, double>> neighbors;
    neighbors.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        neighbors.emplace_back(dist_idx[i].second, sqrt(dist_idx[i].first));
    }

    return neighbors;
}

template <typename T>
vector<int> LSH<T>::range_search(const vector<T>& query, float R) const
{
    if (!built)
        throw runtime_error("LSH index not built! Call build() first.");
    if (!data_ptr) throw runtime_error("LSH::range_search(): index not built or no data available.");

    float R2 = R * R;
    unordered_set<int> visited;
    vector<int> results;

    size_t tableSize = data_ptr->size() / 4 + 1;

    for (int l = 0; l < L; ++l) {
        uint64_t g = hash(l, query) % tableSize;
        auto it = hashTables[l].find(g);
        if (it != hashTables[l].end()) {
            for (int idx : it->second) {
                if (!visited.insert(idx).second) continue;
                float dist2 = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = query[d] - (*data_ptr)[idx][d];
                    dist2 += diff * diff;
                }
                if (dist2 <= R2)
                    results.push_back(idx);
            }
        }
    }

    return results;
}

template class LSH<float>;
template class LSH<uint8_t>;
