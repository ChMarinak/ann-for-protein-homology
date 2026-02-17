// hypercube.cpp
#include "hypercube.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>

using namespace std;

template <typename T>
Hypercube<T>::Hypercube(int dim, int kproj, int M, int probes, double w, int seed)
    : dim_(dim), kproj_(kproj), M_(M), probes_(probes), w_(w), seed_(seed)
{
    mt19937 rng(seed_);
    normal_distribution<float> normal(0.0f, 1.0f);
    uniform_real_distribution<float> uniform(0.0f, w_);

    proj_vectors_.resize(kproj_);
    offsets_.resize(kproj_);
    f_maps_.resize(kproj_);

    // Initialize random projection vectors and offsets
    for (int i = 0; i < kproj_; ++i) {
        proj_vectors_[i].resize(dim_);
        for (int d = 0; d < dim_; ++d)
            proj_vectors_[i][d] = normal(rng);

        offsets_[i] = uniform(rng);
    }
}


template <typename T>
float Hypercube<T>::dot_product(const vector<T>& v,
                                const vector<float>& proj) const {
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i)
        sum += float(v[i]) * proj[i];
    return sum;
}


template <typename T>
int Hypercube<T>::lsh_hash(const vector<T>& v,
                           const vector<float>& proj,
                           float b) const {
    float dot = dot_product(v, proj);
    return floor((dot + b) / w_);
}


template <typename T>
uint64_t Hypercube<T>::hash_vector(const vector<T>& v) const {
    uint64_t vertex = 0;

    for (int i = 0; i < kproj_; ++i) {

        int bucket = lsh_hash(v, proj_vectors_[i], offsets_[i]);

        auto& fmap = f_maps_[i];

        // If bucket not seen before, assign a random bit
        if (fmap.find(bucket) == fmap.end())
            fmap[bucket] = rand() % 2;

        if (fmap[bucket] == 1)
            vertex |= (1ULL << i);
    }

    return vertex;
}


template <typename T>
void Hypercube<T>::build(const vector<vector<T>>& dataset) {
    dataset_ = dataset;
    buckets_.clear();

    // Hash all dataset points to hypercube vertices
    for (int i = 0; i < (int)dataset_.size(); ++i) {
        uint64_t vertex = hash_vector(dataset_[i]);
        buckets_[vertex].push_back(i);
    }
}


template <typename T>
vector<uint64_t>
Hypercube<T>::get_probing_sequence(uint64_t initial_vertex) const {

    vector<uint64_t> sequence;
    sequence.reserve(probes_);

    queue<uint64_t> q;
    unordered_set<uint64_t> visited;

    q.push(initial_vertex);
    visited.insert(initial_vertex);

    // BFS traversal of hypercube graph
    while (!q.empty() &&
           (int)sequence.size() < probes_) {

        uint64_t current = q.front();
        q.pop();

        sequence.push_back(current);

        // Flip each bit to generate Hamming neighbors
        for (int i = 0; i < kproj_; ++i) {
            uint64_t neighbor = current ^ (1ULL << i);

            if (visited.insert(neighbor).second)
                q.push(neighbor);
        }
    }

    return sequence;
}


template <typename T>
vector<pair<int,double>>
Hypercube<T>::find_nearest(const vector<T>& query,
                           int N) const {

    vector<pair<int,double>> result;
    if (dataset_.empty()) return result;

    uint64_t qvertex = hash_vector(query);
    auto vertices = get_probing_sequence(qvertex);

    unordered_set<int> candidates;

    // Collect up to M candidate points
    for (uint64_t v : vertices) {
        auto it = buckets_.find(v);
        if (it == buckets_.end()) continue;

        for (int idx : it->second) {
            candidates.insert(idx);
            if ((int)candidates.size() >= M_)
                break;
        }

        if ((int)candidates.size() >= M_)
            break;
    }

    priority_queue<pair<double,int>> heap;

    // Keep top-N closest points
    for (int idx : candidates) {
        double dist = get_distance(query, dataset_[idx]);

        if ((int)heap.size() < N)
            heap.push({dist, idx});
        else if (dist < heap.top().first) {
            heap.pop();
            heap.push({dist, idx});
        }
    }

    result.resize(heap.size());
    for (int i = heap.size() - 1; i >= 0; --i) {
        result[i] = {heap.top().second,
                     heap.top().first};
        heap.pop();
    }

    return result;
}


template <typename T>
vector<int>
Hypercube<T>::range_search(const vector<T>& query,
                           double R) const {

    vector<int> result;
    if (dataset_.empty()) return result;

    uint64_t qvertex = hash_vector(query);
    auto vertices = get_probing_sequence(qvertex);

    unordered_set<int> visited_points;

    // Examine all points in probed vertices
    for (uint64_t v : vertices) {
        auto it = buckets_.find(v);
        if (it == buckets_.end()) continue;

        for (int idx : it->second) {

            if (!visited_points.insert(idx).second)
                continue;

            if (get_distance(query, dataset_[idx]) <= R)
                result.push_back(idx);
        }
    }

    return result;
}


template <typename T>
double Hypercube<T>::get_distance(
    const vector<T>& a,
    const vector<T>& b) const {

    double sum = 0.0;

    for (int i = 0; i < dim_; ++i) {
        double d = double(a[i]) - double(b[i]);
        sum += d * d;
    }

    return sqrt(sum);
}


// Explicit template instantiations
template class Hypercube<uint8_t>;
template class Hypercube<float>;
