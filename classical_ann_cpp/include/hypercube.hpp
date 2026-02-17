#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <cmath>

using namespace std;

template <typename T>
class Hypercube {
public:
    Hypercube(int dim, int kproj = 14, int M = 10, int probes = 2, double w = 4.0, int seed = 1);

    void build(const vector<vector<T>>& dataset);

    vector<pair<int,double>> find_nearest(
        const vector<T>& query, int N) const;

    vector<int> range_search(
        const vector<T>& query, double R) const;

    double get_distance(
        const vector<T>& a,
        const vector<T>& b) const;

private:
    int dim_;
    int kproj_;
    int M_;
    int probes_;
    double w_;
    int seed_;

    vector<vector<T>> dataset_;

    // Random projection vectors a_i
    vector<vector<float>> proj_vectors_;

    // Random offsets b_i
    vector<float> offsets_;

    // Random mapping f_i : bucket -> {0,1}
    mutable vector<unordered_map<int,int>> f_maps_;

    // Hypercube buckets
    unordered_map<uint64_t, vector<int>> buckets_;

    float dot_product(
        const vector<T>& v,
        const vector<float>& proj) const;

    int lsh_hash(
        const vector<T>& v,
        const vector<float>& proj,
        float b) const;

    uint64_t hash_vector(
        const vector<T>& v) const;

    vector<uint64_t> get_probing_sequence(
        uint64_t initial_vertex) const;
};
