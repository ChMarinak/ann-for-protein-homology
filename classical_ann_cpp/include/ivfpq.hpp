#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include <string>
#include <cstdint>

using namespace std;

template <typename T>
class IVFPQ {
public:
    // Constructor
    IVFPQ(int dimension, int kclusters = 50, int nprobe = 5,
          int M = 8, int nbits = 8, int seed = 1);

    // Train both coarse quantizer (IVF) and subquantizers (PQ)
    void train(const vector<vector<T>>& dataset);

    // Build index: assign points to clusters and compute PQ codes
    void build(const vector<vector<T>>& dataset);

    // Find N nearest neighbors (approximate search)
    vector<pair<int, double>> find_nearest(const vector<T>& query, int N) const;

    // Range search: all points within radius R (approximate)
    vector<int> range_search(const vector<T>& query, double R) const;

    // Get L2 distance (for evaluation)
    double get_distance(const vector<T>& a, const vector<T>& b) const;

    // Get centroids for analysis
    const vector<vector<float>>& get_centroids() const { return coarse_centroids_; }

    // Get silhouette
    double compute_silhouette_sample(int sample_size = 1000) const;

private:
    int dim_;           // vector dimension
    int kclusters_;     // coarse clusters
    int nprobe_;        // clusters to probe
    int M_;             // number of subquantizers
    int nbits_;         // bits per subquantizer
    int seed_;          // random seed
    int Ks_;            // number of PQ clusters = 2^nbits_

    // Coarse quantizer centroids
    vector<vector<float>> coarse_centroids_;

    // PQ codebooks: [M][Ks][dim_/M]
    vector<vector<vector<float>>> pq_codebooks_;

    // Inverted lists (coarse cluster -> vector indices)
    vector<vector<int>> inverted_lists_;

    // PQ-encoded dataset: for each point -> [M] byte codes
    vector<vector<uint8_t>> pq_codes_;

    // Original dataset
    vector<vector<T>> dataset_;

    // Helper methods
    int find_nearest_cluster(const vector<T>& v) const;
    vector<int> find_nprobe_clusters(const vector<T>& query) const;
    void kmeans(const vector<vector<float>>& data, int k, vector<vector<float>>& centroids);
    void train_pq(const vector<vector<T>>& dataset);
    vector<uint8_t> encode_vector(const vector<T>& vec) const;
    double distance(const vector<T>& query, const vector<uint8_t>& code) const;
    vector<int> point_assignments_;
};
