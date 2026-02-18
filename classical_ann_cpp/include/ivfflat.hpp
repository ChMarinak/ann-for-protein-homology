#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include <string>
#include <cstdint>

using namespace std;

template <typename T>
class IVFFlat {
public:
    // Constructor
    IVFFlat(int dimension, int kclusters = 50, int nprobe = 5, int seed = 1);

    // Train the index using k-means clustering
    void train(const vector<vector<T>>& dataset);

    // Build index: assign all dataset points to the nearest cluster
    void build(const vector<vector<T>>& dataset);

    // Find N nearest neighbors
    vector<pair<int, double>> find_nearest(const vector<T>& query, int N) const;

    // Range search - returns points within radius R
    vector<int> range_search(const vector<T>& query, double R) const;

    // Get exact L2 distance between two points
    double get_distance(const vector<T>& a, const vector<T>& b) const;

    // Get cluster centroids (for analysis)
    const vector<vector<float>>& get_centroids() const { return centroids_; }

    // Get silhouette
    double compute_silhouette_sample(int sample_size = 1000) const;

private:
    int dim_;           // dimension of vectors
    int kclusters_;     // number of clusters (centroids)
    int nprobe_;        // number of clusters to check during search
    int seed_;          // seed for reproducibility

    // Centroids (cluster centers)
    vector<vector<float>> centroids_;

    // Inverted lists: cluster_id -> vector of point indices in that cluster
    vector<vector<int>> inverted_lists_;

    // Original dataset
    vector<vector<T>> dataset_;

    // Helper methods
    int find_nearest_cluster(const vector<T>& v) const;
    vector<int> find_nprobe_clusters(const vector<T>& query) const;
    void kmeans_clustering(const vector<vector<T>>& dataset, int num_clusters);
    vector<int> point_assignments_;
};
