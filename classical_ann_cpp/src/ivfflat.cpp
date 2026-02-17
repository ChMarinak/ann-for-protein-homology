#include "ivfflat.hpp"
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <iostream>

using namespace std;

template <typename T>
IVFFlat<T>::IVFFlat(int dimension, int kclusters, int nprobe, int seed)
    : dim_(dimension), kclusters_(kclusters), nprobe_(nprobe), seed_(seed) {
}

template <typename T>
double IVFFlat<T>::get_distance(const vector<T>& a, const vector<T>& b) const {
    double sum = 0.0;
    for (int i = 0; i < dim_; ++i) {
        double d = double(a[i]) - double(b[i]);
        sum += d * d;
    }
    return sqrt(sum);
}

template <typename T>
void IVFFlat<T>::kmeans_clustering(const vector<vector<T>>& dataset, int num_clusters) {
    int n = dataset.size();
    mt19937 rng(seed_);
    uniform_int_distribution<int> dist(0, n - 1);

    // Initialize centroids: simply pick random points
    centroids_.clear();
    set<int> selected_indices;
    for (int i = 0; i < num_clusters && (int)selected_indices.size() < n; ++i) {
        int idx;
        do {
            idx = dist(rng);
        } while (selected_indices.count(idx));
        selected_indices.insert(idx);
        centroids_.push_back(vector<float>(dim_));
        for (int d = 0; d < dim_; ++d) {
            centroids_[i][d] = float(dataset[idx][d]);
        }
    }

    // K-means iterations
    vector<int> assignments(n);
    double prev_inertia = numeric_limits<double>::max();
    
    for (int iter = 0; iter < 10; ++iter) {
        // Assign points to nearest centroid
        double inertia = 0.0;
        for (int i = 0; i < n; ++i) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;
            for (int c = 0; c < (int)centroids_.size(); ++c) {
                double d = 0.0;
                for (int j = 0; j < dim_; ++j) {
                    double diff = double(dataset[i][j]) - centroids_[c][j];
                    d += diff * diff;
                }
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
            inertia += min_dist;
        }
        
        // Early stopping if inertia converged
        if (iter > 0 && abs(inertia - prev_inertia) < 1e-4 * inertia) {
            break;
        }
        prev_inertia = inertia;

        // Update centroids
        vector<vector<double>> new_centroids((int)centroids_.size(), vector<double>(dim_, 0.0));
        vector<int> cluster_sizes((int)centroids_.size(), 0);
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            cluster_sizes[c]++;
            for (int d = 0; d < dim_; ++d) {
                new_centroids[c][d] += double(dataset[i][d]);
            }
        }

        for (int c = 0; c < (int)centroids_.size(); ++c) {
            if (cluster_sizes[c] > 0) {
                for (int d = 0; d < dim_; ++d) {
                    centroids_[c][d] = float(new_centroids[c][d] / cluster_sizes[c]);
                }
            }
        }
    }
}template <typename T>
void IVFFlat<T>::train(const vector<vector<T>>& dataset) {
    cout << "Training...\n";
    kmeans_clustering(dataset, kclusters_);
}

template <typename T>
void IVFFlat<T>::build(const vector<vector<T>>& dataset) {
    dataset_ = dataset;
    int n = dataset.size();

    // Initialize inverted lists
    inverted_lists_.assign(kclusters_, vector<int>());

    // Assign each point to its nearest cluster
    for (int i = 0; i < n; ++i) {
        int cluster = find_nearest_cluster(dataset[i]);
        inverted_lists_[cluster].push_back(i);
        point_assignments_.push_back(cluster); 
    }
}

template <typename T>
int IVFFlat<T>::find_nearest_cluster(const vector<T>& v) const {
    if (centroids_.empty()) return 0;  // Safety check
    double min_dist = numeric_limits<double>::max();
    int best_cluster = 0;
    for (int c = 0; c < kclusters_; ++c) {
        if (c >= (int)centroids_.size()) break;  // Safety check
        double d = 0.0;
        for (int d_idx = 0; d_idx < dim_; ++d_idx) {
            double diff = double(v[d_idx]) - centroids_[c][d_idx];
            d += diff * diff;
        }
        if (d < min_dist) {
            min_dist = d;
            best_cluster = c;
        }
    }
    return best_cluster;
}

template <typename T>
vector<int> IVFFlat<T>::find_nprobe_clusters(const vector<T>& query) const {
    if (centroids_.empty()) return vector<int>();  // Safety check
    vector<pair<double, int>> cluster_distances;
    for (int c = 0; c < (int)centroids_.size(); ++c) {
        double d = 0.0;
        for (int d_idx = 0; d_idx < dim_; ++d_idx) {
            double diff = double(query[d_idx]) - centroids_[c][d_idx];
            d += diff * diff;
        }
        cluster_distances.push_back({d, c});
    }

    // Sort by distance and return top nprobe clusters
    sort(cluster_distances.begin(), cluster_distances.end());
    vector<int> result;
    for (int i = 0; i < min(nprobe_, (int)cluster_distances.size()); ++i) {
        result.push_back(cluster_distances[i].second);
    }
    return result;
}

template <typename T>
vector<pair<int, double>> IVFFlat<T>::find_nearest(const vector<T>& query, int N) const {
    vector<pair<int, double>> result;
    if (dataset_.empty()) return result;

    // Find top nprobe clusters
    auto probe_clusters = find_nprobe_clusters(query);

    // Collect candidates from these clusters
    vector<pair<double, int>> candidates; // (distance, index)
    for (int cluster : probe_clusters) {
        for (int idx : inverted_lists_[cluster]) {
            double d = get_distance(query, dataset_[idx]);
            candidates.push_back({d, idx});
        }
    }

    // Sort by distance and return top-N
    sort(candidates.begin(), candidates.end());
    for (int i = 0; i < min(N, (int)candidates.size()); ++i) {
        result.push_back({candidates[i].second, candidates[i].first});
    }

    return result;
}

template <typename T>
vector<int> IVFFlat<T>::range_search(const vector<T>& query, double R) const {
    vector<int> result;
    if (dataset_.empty()) return result;

    // Find all clusters within reasonable distance
    auto probe_clusters = find_nprobe_clusters(query);

    // Check all points in probed clusters
    for (int cluster : probe_clusters) {
        for (int idx : inverted_lists_[cluster]) {
            double d = get_distance(query, dataset_[idx]);
            if (d <= R) {
                result.push_back(idx);
            }
        }
    }

    return result;
}

template <typename T>
double IVFFlat<T>::compute_silhouette_sample(int sample_size) const {
    int N = dataset_.size();
    if (N == 0 || kclusters_ <= 1) return 0.0;

    sample_size = min(sample_size, N);

    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(seed_);
    shuffle(indices.begin(), indices.end(), rng);
    indices.resize(sample_size);

    vector<double> silhouette_values(sample_size, 0.0);

    // Precompute clusters
    vector<vector<int>> clusters(kclusters_);
    for (int i = 0; i < N; ++i)
        clusters[point_assignments_[i]].push_back(i);

    for (int idx = 0; idx < sample_size; ++idx) {
        int i = indices[idx];
        int ci = point_assignments_[i];

        double a = 0.0;
        if (clusters[ci].size() > 1) {
            for (int j : clusters[ci]) if (i != j) a += get_distance(dataset_[i], dataset_[j]);
            a /= (clusters[ci].size() - 1);
        }

        double b = numeric_limits<double>::max();
        for (int c = 0; c < kclusters_; ++c) {
            if (c == ci || clusters[c].empty()) continue;
            int sample_c = min(100, (int)clusters[c].size());
            vector<int> cluster_c_sample = clusters[c];
            shuffle(cluster_c_sample.begin(), cluster_c_sample.end(), rng);
            cluster_c_sample.resize(sample_c);

            double dist_sum = 0.0;
            for (int j : cluster_c_sample) dist_sum += get_distance(dataset_[i], dataset_[j]);
            b = min(b, dist_sum / sample_c);
        }

        silhouette_values[idx] = (b - a) / max(a, b);
    }

    double total = 0.0;
    for (double s : silhouette_values) total += s;
    return total / sample_size;
}

// Explicit template instantiations
template class IVFFlat<uint8_t>;
template class IVFFlat<float>;
