#include "ivfpq.hpp"
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <iostream>

using namespace std;

template <typename T>
IVFPQ<T>::IVFPQ(int dimension, int kclusters, int nprobe, int M, int nbits, int seed)
    : dim_(dimension), kclusters_(kclusters), nprobe_(nprobe),
      M_(M), nbits_(nbits), seed_(seed) {
    Ks_ = 1 << nbits_;
}

// Basic L2 distance
template <typename T>
double IVFPQ<T>::get_distance(const vector<T>& a, const vector<T>& b) const {
    double sum = 0.0;
    for (int i = 0; i < dim_; ++i) {
        double d = double(a[i]) - double(b[i]);
        sum += d * d;
    }
    return sqrt(sum);
}

// Generic k-means used for both IVF and PQ training
template <typename T>
void IVFPQ<T>::kmeans(const vector<vector<float>>& data, int k, vector<vector<float>>& centroids) {
    int n = data.size();
    int dim = data[0].size();
    mt19937 rng(seed_);
    uniform_int_distribution<int> dist(0, n - 1);

    centroids.clear();
    set<int> used;
    for (int i = 0; i < k; ++i) {
        int idx;
        do { idx = dist(rng); } while (used.count(idx));
        used.insert(idx);
        centroids.push_back(data[idx]);
    }

    vector<int> assignments(n);
    for (int iter = 0; iter < 10; ++iter) {
        // Assign step
        for (int i = 0; i < n; ++i) {
            double best = numeric_limits<double>::max();
            int bestc = 0;
            for (int c = 0; c < k; ++c) {
                double d = 0.0;
                for (int j = 0; j < dim; ++j) {
                    double diff = data[i][j] - centroids[c][j];
                    d += diff * diff;
                }
                if (d < best) { best = d; bestc = c; }
            }
            assignments[i] = bestc;
        }

        // Update step
        vector<vector<double>> new_centroids(k, vector<double>(dim, 0.0));
        vector<int> counts(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < dim; ++j)
                new_centroids[c][j] += data[i][j];
        }
        for (int c = 0; c < k; ++c)
            if (counts[c] > 0)
                for (int j = 0; j < dim; ++j)
                    centroids[c][j] = float(new_centroids[c][j] / counts[c]);
    }
}

// Train coarse and PQ quantizers
template <typename T>
void IVFPQ<T>::train(const vector<vector<T>>& dataset) {
    cout << "Training...\n";
    // Train IVF (coarse)
    vector<vector<float>> data_f(dataset.size(), vector<float>(dim_));
    for (size_t i = 0; i < dataset.size(); ++i)
        for (int j = 0; j < dim_; ++j)
            data_f[i][j] = float(dataset[i][j]);

    kmeans(data_f, kclusters_, coarse_centroids_);

    train_pq(dataset);
}

// PQ training: learns subquantizer codebooks
template <typename T>
void IVFPQ<T>::train_pq(const vector<vector<T>>& dataset) {
    int subdim = dim_ / M_;
    pq_codebooks_.assign(M_, vector<vector<float>>(Ks_, vector<float>(subdim, 0.0)));

    for (int m = 0; m < M_; ++m) {
        vector<vector<float>> sub_data;
        for (auto& v : dataset) {
            vector<float> sub(subdim);
            for (int j = 0; j < subdim; ++j)
                sub[j] = float(v[m * subdim + j]);
            sub_data.push_back(sub);
        }
        kmeans(sub_data, Ks_, pq_codebooks_[m]);
    }
}

// Find nearest IVF cluster
template <typename T>
int IVFPQ<T>::find_nearest_cluster(const vector<T>& v) const {
    double best = numeric_limits<double>::max();
    int bestc = 0;
    for (int c = 0; c < kclusters_; ++c) {
        double d = 0.0;
        for (int j = 0; j < dim_; ++j) {
            double diff = double(v[j]) - coarse_centroids_[c][j];
            d += diff * diff;
        }
        if (d < best) { best = d; bestc = c; }
    }
    return bestc;
}

// Select nprobe clusters
template <typename T>
vector<int> IVFPQ<T>::find_nprobe_clusters(const vector<T>& query) const {
    vector<pair<double,int>> dist;
    for (int c = 0; c < kclusters_; ++c) {
        double d = 0.0;
        for (int j = 0; j < dim_; ++j) {
            double diff = double(query[j]) - coarse_centroids_[c][j];
            d += diff * diff;
        }
        dist.push_back({d,c});
    }
    sort(dist.begin(), dist.end());
    vector<int> out;
    for (int i = 0; i < min(nprobe_, (int)dist.size()); ++i)
        out.push_back(dist[i].second);
    return out;
}

// Encode a vector using PQ
template <typename T>
vector<uint8_t> IVFPQ<T>::encode_vector(const vector<T>& vec) const {
    int subdim = dim_ / M_;
    vector<uint8_t> codes(M_);
    for (int m = 0; m < M_; ++m) {
        double best = numeric_limits<double>::max();
        int bestc = 0;
        for (int k = 0; k < Ks_; ++k) {
            double d = 0.0;
            for (int j = 0; j < subdim; ++j) {
                double diff = double(vec[m * subdim + j]) - pq_codebooks_[m][k][j];
                d += diff * diff;
            }
            if (d < best) { best = d; bestc = k; }
        }
        codes[m] = (uint8_t)bestc;
    }
    return codes;
}

// Build the IVF+PQ index
template <typename T>
void IVFPQ<T>::build(const vector<vector<T>>& dataset) {
    dataset_ = dataset;
    inverted_lists_.assign(kclusters_, {});
    pq_codes_.resize(dataset.size());

    for (int i = 0; i < (int)dataset.size(); ++i) {
        int c = find_nearest_cluster(dataset[i]);
        inverted_lists_[c].push_back(i);
        pq_codes_[i] = encode_vector(dataset[i]);
        point_assignments_.push_back(c); 
    }
}

// Compute asymmetric distance (ADC)
template <typename T>
double IVFPQ<T>::distance(const vector<T>& query, const vector<uint8_t>& code) const {
    int subdim = dim_ / M_;
    double total = 0.0;
    for (int m = 0; m < M_; ++m) {
        int k = code[m];
        for (int j = 0; j < subdim; ++j) {
            double diff = double(query[m * subdim + j]) - pq_codebooks_[m][k][j];
            total += diff * diff;
        }
    }
    return sqrt(total);
}

// Find nearest neighbors (approximate)
template <typename T>
vector<pair<int,double>> IVFPQ<T>::find_nearest(const vector<T>& query, int N) const {
    vector<pair<double,int>> candidates;
    auto clusters = find_nprobe_clusters(query);

    for (int c : clusters) {
        for (int idx : inverted_lists_[c]) {
            double d = distance(query, pq_codes_[idx]);
            candidates.push_back({d, idx});
        }
    }

    sort(candidates.begin(), candidates.end());
    vector<pair<int,double>> result;
    for (int i = 0; i < min(N, (int)candidates.size()); ++i)
        result.push_back({candidates[i].second, candidates[i].first});
    return result;
}

// Range search
template <typename T>
vector<int> IVFPQ<T>::range_search(const vector<T>& query, double R) const {
    vector<int> result;
    auto clusters = find_nprobe_clusters(query);

    for (int c : clusters) {
        for (int idx : inverted_lists_[c]) {
            double d = distance(query, pq_codes_[idx]);
            if (d <= R) result.push_back(idx);
        }
    }
    return result;
}

template <typename T>
double IVFPQ<T>::compute_silhouette_sample(int sample_size) const {
    int N = dataset_.size();
    if (N == 0 || kclusters_ <= 1) return 0.0;

    // Adjust sample size if dataset is small
    sample_size = min(sample_size, N);

    // Randomly sample points
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(seed_);
    shuffle(indices.begin(), indices.end(), rng);
    indices.resize(sample_size);

    vector<double> silhouette_values(sample_size, 0.0);

    // Precompute cluster members (coarse clusters)
    vector<vector<int>> clusters(kclusters_);
    for (int i = 0; i < N; ++i)
        clusters[point_assignments_[i]].push_back(i);

    for (int idx = 0; idx < sample_size; ++idx) {
        int i = indices[idx];
        int ci = point_assignments_[i];

        // Compute a(i): mean distance within same cluster
        double a = 0.0;
        if (clusters[ci].size() > 1) {
            for (int j : clusters[ci]) if (i != j)
                a += get_distance(dataset_[i], dataset_[j]);
            a /= (clusters[ci].size() - 1);
        }

        // Compute b(i): nearest other cluster mean distance
        double b = numeric_limits<double>::max();
        for (int c = 0; c < kclusters_; ++c) {
            if (c == ci || clusters[c].empty()) continue;

            // Sample up to 100 points per cluster for speed
            int sample_c = min(100, (int)clusters[c].size());
            vector<int> cluster_c_sample = clusters[c];
            shuffle(cluster_c_sample.begin(), cluster_c_sample.end(), rng);
            cluster_c_sample.resize(sample_c);

            double dist_sum = 0.0;
            for (int j : cluster_c_sample)
                dist_sum += get_distance(dataset_[i], dataset_[j]);

            double mean_dist = dist_sum / sample_c;
            b = min(b, mean_dist);
        }

        silhouette_values[idx] = (b - a) / max(a, b);
    }

    double total = 0.0;
    for (double s : silhouette_values)
        total += s;

    return total / sample_size;
}

// Explicit template instantiations
template class IVFPQ<uint8_t>;
template class IVFPQ<float>;
