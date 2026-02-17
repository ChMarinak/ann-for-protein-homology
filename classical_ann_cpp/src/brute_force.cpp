#include "brute_force.hpp"
#include <iostream>  
#include <fstream>   
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>


using namespace std;


template <typename T>
void write_output_true(
    const Dataset<T>& queries,
    const vector<SearchResult>& true_res,
    const Config& cfg)
{
    ofstream out(cfg.outpath);
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << cfg.outpath << "\n";
        return;
    }

    // Use the actual number of results (may be limited by cfg.max_queries)
    int Q = (int)true_res.size();
    out << cfg.algo << "\n";

    double total_tTrue = 0.0;

    for (int i = 0; i < Q; ++i) {
        out << "Query: " << i << "\n";

        // N-nearest
        int ncount = min(cfg.N, (int)true_res[i].indices.size());
        for (int j = 0; j < ncount; ++j) {
            out << "Nearest neighbor-" << j + 1 << ": "
                << true_res[i].indices[j] << "\n";
            out << "distanceTrue: " << true_res[i].dists[j] << "\n";
        }

        // R-near
        if (cfg.range_search) {
            out << "R-near neighbors:\n";
            for (auto idx : true_res[i].r_neighbors) {
                out << idx << "\n";
            }
        }

        out << "tTrueAverage: " << true_res[i].time << "\n\n";
        double qps_per_query = (true_res[i].time > 0.0) ? (1.0 / true_res[i].time) : 0.0;
        total_tTrue += true_res[i].time;
    }

    out << "Overall tTrueAverage: " << total_tTrue / Q << "\n";
    double overall_qps = (total_tTrue > 0.0) ? (static_cast<double>(Q) / total_tTrue) : 0.0;
    out << "Overall QPS: " << overall_qps << "\n";
    out.close();
}

template void write_output_true<uint8_t>(const Dataset<uint8_t>&,
    const vector<SearchResult>&, const Config&);

template void write_output_true<float>(const Dataset<float>&,
    const vector<SearchResult>&, const Config&);


template <typename Tq, typename Td>
SearchResult brute_force_query(
    const vector<Tq>& query,
    const Dataset<Td>& data,
    const Config& cfg)
{
    int N = cfg.N;
    float R = cfg.R;
    int D = data.dim;
    int Ndata = data.data.size();
    vector<pair<float, int>> dist_idx;
    dist_idx.reserve(Ndata);

    SearchResult res;

    float R2 = R * R;  // compare squared

    for (int i = 0; i < Ndata; ++i) {
        float dist2 = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = float(query[d]) - float(data.data[i][d]);
            dist2 += diff * diff;
        }
        dist_idx.emplace_back(dist2, i);

        // Collect R-near
        if (cfg.range_search && dist2 <= R2) {
            res.r_neighbors.push_back(i);
        }
    }

    // Sort for N-nearest
    sort(dist_idx.begin(), dist_idx.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    int limit = min(N, (int)dist_idx.size());
    res.indices.reserve(limit);
    res.dists.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        res.indices.push_back(dist_idx[i].second);
        res.dists.push_back(sqrt(dist_idx[i].first));
    }

    return res;
}

template SearchResult brute_force_query<float, float>(const vector<float>&, const Dataset<float>&, const Config&);

template SearchResult brute_force_query<uint8_t, uint8_t>(const vector<uint8_t>&, const Dataset<uint8_t>&, const Config&);
