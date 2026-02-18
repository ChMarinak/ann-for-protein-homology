#include "brute_force.hpp"
#include <iostream>  
#include <fstream>   
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace std;

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
