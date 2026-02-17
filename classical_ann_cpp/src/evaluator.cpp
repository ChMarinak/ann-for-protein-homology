#include "evaluator.hpp"
#include "utils.hpp"
#include <algorithm>

using namespace std;

EvalMetrics evaluate(
    const vector<SearchResult>& results,
    const vector<SearchResult>& gt,
    const Config& cfg,
    double tTrueAvg_from_gt
) {
    EvalMetrics metrics;
    int Q = results.size();

    double total_AF = 0.0;
    double total_Recall = 0.0;
    double total_time = 0.0;
    double total_tTrue = 0.0;

    for (int q = 0; q < Q; ++q) {
        int N = min({cfg.N, (int)gt[q].indices.size(),
                               (int)results[q].indices.size()});

        double sum_af = 0.0;
        int match_count = 0;

        for (int i = 0; i < N; ++i) {
            double dist_true   = gt[q].dists[i];
            double dist_approx = results[q].dists[i];

            sum_af += dist_approx / (dist_true + 1e-9);

            if (find(gt[q].indices.begin(),
                     gt[q].indices.begin() + N,
                     results[q].indices[i])
                != gt[q].indices.begin() + N)
                match_count++;
        }

        double AF     = (N > 0) ? sum_af / N : 0.0;
        double recall = (N > 0) ? (double)match_count / N : 0.0;

        metrics.AFs.push_back(AF);
        metrics.Recalls.push_back(recall);

        total_AF     += AF;
        total_Recall += recall;
        total_time   += results[q].time;
        total_tTrue  += gt[q].time;
    }

    metrics.avg_AF      = Q > 0 ? total_AF / Q : 0.0;
    metrics.avg_Recall  = Q > 0 ? total_Recall / Q : 0.0;
    metrics.tApproxAvg  = Q > 0 ? total_time / Q : 0.0;
    metrics.tTrueAvg    = Q > 0 ? total_tTrue / Q : tTrueAvg_from_gt;
    metrics.QPS         = total_time > 0 ? Q / total_time : 0.0;

    return metrics;
}
