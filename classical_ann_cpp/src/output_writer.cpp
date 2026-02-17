#include "output_writer.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <filesystem>

using namespace std;


// FULL OUTPUT (Approximate + Evaluation)
template <typename T>
void write_output(
    const vector<T>& queries,
    const vector<SearchResult>& results,
    const vector<SearchResult>& gt,
    const Config& cfg,
    const EvalMetrics& metrics)
{
    ofstream fout(cfg.outpath);
    fout << fixed << setprecision(6);

    fout << cfg.algo << endl;

    int Q = results.size();
    int metric_idx = 0;

    for (int q = 0; q < Q; ++q) {
        fout << "Query: " << q << endl;

        int N = min({cfg.N,
                     (int)results[q].indices.size(),
                     (int)gt[q].indices.size()});

        for (int i = 0; i < N; ++i) {
            fout << "Nearest neighbor-" << (i + 1)
                 << ": " << results[q].indices[i] << endl;

            fout << "distanceApproximate: "
                 << results[q].dists[i] << endl;

            fout << "distanceTrue: "
                 << gt[q].dists[i] << endl;
        }

        if (cfg.range_search &&
            !results[q].r_neighbors.empty())
        {
            fout << "R-near neighbors:" << endl;
            for (int idx : results[q].r_neighbors)
                fout << idx << endl;
        }

        if (metric_idx < (int)metrics.AFs.size()) {
            fout << "Average AF: "
                 << metrics.AFs[metric_idx] << endl;

            fout << "Recall@N: "
                 << metrics.Recalls[metric_idx] << endl;

            metric_idx++;
        } else {
            fout << "Average AF: 0.0\n";
            fout << "Recall@N: 0.0\n";
        }

        fout << "QPS: " << metrics.QPS << endl;
        fout << "tApproximateAverage: "
             << results[q].time << endl;

        fout << "tTrueAverage: "
             << gt[q].time << endl;

        fout << endl;
    }

    fout << "Overall Average AF: "
         << metrics.avg_AF << endl;

    fout << "Overall Recall@N: "
         << metrics.avg_Recall << endl;

    fout << "Overall QPS: "
         << metrics.QPS << endl;

    fout << "Overall tApproximateAverage: "
         << metrics.tApproxAvg << endl;

    fout << "Overall tTrueAverage: "
         << metrics.tTrueAvg << endl;

    fout.close();
    cout << "Output written to: "
         << cfg.outpath << "\n";
}


// Neural-LSH graph format
template <typename T>
void write_output_neural(
    const Dataset<T>& data,
    const vector<SearchResult>& results,
    const Config& cfg)
{
    ofstream fout(cfg.outpath);

    int limit = min((int)data.data.size(),
                    (int)results.size());

    for (int i = 0; i < limit; ++i) {
        fout << i;

        for (int nb : results[i].indices) {
            if (nb == i) continue;
            fout << " " << nb;
        }
        fout << "\n";
    }

    fout.close();

    cout << "Neural-LSH k-NN graph written to: "
         << cfg.outpath << "\n";
}


// BIO output format
void write_bio_output(
    const vector<SearchResult>& results,
    const Config& cfg)
{
    filesystem::path outpath(cfg.outpath);
    if (outpath.has_parent_path())
        filesystem::create_directories(
            outpath.parent_path());

    ofstream fout(cfg.outpath);

    int Q = results.size();

    double total_time = 0.0;
    for (const auto& r : results)
        total_time += r.time;

    double time_per_query =
        (Q > 0) ? total_time / Q : 0.0;

    double qps =
        (time_per_query > 0)
        ? 1.0 / time_per_query
        : 0.0;

    fout << "TYPE bio\n";
    fout << "METHOD " << cfg.algo << "\n";
    fout << "N_eval " << cfg.N << "\n";
    fout << "Time_per_query "
         << time_per_query << "\n";
    fout << "QPS " << qps << "\n";

    for (int q = 0; q < Q; ++q) {
        fout << "\nQUERY " << q << "\n";

        int K = min(cfg.N,
                    (int)results[q].indices.size());

        for (int i = 0; i < K; ++i) {
            fout << (i + 1) << " "
                 << results[q].indices[i] << " "
                 << results[q].dists[i] << "\n";
        }
    }

    fout.close();
}


// Brute-only output
template <typename T>
void write_output_true(
    const vector<T>& queries,
    const vector<SearchResult>& results,
    const Config& cfg)
{
    ofstream fout(cfg.outpath);
    fout << fixed << setprecision(6);

    fout << "brute\n";

    int Q = results.size();

    for (int q = 0; q < Q; ++q) {
        fout << "Query: " << q << endl;

        int N = min(cfg.N,
                    (int)results[q].indices.size());

        for (int i = 0; i < N; ++i) {
            fout << "Nearest neighbor-"
                 << (i + 1)
                 << ": "
                 << results[q].indices[i]
                 << endl;

            fout << "distanceTrue: "
                 << results[q].dists[i]
                 << endl;
        }

        fout << "tTrueAverage: "
             << results[q].time << endl;

        fout << endl;
    }

    fout.close();
}


// Explicit template instantiations

// write_output
template void write_output<std::vector<float>>(
    const std::vector<std::vector<float>>&,
    const std::vector<SearchResult>&,
    const std::vector<SearchResult>&,
    const Config&,
    const EvalMetrics&);

template void write_output<std::vector<uint8_t>>(
    const std::vector<std::vector<uint8_t>>&,
    const std::vector<SearchResult>&,
    const std::vector<SearchResult>&,
    const Config&,
    const EvalMetrics&);


// write_output_neural
template void write_output_neural<float>(
    const Dataset<float>&,
    const std::vector<SearchResult>&,
    const Config&);

template void write_output_neural<uint8_t>(
    const Dataset<uint8_t>&,
    const std::vector<SearchResult>&,
    const Config&);


// write_output_true
template void write_output_true<std::vector<float>>(
    const std::vector<std::vector<float>>&,
    const std::vector<SearchResult>&,
    const Config&);

template void write_output_true<std::vector<uint8_t>>(
    const std::vector<std::vector<uint8_t>>&,
    const std::vector<SearchResult>&,
    const Config&);