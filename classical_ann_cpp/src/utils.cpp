#include "utils.hpp"

#include "file_parser.hpp"
#include "evaluator.hpp"
#include "output_writer.hpp"

#include "brute_force.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <set>
#include <filesystem>

using namespace std;


// Argument Parsing
void parse_arguments(int argc, char** argv, Config& cfg)
{
    set<string> algo_flags = {
        "-lsh", "-hypercube", "-ivfflat",
        "-ivfpq", "-brute"
    };

    // Detect algorithm flag first
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (algo_flags.count(arg))
            cfg.algo = arg.substr(1);
    }

    // Parse rest
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (algo_flags.count(arg)) continue;

        if (arg.size() == 2 && arg[0] == '-') {
            if (i + 1 >= argc) break;
            string val = argv[++i];

            switch (arg[1]) {
                case 'd': cfg.dpath = val; break;
                case 'q': cfg.qpath = val; break;
                case 'o': cfg.outpath = val; break;
                case 'e': cfg.evalpath = val; break;
                case 't': cfg.type = val; break;
                case 'N': cfg.N = atoi(val.c_str()); break;
                case 'R': cfg.R = atof(val.c_str()); break;
                case 'k': cfg.k = atoi(val.c_str()); break;
                case 'L': cfg.L = atoi(val.c_str()); break;
                case 'w': cfg.w = atof(val.c_str()); break;
                case 's': cfg.seed = atoi(val.c_str()); break;
                case 'P': cfg.kproj = atoi(val.c_str()); break;
                case 'M': cfg.M = atoi(val.c_str()); break;
                case 'p': cfg.probes = atoi(val.c_str()); break;
                case 'c': cfg.kclusters = atoi(val.c_str()); break;
                case 'n': cfg.nprobe = atoi(val.c_str()); break;
                case 'b': cfg.nbits = atoi(val.c_str()); break;
                case 'r': cfg.range_search = (val == "true"); break;
                case 'Q': cfg.max_queries = atoi(val.c_str()); break;
                default: break;
            }
        }
    }

    if (cfg.type == "sift" && cfg.R == 2000.0f)
        cfg.R = 2.0f;
}


// Timing Utility
template <typename Func>
double measure_time(Func&& f)
{
    using namespace chrono;
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}


// Core Algorithm Runner
template <typename T>
void run_algorithm(const Dataset<T>& data,
                   const Dataset<T>& queries,
                   const Config& cfg)
{
    cout << "Running search algorithm: "
         << cfg.algo << endl;

    int Q = queries.data.size();
    if (cfg.max_queries > 0)
        Q = min(Q, cfg.max_queries);

    vector<SearchResult> results(Q);

    auto fill_result = [&](SearchResult& res,
                           const vector<pair<int,double>>& neighbors)
    {
        res.indices.clear();
        res.dists.clear();

        for (auto& p : neighbors) {
            res.indices.push_back(p.first);
            res.dists.push_back((float)p.second);
        }
    };


    // ================= BRUTE =================
    if (cfg.algo == "brute") {

        double total_query_time = measure_time([&]() {
            for (int qi = 0; qi < Q; ++qi) {
                results[qi].time = measure_time([&]() {
                    results[qi] =
                        brute_force_query(
                            queries.data[qi], data, cfg);
                });
            }
        });

        cout << "Brute-force Total Query Time: "
            << total_query_time << " sec\n";
    }

    // ================= LSH =================
    else if (cfg.algo == "lsh") {

        LSH<T> index(data.dim,
                    cfg.L, cfg.k,
                    cfg.w, cfg.seed);

        double build_time = measure_time([&]() {
            index.build(data.data);
        });

        cout << "LSH Build Time: "
            << build_time << " sec\n";

        double total_query_time = measure_time([&]() {
            for (int qi = 0; qi < Q; ++qi) {

                results[qi].time = measure_time([&]() {
                    auto neighbors =
                        index.find_nearest(
                            queries.data[qi], cfg.N);
                    fill_result(results[qi], neighbors);
                });

                if (cfg.range_search)
                    results[qi].r_neighbors =
                        index.range_search(
                            queries.data[qi], cfg.R);
            }
        });

        cout << "LSH Total Query Time: "
            << total_query_time << " sec\n";
    }

    // ================= HYPERCUBE =================
    else if (cfg.algo == "hypercube") {

        Hypercube<T> index(data.dim,
                        cfg.kproj,
                        cfg.M,
                        cfg.probes,
                        cfg.w,
                        cfg.seed);

        double build_time = measure_time([&]() {
            index.build(data.data);
        });

        cout << "Hypercube Build Time: "
            << build_time << " sec\n";

        double total_query_time = measure_time([&]() {
            for (int qi = 0; qi < Q; ++qi) {

                results[qi].time = measure_time([&]() {
                    auto neighbors =
                        index.find_nearest(
                            queries.data[qi], cfg.N);
                    fill_result(results[qi], neighbors);
                });

                if (cfg.range_search)
                    results[qi].r_neighbors =
                        index.range_search(
                            queries.data[qi], cfg.R);
            }
        });

        cout << "Hypercube Total Query Time: "
            << total_query_time << " sec\n";
    }

    // ================= IVFFlat =================
    else if (cfg.algo == "ivfflat") {
        IVFFlat<T> index(data.dim,
                         cfg.kclusters,
                         cfg.nprobe,
                         cfg.seed);

        double build_time = measure_time([&]() {
            index.train(data.data);
            index.build(data.data);
        });

        cout << "IVF-Flat Train+Build Time: "
             << build_time << " sec\n";

        double total_query_time = measure_time([&]() {
            for (int qi = 0; qi < Q; ++qi) {

                results[qi].time = measure_time([&]() {
                    auto neighbors =
                        index.find_nearest(
                            queries.data[qi], cfg.N);
                    fill_result(results[qi], neighbors);
                });

                if (cfg.range_search)
                    results[qi].r_neighbors =
                        index.range_search(
                            queries.data[qi], cfg.R);
            }
        });

        cout << "IVF-Flat Total Query Time: "
             << total_query_time << " sec\n";
    }

    // ================= IVFPQ =================
    else if (cfg.algo == "ivfpq") {
        IVFPQ<T> index(data.dim,
                       cfg.kclusters,
                       cfg.nprobe,
                       cfg.M,
                       cfg.nbits,
                       cfg.seed);

        double build_time = measure_time([&]() {
            index.train(data.data);
            index.build(data.data);
        });

        cout << "IVF-PQ Train+Build Time: "
             << build_time << " sec\n";

        double total_query_time = measure_time([&]() {
            for (int qi = 0; qi < Q; ++qi) {

                results[qi].time = measure_time([&]() {
                    auto neighbors =
                        index.find_nearest(
                            queries.data[qi], cfg.N);
                    fill_result(results[qi], neighbors);
                });

                if (cfg.range_search)
                    results[qi].r_neighbors =
                        index.range_search(
                            queries.data[qi], cfg.R);
            }
        });

        cout << "IVF-PQ Total Query Time: "
             << total_query_time << " sec\n";
    }


    else {
        cerr << "Unknown algorithm: "
             << cfg.algo << endl;
        return;
    }


    // ================= SELF SEARCH =================
    if (cfg.qpath.empty()) {
        write_output_neural(data, results, cfg);
        return;
    }


    // ================= BIO OUTPUT =================
    if (cfg.type == "bio") {
        write_bio_output(results, cfg);
        return;
    }


    // ================= BRUTE OUTPUT =================
    if (cfg.algo == "brute") {
        write_output_true(queries.data, results, cfg);
        return;
    }


    // ================= EVALUATION =================
    vector<SearchResult> gt;
    double tTrueAvg_from_gt = 0.0;

    if (cfg.evalpath.empty()) {
        gt.resize(Q);

        for (int qi = 0; qi < Q; ++qi) {
            gt[qi] = brute_force_query(
                queries.data[qi], data, cfg);
            tTrueAvg_from_gt += gt[qi].time;
        }

        if (Q > 0)
            tTrueAvg_from_gt /= Q;
    }
    else {
        gt = load_ground_truth(
            cfg.evalpath,
            tTrueAvg_from_gt,
            Q);
    }

    auto metrics = evaluate(
        results,
        gt,
        cfg,
        tTrueAvg_from_gt);

    write_output(
        queries.data,
        results,
        gt,
        cfg,
        metrics);
}


// Dataset Dispatcher
void run_search(const Config& cfg)
{
    if (cfg.type == "mnist") {
        auto data = read_mnist(cfg.dpath);
        if (cfg.qpath.empty())
            run_algorithm<uint8_t>(data, data, cfg);
        else {
            auto queries = read_mnist(cfg.qpath);
            run_algorithm<uint8_t>(data, queries, cfg);
        }
    }

    else if (cfg.type == "sift") {
        auto data = read_sift(cfg.dpath);
        if (cfg.qpath.empty())
            run_algorithm<float>(data, data, cfg);
        else {
            auto queries = read_sift(cfg.qpath);
            run_algorithm<float>(data, queries, cfg);
        }
    }

    else if (cfg.type == "bio") {
        auto data = read_bio(cfg.dpath, 320);
        if (cfg.qpath.empty())
            run_algorithm<float>(data, data, cfg);
        else {
            auto queries = read_bio(cfg.qpath, 320);
            run_algorithm<float>(data, queries, cfg);
        }
    }
}


// Explicit template instantiations
template void run_algorithm<uint8_t>(
    const Dataset<uint8_t>&,
    const Dataset<uint8_t>&,
    const Config&);

template void run_algorithm<float>(
    const Dataset<float>&,
    const Dataset<float>&,
    const Config&);
