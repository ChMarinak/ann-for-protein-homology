#pragma once
#include "file_parser.hpp"
#include <vector>

using namespace std;

struct Config;  // forward declaration

struct EvalMetrics {
    vector<double> AFs;
    vector<double> Recalls;
    double avg_AF = 0.0;
    double avg_Recall = 0.0;
    double QPS = 0.0;
    double tApproxAvg = 0.0;
    double tTrueAvg = 0.0;
};

EvalMetrics evaluate(
    const vector<SearchResult>& results,
    const vector<SearchResult>& gt,
    const Config& cfg,
    double tTrueAvg_from_gt
);
