#pragma once
#include <vector>
#include <string>

#include "file_parser.hpp"
#include "evaluator.hpp"
#include "utils.hpp"

using namespace std;

// Full evaluation output 
template <typename T>
void write_output(
    const vector<T>& queries,
    const vector<SearchResult>& results,
    const vector<SearchResult>& gt,
    const Config& cfg,
    const EvalMetrics& metrics
);


// Neural-LSH / KaHIP graph writer
template <typename T>
void write_output_neural(
    const Dataset<T>& data,
    const vector<SearchResult>& results,
    const Config& cfg
);


// BIO format writer
void write_bio_output(
    const vector<SearchResult>& results,
    const Config& cfg
);


// Brute-only output
template <typename T>
void write_output_true(
    const vector<T>& queries,
    const vector<SearchResult>& results,
    const Config& cfg
);
