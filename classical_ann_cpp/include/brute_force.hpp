#pragma once
#include "file_parser.hpp"
#include "utils.hpp" 

using namespace std;

template <typename T>
void write_output_true(
    const Dataset<T>& queries,
    const vector<SearchResult>& true_res,
    const Config& cfg);

template <typename Tq, typename Td>
SearchResult brute_force_query(
    const vector<Tq>& query,
    const Dataset<Td>& data,
    const Config& cfg);


