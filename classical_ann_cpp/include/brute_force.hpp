#pragma once
#include "file_parser.hpp"
#include "utils.hpp" 

using namespace std;

template <typename Tq, typename Td>
SearchResult brute_force_query(
    const vector<Tq>& query,
    const Dataset<Td>& data,
    const Config& cfg);


