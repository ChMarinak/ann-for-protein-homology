#include "file_parser.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;

static int32_t reverse_int(int32_t i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int32_t)c1 << 24) +
           ((int32_t)c2 << 16) +
           ((int32_t)c3 << 8) +
           c4;
}

Dataset<uint8_t> read_mnist(const string& path) {
    Dataset<uint8_t> ds;
    ifstream file(path, ios::binary);

    if (!file.is_open()) {
        cerr << "Cannot open MNIST file: " << path << "\n";
        return ds;
    }

    int32_t magic_number, num_images, num_rows, num_cols;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    magic_number = reverse_int(magic_number);
    num_images   = reverse_int(num_images);
    num_rows     = reverse_int(num_rows);
    num_cols     = reverse_int(num_cols);

    ds.dim = num_rows * num_cols;
    ds.data.reserve(num_images);

    for (int i = 0; i < num_images; ++i) {
        vector<uint8_t> image(ds.dim);
        file.read(reinterpret_cast<char*>(image.data()), ds.dim);
        ds.data.push_back(move(image));
    }

    return ds;
}

Dataset<float> read_sift(const string& path) {
    Dataset<float> ds;
    ifstream file(path, ios::binary);

    if (!file.is_open()) {
        cerr << "Cannot open SIFT file: " << path << "\n";
        return ds;
    }

    while (true) {
        int dim;
        if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) break;

        vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file) break;

        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = sqrt(norm);
        if (norm > 0)
            for (float& v : vec) v /= norm;

        ds.data.push_back(move(vec));
        ds.dim = dim;
    }

    return ds;
}

Dataset<float> read_bio(const string& path, int dim) {
    Dataset<float> ds;
    ds.dim = dim;

    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open BIO embedding file: " << path << "\n";
        return ds;
    }

    while (true) {
        vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()),
                  dim * sizeof(float));
        if (!file) break;
        ds.data.push_back(move(vec));
    }

    return ds;
}

vector<SearchResult> load_ground_truth(
    const string& filename,
    double& overall_tTrue,
    int max_queries
) {
    ifstream fin(filename);
    vector<SearchResult> gt;
    overall_tTrue = 0.0;

    if (!fin.is_open()) {
        cerr << "Error opening ground truth file: " << filename << endl;
        return gt;
    }

    string line;
    SearchResult current;
    int query_count = 0;

    while (getline(fin, line)) {
        if (line.rfind("Query:", 0) == 0) {
            if (!current.indices.empty()) {
                gt.push_back(current);
                query_count++;
                if (max_queries > 0 && query_count >= max_queries) break;
                current = SearchResult();
            }
        }
        else if (line.rfind("Nearest neighbor-", 0) == 0) {
            int idx;
            istringstream(line.substr(line.find(":") + 1)) >> idx;
            current.indices.push_back(idx);
        }
        else if (line.rfind("distanceTrue:", 0) == 0) {
            float dist;
            istringstream(line.substr(line.find(":") + 1)) >> dist;
            current.dists.push_back(dist);
        }
        else if (line.rfind("tTrueAverage:", 0) == 0) {
            istringstream(line.substr(line.find(":") + 1)) >> current.time;
        }
        else if (line.rfind("Overall tTrueAverage:", 0) == 0) {
            istringstream(line.substr(line.find(":") + 1)) >> overall_tTrue;
        }
    }

    if (!current.indices.empty())
        gt.push_back(current);

    return gt;
}
