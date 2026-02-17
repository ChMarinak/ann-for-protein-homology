# Implementation of Approximate Nearest Neighbor Search Algorithms

## Overview

This project implements multiple Approximate Nearest Neighbor (ANN) search algorithms for high-dimensional vector spaces. The objective is to efficiently retrieve the top-N nearest neighbors of query vectors while balancing speed and accuracy.

The following algorithms are implemented:

- LSH (Locality Sensitive Hashing)
- Hypercube
- IVFFlat (Inverted File + Flat Search)
- IVFPQ (Inverted File + Product Quantization)
- Brute Force (Exact search for ground truth)

---

## Supported Data Types

| Dataset | Description | Dimension | Data Type |
|---------|------------|-----------|-----------|
| MNIST   | 28×28 grayscale images | 784       | uint8_t  |
| SIFT    | Feature vectors         | 128       | float (L2-normalized) |
| BIO     | Protein embeddings      | 320       | float    |

---

## Implementation Features

- **Template-based design:** Supports both `uint8_t` (MNIST) and `float` (SIFT/BIO) data without duplicating code.  
- **Exact ground-truth computation:** Brute-force linear search is used to validate ANN results and compute true distances.  
- **Accurate timing measurements:** Records both query execution time and index build/training time separately.  
- **Modular architecture:** Clear separation of responsibilities across modules (I/O, parsing, evaluation, output, algorithms).  
- **Reproducibility:** Random seed parameter ensures deterministic results for experiments.  
- **Flexible output modes:** Supports standard ANN evaluation, self-search graphs for Neural-LSH / KaHIP, and BIO embedding export.  
- **Cluster quality metrics:** IVF-based algorithms optionally compute silhouette scores for evaluating clustering quality.  
- **Ground-truth evaluation input:** `-e <ground_truth_file>` allows ANN results to be compared against a precomputed ground truth.  
- **Self-search mode:** If `-q` is omitted, the program runs in self-search mode and outputs a Neural-LSH / KaHIP-style k-NN graph.

---

## Project Architecture
```text
Project Architecture
classical_ann_cpp/
├── include/
│ ├── brute_force.hpp
│ ├── evaluator.hpp
│ ├── file_parser.hpp
│ ├── hypercube.hpp
│ ├── ivfflat.hpp
│ ├── ivfpq.hpp
│ ├── lsh.hpp
│ ├── output_writer.hpp
│ └── utils.hpp
├── src/
│ ├── brute_force.cpp
│ ├── evaluator.cpp
│ ├── file_parser.cpp
│ ├── hypercube.cpp
│ ├── ivfflat.cpp
│ ├── ivfpq.cpp
│ ├── lsh.cpp
│ ├── main.cpp
│ ├── output_writer.cpp
│ └── utils.cpp
├── Makefile
├── README.md
└── search
```

---

## Module Responsibilities

### 1. main.cpp

**Role:** Program entry point.

Responsibilities:

- Creates `Config` object
- Parses command-line arguments
- Calls `run_search(cfg)`
- Handles exit codes

---

### 2. utils (Orchestration Layer)

**Role:** Controls overall program execution flow.

Responsibilities:

- Parse command-line arguments
- Select dataset type
- Dispatch selected ANN algorithm
- Trigger evaluation
- Trigger output writing

Key Functions:

- `parse_arguments()`
- `run_search()`
- `run_algorithm()`

---

### 3. file_parser (I/O Layer)

**Role:** Dataset loading and ground-truth parsing.

Functions:

- `read_mnist()` – Loads IDX format
- `read_sift()` – Loads binary SIFT and applies L2 normalization
- `read_bio()` – Loads 320-dimensional float embeddings
- `load_ground_truth()` – Parses brute-force results

---

### 4. evaluator (Metrics Layer)

**Role:** Computes evaluation metrics.

Metrics:

- AF (Approximation Factor)
- Recall@N
- QPS (Queries per Second)
- Average approximate time
- Average true time

Function:

- `evaluate()`

---

### 5. output_writer (Export Layer)

**Role:** Writes results to file in different formats.

Supported Output Modes:

#### Standard ANN Evaluation Format

Includes:

- Per-query results
- AF & Recall per query
- Range search results (if enabled)
- Overall metrics

#### Brute Force Output

Includes:

- True nearest neighbors
- Exact distances
- True timing

#### Neural-LSH / KaHIP Mode (Self-Search)

If `-q` is not provided:

- Dataset is used as queries (self-search)
- Outputs k-NN graph format:
<point_index> <neighbor1> <neighbor2> ...


- No evaluation metrics
- No self-loops

#### BIO Mode

When `-t bio`:

Outputs:

- Time per query
- QPS
- Top-N neighbors with distances

Designed for downstream BLAST-style evaluation.

---

## Algorithms

### 1. Brute Force

Exact linear scan over all points.

Used for:

- Ground truth
- AF and Recall computation
- Validation

Distance metric: Euclidean

---

### 2. LSH (Locality Sensitive Hashing)

- Random projection hashing
- L independent hash tables
- Multi-table search strategy

Tradeoff: Very fast, moderate recall

---

### 3. Hypercube

- Random projections → binary hash
- Hamming distance probing
- Multi-probe strategy

Tradeoff: Higher accuracy than LSH, slightly slower

---

### 4. IVFFlat

- K-means clustering
- Inverted file structure
- Searches only nearest centroids (`nprobe`)

Tradeoff: Strong balance between speed and recall

---

### 5. IVFPQ

- IVF coarse clustering
- Product Quantization encoding
- Compressed vector representation
- Fast approximate distance lookup

Tradeoff: Very fast for large datasets, small accuracy loss

---

## Compilation

### Clean
```bash
make clean
```

### Compile
```bash
make
```

### Full rebuild
```bash
make clean && make
```

### Output
- `build/`: Folder with object files (.o) and dependency files (.d)
- `search`: Final executable

---

## Program Usage Instructions

### Basic Syntax

```bash
./search -d <dataset> -q <queries> -o <output> -t <type> [algorithm flags] [parameters]
```

### Parameters

#### Required Parameters

| Flag | Type | Description |
|------|------|-------------|
| `-d <input file>` | string | Path to dataset (base vectors) |
| `-q <query file>` | string | Path to query dataset |
| `-o <output file>` | string | Path for output results |
| `-t <flag>` | string | Data type: `mnist` or `sift` |

#### Optional Evaluation Parameter

| Flag | Type | Description |
|------|------|-------------|
| `-e <ground_truth_file>` | string | Path to precomputed ground truth file for evaluation |

#### Common Parameters for All Algorithms

| Flag | Type | Description |
|------|------|-------------|
| `-N <int>` | integer | Number of nearest neighbors to find |
| `-R <float>` | float | Radius for range search |
| `-r <true\|false>` | boolean | Enable range search |
| `-s <int>` | integer | Random seed for reproducibility |

#### LSH Parameters

| Flag | Type | Description |
|------|------|-------------|
| `-k <int>` | integer | Number of hash functions per table |
| `-L <int>` | integer | Number of independent hash tables |
| `-w <float>` | float | Bucket width for hashing |
| `-lsh` | - | Algorithm flag |

#### Hypercube Parameters

| Flag | Type | Description |
|------|------|-------------|
| `-P <int>` | integer | Number of random projections (d) |
| `-w <float>` | float | Window size for projections |
| `-M <int>` | integer | Max candidates in probing |
| `-p <int>` | integer | Number of hypercube probes |
| `-hypercube` | - | Algorithm flag |

#### IVFFlat Parameters

| Flag | Type | Description |
|------|------|-------------|
| `-c <int>` | integer | Number of K-means clusters |
| `-n <int>` | integer | Number of probes (inverted lists) |
| `-ivfflat` | - | Algorithm flag for IVFFlat |

#### IVFPQ Parameters

| Flag | Type | Description |
|------|------|-------------|
| `-c <int>` | integer | Number of K-means clusters |
| `-n <int>` | integer | Number of probes (inverted lists) |
| `-M <int>` | integer | Number of product quantizers |
| `-b <int>` | integer | Bits per quantizer |
| `-ivfpq` | - | Algorithm flag for IVFPQ |

---

### Example Commands

#### LSH (MNIST)
```bash
./search -d train.idx3-ubyte -q test.idx3-ubyte \
         -k 4 -L 5 -w 4.0 \
         -o lsh_output.txt -N 10 -t mnist -lsh
```

#### Hypercube (MNIST) with Ground Truth
```bash
./search -d train.idx3-ubyte -q test.idx3-ubyte \
         -e ground_truth.txt \
         -P 14 -M 10 -p 2 \
         -o hc_output.txt -N 10 -t mnist -hypercube
```

#### IVFPQ (SIFT)
```bash
./search -d data/mnist/train-images.idx3-ubyte -q data/mnist/test-images.idx3-ubyte \
         -c 100 -n 3 -o results/ivfflat_output.txt -N 10 -R 2000.0 \
         -t mnist -r false -ivfpq
```

#### IVFFlat (BIO)
```bash
./search -d bio_base.bin -q bio_query.bin \
         -c 100 -n 5 -o bio_output.txt \
         -N 10 -t bio -ivfflat
```

#### Self-Search Mode (Neural-LSH Graph)
```bash
./search -d sift_base.fvecs -o knn_graph.txt \
         -N 10 -t sift 
```

## Output Format

The program produces an output file with the following structure:

```
algorithm_name
Query: 0
Nearest neighbor-1: <index>
distanceApproximate: <distance>
distanceTrue: <ground_truth_distance>
Average AF: <value>
Recall@N: <value>
QPS: <queries_per_second>
tApproximateAverage: <time>
tTrueAverage: <ground_truth_time>

...

Overall Average AF: <value>
Overall Recall@N: <value>
Overall QPS: <value>
Overall tApproximateAverage: <value>
Overall tTrueAverage: <value>
```

### Evaluation Metrics

- **AF (Approximation Factor):** Ratio of approximate distance to true distance
- **Recall@N:** Percentage (%) of the N nearest neighbors found correctly
- **QPS:** Queries per second (throughput)
- **tApproximate:** Algorithm execution time (average)
- **tTrue:** Brute force execution time for ground truth

---

## Tested Datasets

### MNIST
- Training set: 60,000 images × 784 dimensions (28×28 pixels)
- Test/Query set: 10,000 images × 784 dimensions
- Data type: uint8 (0-255)

### SIFT
- Base set: 1,000,000 vectors × 128 dimensions
- Query set: 10,000 vectors × 128 dimensions
- Data type: float32 (L2-normalized)

### BIO
- Base set: protein embeddings, 50,000 vectors × 320 dimensions
- Query set: protein embeddings, 12 × 320 dimensions
- Data type: float32
- Used for approximate nearest neighbor search in biological sequence embeddings
