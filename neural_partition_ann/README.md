# Implementation of Vector Search with Neural LSH

## Overview

This project implements Neural LSH (Locality Sensitive Hashing with learned partitioning) for efficient Approximate Nearest Neighbor (ANN) search in high-dimensional spaces. The method learns to partition the dataset using a neural network, supporting multi-probe search and both range and k-NN queries.

---

## Supported Data Types

| Dataset | Description | Dimension | Data Type |
|---------|------------|-----------|-----------|
| MNIST   | 28×28 grayscale images | 784       | uint8_t  |
| SIFT    | Feature vectors         | 128       | float (L2-normalized) |
| BIO     | Protein embeddings      | 320       | float    |

---

## Implementation Features

- **Learned partitioning:** MLP classifier predicts partition assignments for fast search.
- **Multi-probe search:** Probes top-T partitions for higher recall.
- **Flexible partitioning:** Supports both KaHIP and Louvain graph partitioning.
- **Range and k-NN search:** Supports both query types.
- **Ground-truth evaluation:** Optionally loads precomputed ground truth for metrics.
- **GPU acceleration:** Uses CUDA if available.
- **Metrics:** Computes AF, Recall@N, QPS, and timing.
- **BIO support:** Handles protein embedding datasets.

---

## Data Flow

1. **Build Phase** (`nlsh_build.py` / `nlsh_build_louvain.py`):
   - Input: Dataset (MNIST/SIFT/BIO)
   - Partitioning: KaHIP or Louvain
   - Training: MLP classifier
   - Output: `model.pth`, `inverted_file.npz`

2. **Search Phase** (`nlsh_search.py`):
   - Input: Queries + Index (model + inverted file)
   - Multi-probe: Select top-T partitions from MLP
   - Range search: Find all within radius R (if enabled)
   - Output: Results + Metrics (AF, Recall, QPS)

---

## Project Architecture

```text
neural_partition_ann/
├── src/
│   ├── custom_types.py
│   ├── dataset_parser.py
│   ├── models.py
│   ├── graph_utils.py
│   ├── kahip_utils.py
│   ├── louvain_utils.py
│   ├── search.py
│   ├── evaluator.py
│   └── writer.py
├── nlsh_build.py
├── nlsh_build_louvain.py
├── nlsh_search.py
├── README.md
```

---

## Compilation & Installation

```bash
pip install kahip torch
```

---

## Program Usage Instructions

### Build Phase

#### Basic Syntax

**KaHIP Partitioning:**
```bash
python3 nlsh_build.py -d <dataset> -i <index_path> -type <mnist|sift|bio> [options]
```

**Louvain Partitioning:**
```bash
python3 nlsh_build_louvain.py -d <dataset> -i <index_path> -type <mnist|sift|bio> [options]
```

#### Required Parameters

| Flag                | Description                                   |
|---------------------|-----------------------------------------------|
| `-d <input file>`   | Dataset file (input.dat)                      |
| `-i, --index_path`  | Output directory for model and inverted file  |
| `-type <flag>`      | Data type: `mnist`, `sift`, or `bio`          |

#### Optional Parameters

**Common:**

| Flag                | Default   | Description                                 |
|---------------------|-----------|---------------------------------------------|
| `-graph <file>`     | None      | Path to precomputed k-NN graph              |
| `--knn <int>`       | 10        | Number of neighbors for k-NN graph          |
| `--epochs <int>`    | 10        | Training epochs                             |
| `--batch_size <int>`| 128       | Training batch size                         |
| `--lr <float>`      | 0.001     | Learning rate                               |
| `--layers <int>`    | 3         | Number of MLP layers                        |
| `--nodes <int>`     | 64        | Hidden nodes per MLP layer                  |
| `--seed <int>`      | 1         | Random seed                                 |

**KaHIP-specific (`nlsh_build.py`):**

| Flag                | Default   | Description                                 |
|---------------------|-----------|---------------------------------------------|
| `-m <int>`          | 100       | Number of partitions                        |
| `--imbalance <float>` | 0.03    | Allowed partition imbalance                 |
| `--kahip_mode <int>` | 2        | KaHIP mode                                  |

**Louvain-specific (`nlsh_build_louvain.py`):**

| Flag                | Default   | Description                                 |
|---------------------|-----------|---------------------------------------------|
| `--resolution <float>` | 1.0    | Louvain resolution parameter (controls number of communities) |

---

### Search Phase

#### Basic Syntax

```bash
python3 nlsh_search.py -d <dataset> -q <queries> -i <index_path> -o <output> -type <mnist|sift|bio> [options]
```

#### Required Parameters

| Flag                | Description                                   |
|---------------------|-----------------------------------------------|
| `-d <input file>`   | Dataset file (input.dat)                      |
| `-q <query file>`   | Query file (queries.dat)                      |
| `-i, --index_path`  | Directory with model.pth + inverted_file.npz  |
| `-o <output file>`  | Output file                                   |
| `-type <flag>`      | Data type: `mnist`, `sift`, or `bio`          |

#### Optional Parameters

| Flag                | Default   | Description                                 |
|---------------------|-----------|---------------------------------------------|
| `-N <int>`          | 1         | Number of nearest neighbors                 |
| `-T <int>`          | 5         | Number of partitions for multi-probe        |
| `-R <float>`        | 2000/0.5* | Range search radius                         |
| `-range <bool>`     | true      | Enable range search                         |
| `--seed <int>`      | 1         | Random seed                                 |
| `--ground_truth <file>` | None  | Precomputed ground truth file               |
| `--num_queries <int>` | None   | Limit number of queries processed           |

*Default: 2000 for MNIST, 0.5 for SIFT, 0.5 for BIO

---

## Example Commands

### Build Phase

#### MNIST index with 100 partitions (KaHIP):
```bash
python3 nlsh_build.py \
  -d mnist_train.dat \
  -i ./index_mnist \
  -type mnist \
  -m 100
```

#### SIFT index with custom k-NN graph (Louvain):
```bash
python3 nlsh_build_louvain.py \
  -d sift_base.dat \
  -i ./index_sift \
  -type sift \
  -graph sift_knn.txt \
  --resolution 1.2
```

#### BIO protein embeddings (KaHIP):
```bash
python3 nlsh_build.py \
  -d bio_base.bin \
  -i ./index_bio \
  -type bio \
  -m 50
```

---

### Search Phase

#### MNIST with 10 nearest neighbors:
```bash
python3 nlsh_search.py \
  -d mnist_train.dat \
  -q mnist_test.dat \
  -i ./index_mnist \
  -o results_mnist.txt \
  -type mnist \
  -N 10 \
  -T 10
```

#### SIFT with precomputed ground truth (Louvain index):
```bash
python3 nlsh_search.py \
  -d sift_base.dat \
  -q sift_queries.dat \
  -i ./index_sift \
  -o results_sift.txt \
  -type sift \
  -N 100 \
  -T 20 \
  --ground_truth ground_truth_sift.txt
```

---

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

## Module Responsibilities

### `nlsh_search.py`

**Functions:**
- `parse_args()`: CLI argument parsing
- `load_dataset()`: Load the required dataset using the correct type loader
- `main()`: Orchestration

### `nlsh_build.py`

**Functions:**
- `parse_args()`: CLI argument parsing
- `ensure_knn_graph()`: Create or load k-NN graph
- `main()`: Orchestrate build process

**Process:**
1. Load dataset
2. Partition with KaHIP (m partitions)
3. Train MLP classifier
4. Save: `model.pth` + `inverted_file.npz`

### `nlsh_build_louvain.py`

**Functions:**
- `parse_args()`: CLI argument parsing
- `ensure_knn_graph()`: Create or load k-NN graph
- `main()`: Orchestrate build process

**Process:**
1. Load dataset
2. Partition with Louvain
3. Train MLP classifier
4. Save: `model.pth` + `inverted_file.npz`

**Notes:**
- Louvain produces a number of communities not directly set by m — the number of communities depends on the resolution parameter.

### Library Files (`src/`)

#### `custom_types.py`
**Data Structures:**
- `Dataset`: Stores vectors with dimension
- `SearchResult`: Search results (indices, distances, time)

#### `dataset_parser.py`
**Functions:**
- `read_mnist()`: Load MNIST (uint8, 784D)
- `read_sift()`: Load SIFT (float32, 128D, L2-normalized)
- `load_ground_truth()`: Load precomputed ground truth from file

#### `models.py`
**Classes:**
- `PartitionDataset`: PyTorch Dataset wrapper
- `MLPClassifier`: 3-layer MLP for partition prediction

**Functions:**
- `train_classifier()`: Train MLP with CrossEntropy loss

#### `graph_utils.py`
**Functions:**
- `k-NN graph()`: Return the graph either generated or precomputed
- `read_knn_file()`: Load k-NN file
- `symmetrize_graph()`: Convert to symmetrized graph with weights
- `graph_to_csr()`: Convert to CSR format for KaHIP

#### `kahip_utils.py`
**Functions:**
- `partition_graph()`: Wrapper for KaHIP partitioning

#### `louvain_utils.py`
**Functions:**
- `partition_graph_louvain(graph, resolution=1.0, random_state=None)`: Run Louvain community detection on a graph

#### `search.py`
**Functions:**
- `euclidean_distance(x, y)`: Compute L2 (Euclidean) distance between two vectors
- `search_neural_lsh(query, model, dataset_gpu, inverted, T, N, R, do_range, device)`: Multi-probe search using the neural LSH model

#### `evaluator.py`
**Functions:**
- `compute_ground_truth(query, dataset, k)`: Brute-force exact k-NN search
- `compute_af(approx_dists, true_dists)`: Compute Approximation Factor (AF)
- `compute_recall(approx_indices, true_indices)`: Compute Recall@N

#### `writer.py`
**Functions:**
- `write_bio_output(output_path, results, N, avg_time, qps)`: Write output in bio format
- `write_standard_output(output_path, results, ground_truths, afs, recalls, avg_af, avg_recall, avg_approx_time, avg_true_time, qps, N)`: Write standard output with metrics

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

---

## References

- [KaHIP Graph Partitioning](https://github.com/KaHIP/KaHIP)
- [Louvain Community Detection](https://github.com/vtraag/louvain-igraph)
- [PyTorch](https://pytorch.org/)

---