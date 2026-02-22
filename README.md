# Remote Protein Homology Search with Approximate Methods & ESM-2

## Overview

This project aims to identify remote homologs of proteins—proteins that share similar 3D structure and function despite low sequence similarity ("Twilight Zone", <30%). The system leverages ESM-2 embeddings and five Approximate Nearest Neighbor (ANN) algorithms, benchmarking their results against BLAST and evaluating biological relevance using UniProt/SwissProt annotations.

---

## Supported Data Types

| Dataset | Description            | Dimension | Data Type |
|---------|-----------------------|-----------|-----------|
| BIO     | Protein embeddings    | 320       | float32   |

---

## Implementation Features

- **ESM-2 embeddings:** Uses facebook/esm2_t6_8M_UR50D for protein representation (320D).
- **Five ANN algorithms:** Euclidean LSH, Hypercube, IVF-Flat, IVF-PQ, Neural LSH.
- **BLAST integration:** Local alignment as a reference.
- **Biological evaluation:** Uses UniProt/SwissProt annotations (Pfam, GO, EC).
- **Structured output:** Two-level reporting (summary & detailed).
- **Batch processing:** Efficient handling of large datasets.
- **GPU acceleration:** CUDA support where available.

---

## Project Architecture

```text
ann-for-protein-homology/
├── protein_embed.py
├── protein_search.py
├── protein_pipeline/
│   ├── ann.py
│   ├── blast.py
│   ├── report.py
│   └── bio_comment.py
├── classical_ann_cpp/
├── neural_partition_ann/
├── README.md
```

---

## Installation

```bash
pip install kahip fair-esm biopython
apt-get update
apt-get install -y ncbi-blast+
```

---

## Program Usage Instructions

### 1. Embedding Generation (ESM-2)

**Basic Syntax:**
```bash
python3 protein_embed.py -i <input_fasta> -o <output_vectors>
```

**Required Parameters:**

| Flag | Description |
|------|-------------|
| `-i` | Input FASTA file with proteins (e.g., swissprot.fasta) |
| `-o` | Output file for vectors (e.g., protein_vectors.dat)    |

**Example:**
```bash
python3 protein_embed.py -i swissprot.fasta -o protein_vectors.dat
```

---

### 2. Search Benchmark (5 ANN Methods + BLAST)

**Basic Syntax:**
```bash
python3 protein_search.py -d <vectors_file> -q <query_fasta> -o <output_file> -method <all|lsh|hypercube|ivf|neural>
```

**Required Parameters:**

| Flag | Description |
|------|-------------|
| `-d` | Vectors file (protein_vectors.dat)      |
| `-q` | Query FASTA file (e.g., targets.fasta)  |
| `-o` | Output results file                     |

**Example:**
```bash
python3 protein_search.py \
  -d protein_vectors.dat \
  -q targets.fasta \
  -o results.txt \
  -method all \
  -N 50
```

---

## Output Format

### Level 1: Method Summary

```
Query Protein: <ID>
N = 50 (Top-N list size for Recall@N)

----------------------------------------------------------------------
Method         | Time/query (s) | QPS | Recall@N vs BLAST Top-N
----------------------------------------------------------------------
Euclidean LSH  | 0.020          | 50  | 0.92
Hypercube      | 0.030          | 33  | 0.88
IVF-Flat       | 0.008          | 125 | 0.93
IVF-PQ         | 0.005          | 200 | 0.90
Neural LSH     | 0.010          | 100 | 0.95
BLAST (Ref)    | 1.500          | 0.7 | 1.00
----------------------------------------------------------------------
```

### Level 2: Top-N Neighbors per Method

```
Method: Euclidean LSH
Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment
-----------------------------------------------------------------------------
1    | <Prot_A>    | 0.15    | 22%            | Yes             | Remote homolog? (shared Pfam domain)
2    | <Prot_D>    | 0.16    | 19%            | No              | Possible false positive
...
```

---

## Module Responsibilities

### Main Scripts

| File                | Description                                      |
|---------------------|--------------------------------------------------|
| `protein_embed.py`  | Generates ESM-2 embeddings for proteins          |
| `protein_search.py` | Benchmarks 5 ANN methods and BLAST, outputs report|

### Library Files (`protein_pipeline/`)

| File         | Description                                              | Key Functions                                    |
|--------------|----------------------------------------------------------|--------------------------------------------------|
| `ann.py`     | Wrapper for all ANN methods (LSH, Hypercube, IVF, Neural)| `build_index()`, `search()`, `evaluate()`        |
| `blast.py`   | BLAST execution and result parsing                       | `run_blast()`, `parse_blast_results()`           |
| `report.py`  | Report generation & biological evaluation                | `generate_summary_report()`, `annotate_neighbors()`|
| `bio_comment.py` | Biological annotation and classification             | `bio_comment()`, `load_pfam_map()`, `load_go_map()`|

### ANN Modules

Contain custom implementations of the ANN algorithms, adapted for protein embeddings. They are seperated into the `classical_ann_cpp/` and `neural_partition_ann/` directories

---

## Evaluation Metrics

- **Recall@N:** Percentage of ANN Top-N results also in BLAST Top-N.
- **QPS (Queries Per Second):** Search speed.
- **L2 Distance:** Euclidean distance in embedding space.

---

## Tested Datasets

- **BIO:** Protein embeddings, e.g., SwissProt, 50,000 vectors × 320D (base), 12 × 320D (queries).

---

## Data Flow

1. **Embedding Phase (`protein_embed.py`):**
   - Input: FASTA (e.g., swissprot.fasta)
   - Processing: ESM-2 mean pooling
   - Output: `protein_vectors.dat` (N × 320)

2. **Search Phase (`protein_search.py`):**
   - Input: Vectors + Queries (FASTA)
   - Indexing: 5 ANN indices
   - Search: k-NN for each query
   - BLAST: Parallel execution
   - Output: Results + Metrics

3. **Biological Evaluation:**
   - UniProt annotation retrieval (Pfam, GO, EC)
   - Remote homolog identification (low BLAST identity + shared annotations)
   - False positive classification

---

## References

- [ESM-2 Protein Language Model](https://github.com/facebookresearch/esm)
- [NCBI BLAST+](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
- [UniProt/SwissProt](https://www.uniprot.org/)