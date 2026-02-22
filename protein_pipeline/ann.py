# ann.py

import os
import subprocess
import time
from pathlib import Path


# Classic ANN parameters
ALGO_PARAMS = {
    "lsh": {
        "-k": 5,
        "-L": 6,
        "-w": 1,
    },
    "hypercube": {
        "-P": 3,
        "-w": 4,
        "-M": 6000,
        "-p": 1,
    },
    "ivfflat": {
        "-c": 750,
        "-n": 35,
    },
    "ivfpq": {
        "-c": 700,
        "-n": 35,
        "-M": 64,
        "-b": 8,
    },
}


# Neural LSH parameters
NLSH_BUILD = {
    "kahip": {
        "script": "neural_partition_ann/nlsh_build.py",
        "extra": [
            "--knn", "30",
            "-m", "400",
            "--imbalance", "0.5",
            "--kahip_mode", "0",
            "--epochs", "20",
        ],
    },
    "louvain": {
        "script": "neural_partition_ann/nlsh_build_louvain.py",
        "extra": [
            "--knn", "20",
            "--resolution", "10",
            "--epochs", "20",
        ],
    },
}

NLSH_SEARCH_EXTRA = ["-T", "5"]


def embed_queries(query_fasta, out):
    """
    Calls protein_embed.py to embed query FASTA sequences
    """
    if os.path.exists(out):
        return
    
    subprocess.run([
        "python3", "protein_embed.py",
        "-i", query_fasta,
        "-o", out
    ], check=True)
    print(f"Embedding complete. Output saved to '{out}'.")


# ANN runner
def run_ann(method, db_vecs, q_vecs, out_file, N):
    """
    Runs classic ANN or Neural LSH transparently based on `method`.
    """

    # Neural LSH
    if method in ("kahip", "louvain"):
        # Where the precomputed graph would be
        graph_path = Path("data/graph.txt")

        # Where the nlsh_index should go
        nlsh_index_root="nlsh_index"
        index_dir = Path(nlsh_index_root) / method
        index_dir.mkdir(parents=True, exist_ok=True)

        build_cfg = NLSH_BUILD[method]

        # Base build command (no graph yet)
        build_cmd = [
            "python3", build_cfg["script"],
            "-d", db_vecs,
            "-i", str(index_dir),
            "-type", "bio",
        ] + build_cfg["extra"]

        # Add graph if it exists
        if graph_path.exists():
            build_cmd += ["-graph", str(graph_path)]

        subprocess.run(build_cmd, check=True)

        # Search
        search_cmd = [
            "python3", "neural_partition_ann/nlsh_search.py",
            "-d", db_vecs,
            "-q", q_vecs,
            "-i", str(index_dir),
            "-o", out_file,
            "-type", "bio",
            "-N", str(N),    
        ] + NLSH_SEARCH_EXTRA

        start = time.time()
        subprocess.run(search_cmd, check=True)
        return time.time() - start

    # Classic ANN
    base_cmds = {
        "lsh":       ["./classical_ann_cpp/search", "-d", db_vecs, "-q", q_vecs, "-t", "bio", "-lsh"],
        "hypercube": ["./classical_ann_cpp/search", "-d", db_vecs, "-q", q_vecs, "-t", "bio", "-hypercube"],
        "ivfflat":   ["./classical_ann_cpp/search", "-d", db_vecs, "-q", q_vecs, "-t", "bio", "-ivfflat"],
        "ivfpq":     ["./classical_ann_cpp/search", "-d", db_vecs, "-q", q_vecs, "-t", "bio", "-ivfpq"],
    }

    if method not in base_cmds:
        raise ValueError(f"Unknown ANN method: {method}")

    cmd = base_cmds[method][:]

    for flag, value in ALGO_PARAMS.get(method, {}).items():
        cmd.extend([flag, str(value)])

    cmd.extend(["-o", out_file, "-N", str(N)])

    start = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - start


def parse_ann(path):
    ann_outputs = []
    current_query = []

    time_per_query = None
    qps = None

    with open(path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Time_per_query"):
                time_per_query = float(line.split()[1])

            elif line.startswith("QPS"):
                qps = float(line.split()[1])

            elif line.startswith("QUERY"):
                if current_query:
                    ann_outputs.append(current_query)
                current_query = []

            elif line and line[0].isdigit():
                _, idx, dist = line.split()
                current_query.append((int(idx), float(dist)))

        if current_query:
            ann_outputs.append(current_query)

    return {
        "results": ann_outputs,
        "time_per_query": time_per_query,
        "qps": qps,
    }
