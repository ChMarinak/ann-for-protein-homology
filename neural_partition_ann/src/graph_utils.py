# graph_utils.py

import os
import subprocess
from typing import List, Tuple, Dict
import numpy as np


def ensure_knn_graph(search_executable, dataset_path, data_type, knn, graph_path=None):
    """
    Returns the path to a k-NN graph file.
    If graph_path is given and exists, uses it.
    Otherwise generates the graph using the search executable .
    """
    if graph_path is not None and os.path.exists(graph_path):
        return graph_path

    # If not provided or file missing, generate
    graph_file = graph_path or f"graph_{data_type}_{knn}.txt"
    print(f"Generating k-NN graph using C++ search executable...")
    cmd = [
        search_executable,
        "-d", dataset_path,
        "-o", graph_file,
        "-t", data_type,
        "-c", "50",
        "-n", "5",
        "-ivfflat",
        "-N", str(knn+1)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("k-NN graph generated at:", graph_file)
    return graph_file


def read_knn_file(filename: str, max_neighbors: int = None) -> Dict[int, List[int]]:
    """
    Read a kNN file and return a dict: node -> list of neighbors.
    If max_neighbors is set, truncate each neighbor list to at most that length.
    """
    knn = {}
    with open(filename, "r") as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            node = parts[0]
            neighbors = parts[1:]
            if max_neighbors is not None and len(neighbors) > max_neighbors:
                neighbors = neighbors[:max_neighbors]
            knn[node] = neighbors
    return knn


def symmetrize_graph(knn: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    """
    Convert directed neighbor lists into a symmetrized graph with weights:
    weight = 2 if mutual, 1 otherwise
    """
    graph = {}
    for node, neighbors in knn.items():
        graph.setdefault(node, {})
        for nbr in neighbors:
            if nbr not in graph:
                graph[nbr] = {}
            # check if mutual
            if node in knn.get(nbr, []):
                graph[node][nbr] = 2
                graph[nbr][node] = 2
            else:
                # only set if not already set to 2
                if graph[node].get(nbr, 0) < 2:
                    graph[node][nbr] = 1
                if graph[nbr].get(node, 0) < 2:
                    graph[nbr][node] = 1
    return graph


def graph_to_csr(graph: Dict[int, Dict[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert symmetrized graph to CSR format for KaHIP
    """
    n = len(graph)
    vwgt = np.ones(n, dtype=np.int32)
    xadj = np.zeros(n + 1, dtype=np.int32)
    
    adjncy_list = []
    adjcwgt_list = []

    for i in range(n):
        neighbors = list(graph[i].keys())
        weights = [graph[i][nbr] for nbr in neighbors]
        adjncy_list.extend(neighbors)
        adjcwgt_list.extend(weights)
        xadj[i + 1] = xadj[i] + len(neighbors)
    
    adjncy = np.array(adjncy_list, dtype=np.int32)
    adjcwgt = np.array(adjcwgt_list, dtype=np.int32)

    return vwgt, xadj, adjncy, adjcwgt


def save_csr(filename_prefix: str, vwgt, xadj, adjncy, adjcwgt):
    """Optional: save arrays as npy"""
    np.save(f"{filename_prefix}_vwgt.npy", vwgt)
    np.save(f"{filename_prefix}_xadj.npy", xadj)
    np.save(f"{filename_prefix}_adjncy.npy", adjncy)
    np.save(f"{filename_prefix}_adjcwgt.npy", adjcwgt)
