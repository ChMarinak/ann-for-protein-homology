# louvain_utils.py

import networkx as nx
import community.community_louvain as community_louvain
import numpy as np

def partition_graph_louvain(graph, seed=1, resolution=1.0):
    """
    Partition an undirected weighted graph using Louvain method.
    Returns a list of partition IDs (one per node) and the number of edges cut.
    graph: dict[node] -> dict[neighbor] -> weight
    """
    # Build NetworkX graph
    G = nx.Graph()
    for node, neighbors in graph.items():
        for nbr, w in neighbors.items():
            G.add_edge(node, nbr, weight=w)

    # Louvain partitioning
    np.random.seed(seed)
    partition = community_louvain.best_partition(G, random_state=seed, weight='weight', resolution=resolution)

    # Convert to blocks array (node index -> partition id)
    n = len(graph)
    blocks = np.zeros(n, dtype=np.int32)
    for node, part_id in partition.items():
        blocks[node] = part_id

    # Edge cut estimation
    edgecut = 0
    for u, v, w in G.edges(data='weight'):
        if blocks[u] != blocks[v]:
            edgecut += w

    return blocks, edgecut

