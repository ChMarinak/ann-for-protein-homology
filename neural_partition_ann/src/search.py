# search.py

from typing import Dict, List, Tuple
import numpy as np
import torch
from src.models import MLPClassifier


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(x - y))


def search_neural_lsh(
    query: np.ndarray,
    model: MLPClassifier,
    dataset_gpu: torch.Tensor,
    inverted: Dict[int, List[int]],
    T: int = 5,
    N: int = 1,
    R: float = 2000.0,
    do_range: bool = True,
    device: torch.device = None,
) -> Tuple[List[int], List[float], List[int]]:

    if device is None:
        device = torch.device("cpu")

    query_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict partition probabilities
    with torch.no_grad():
        logits = model(query_tensor)
        probs = torch.softmax(logits, dim=1)

        _, top_partitions = torch.topk(probs[0], k=min(T, probs.shape[1]))
        top_partitions = top_partitions.cpu().numpy()

    # Gather candidate indices
    candidates = set()
    for partition in top_partitions:
        partition = int(partition)
        if partition in inverted:
            candidates.update(inverted[partition])

    if not candidates:
        return [], [], []

    cand_indices = np.fromiter(candidates, dtype=np.int64)

    # Compute distances on GPU
    cand_vectors = dataset_gpu[cand_indices]
    query_vec = query_tensor.squeeze(0)

    with torch.no_grad():
        dists = torch.norm(cand_vectors - query_vec, dim=1)

    dists = dists.cpu().numpy()

    # Sort
    order = np.argsort(dists)
    sorted_indices = cand_indices[order]
    sorted_dists = dists[order]

    nn_indices = sorted_indices[:N].tolist()
    nn_dists = sorted_dists[:N].tolist()

    range_indices = []
    if do_range:
        mask = sorted_dists <= R
        range_indices = sorted_indices[mask].tolist()

    return nn_indices, nn_dists, range_indices
