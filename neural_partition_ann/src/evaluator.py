# evaluator.py

from typing import List, Tuple
import numpy as np
from src.search import euclidean_distance


def compute_ground_truth(
    query: np.ndarray,
    dataset: np.ndarray,
    k: int = 1
) -> Tuple[List[int], List[float]]:
    """Brute-force exact k-NN."""
    distances = []

    for idx, point in enumerate(dataset):
        dist = euclidean_distance(query, point)
        distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])

    gt_indices = [idx for idx, _ in distances[:k]]
    gt_dists = [dist for _, dist in distances[:k]]

    return gt_indices, gt_dists


def compute_af(
    approx_dists: List[float],
    true_dists: List[float]
) -> float:
    """Approximation Factor."""
    if not approx_dists or not true_dists:
        return 0.0

    ratios = []
    for a, t in zip(approx_dists, true_dists):
        if t > 0:
            ratios.append(a / t)

    return float(np.mean(ratios)) if ratios else 0.0


def compute_recall(
    approx_indices: List[int],
    true_indices: List[int]
) -> float:
    """Recall@N."""
    if not approx_indices or not true_indices:
        return 0.0

    n = len(approx_indices)

    approx_set = set(approx_indices)
    true_set = set(true_indices[:n])

    return len(approx_set & true_set) / float(n)
