# custom_types.py

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Dataset:
    data: List[np.ndarray]
    dim: int = 0

@dataclass
class SearchResult:
    indices: List[int]
    dists: List[float]
    time: float = 0.0