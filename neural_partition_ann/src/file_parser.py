# file_parser.py

import struct
import numpy as np
from pathlib import Path
import torch
from src.models import MLPClassifier
from typing import List, Union, Tuple, Dict

from src.custom_types import Dataset, SearchResult


def read_mnist(path: Union[str, Path]) -> Dataset:
    """Read MNIST binary file and return Dataset with uint8 vectors."""
    ds = Dataset(data=[])
    path = Path(path)

    with path.open('rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Invalid MNIST magic number {magic_number}")

        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        dim = num_rows * num_cols
        ds.dim = dim

        for _ in range(num_images):
            image = np.frombuffer(f.read(dim), dtype=np.uint8)
            ds.data.append(image)

    return ds


def read_sift(path: Union[str, Path]) -> Dataset:
    """Read SIFT binary file and return Dataset with float vectors (L2 normalized)."""
    ds = Dataset(data=[])
    path = Path(path)

    with path.open('rb') as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                break
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            ds.data.append(vec)
            ds.dim = dim

    return ds


def read_bio(path: Union[str, Path]) -> Dataset:
    """Read BIO binary file and return Dataset with float vectors."""
    ds = Dataset(data=[])
    path = Path(path)
    dim: int = 320 # Fixed dimension for BIO dataset
    data = np.fromfile(path, dtype=np.float32)

    vectors = data.reshape(-1, dim)

    for v in vectors:
        ds.data.append(v)

    ds.dim = dim
    return ds


def load_ground_truth(filename: Union[str, Path], max_queries: int = -1) -> Tuple[List[SearchResult], float]:
    """Load ground truth file into list of SearchResult objects and overall tTrueAverage."""
    filename = Path(filename)
    gt: List[SearchResult] = []
    overall_tTrue = 0.0
    current = SearchResult(indices=[], dists=[], time=0.0)
    query_count = 0

    with filename.open('r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Query:"):
                if current.indices:
                    gt.append(current)
                    query_count += 1
                    if 0 < max_queries <= query_count:
                        break
                    current = SearchResult(indices=[], dists=[], time=0.0)
            elif line.startswith("Nearest neighbor-"):
                idx = int(line.split(":")[1].strip())
                current.indices.append(idx)
            elif line.startswith("distanceTrue:"):
                dist = float(line.split(":")[1].strip())
                current.dists.append(dist)
            elif line.startswith("tTrueAverage:"):
                current.time = float(line.split(":")[1].strip())
            elif line.startswith("Overall tTrueAverage:"):
                overall_tTrue = float(line.split(":")[1].strip())

    if current.indices:
        gt.append(current)

    return gt, overall_tTrue


def load_model(model_path: str, device: torch.device) -> Tuple[MLPClassifier, int, int]:
    """Load trained MLP model from checkpoint (reconstruct exact architecture)."""

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    d = checkpoint['d']
    m = checkpoint['m']

    # Saved architecture params, fallback to defaults
    nodes = checkpoint.get('nodes', 64)
    layers = checkpoint.get('layers', 3)

    # Construct the model
    model = MLPClassifier(d_in=d, n_out=m, hidden=nodes, layers=layers).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, d, m


def load_inverted_file(invfile_path: str) -> Dict[int, List[int]]:
    """Load inverted file (mapping from partition -> list of data indices)."""
    data = np.load(invfile_path, allow_pickle=True)
    inverted = {}
    for key in data.files:
        try:
            inverted[int(key)] = list(data[key])
        except Exception as e:
            print(f"Warning: could not parse key {key}: {e}")
    return inverted