import argparse
import os
import torch
import numpy as np

from src.file_parser import read_mnist, read_sift, read_bio
from src.graph_utils import ensure_knn_graph, read_knn_file, symmetrize_graph
from src.louvain_utils import partition_graph_louvain
from src.models import train_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Build Neural LSH index using Louvain")

    parser.add_argument("-d", required=True, help="Input dataset file (input.dat)")
    parser.add_argument("-i", "--index_path", required=True, help="Output index directory")
    parser.add_argument("-type", required=True, choices=["sift", "mnist", "bio"],
                        help="Type of dataset (sift, mnist, bio)")
    parser.add_argument("-graph", help="Optional path to precomputed k-NN graph file")
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution parameter")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    if args.type == "sift":
        X = read_sift(args.d)
    elif args.type == "mnist":
        X = read_mnist(args.d)
    elif args.type == "bio":
        X = read_bio(args.d)

    n = len(X.data)
    d = X.dim

    X_np = np.vstack(X.data)

    # Build or load k-NN graph
    search_executable = "./search"
    graph_file = ensure_knn_graph(search_executable, args.d, args.type, args.knn, args.graph)

    knn = read_knn_file(graph_file, max_neighbors=args.knn)

    graph = symmetrize_graph(knn)

    # Louvain partitioning
    print(f"Running Louvain with resolution={args.resolution}")
    blocks, edgecut = partition_graph_louvain(graph, seed=args.seed, resolution=args.resolution)
    print(f"Partition completed: partitions = {max(blocks)+1}")

    # Train MLP classifier
    print("Training PyTorch classifier...")
    model = train_classifier(
        X_np,
        blocks,
        m=max(blocks)+1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.nodes,
    )

    # Save model and inverted file
    os.makedirs(args.index_path, exist_ok=True)
    model_path = os.path.join(args.index_path, "model.pth")
    invfile_path = os.path.join(args.index_path, "inverted_file.npz")

    torch.save({
        "model_state": model.state_dict(),
        "d": d,
        "m": max(blocks)+1,
        "seed": args.seed,
        "layers": args.layers,
        "nodes": args.nodes
    }, model_path)

    inverted = {}
    for idx, part in enumerate(blocks):
        inverted.setdefault(part, []).append(idx)
    np.savez_compressed(invfile_path, **{str(k): np.array(v) for k, v in inverted.items()})

    print("Build completed.")


if __name__ == "__main__":
    main()