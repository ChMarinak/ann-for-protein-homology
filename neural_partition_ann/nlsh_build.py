import argparse
import os
import torch
import numpy as np

from src.file_parser import read_mnist, read_sift, read_bio
from src.graph_utils import ensure_knn_graph, read_knn_file, symmetrize_graph, graph_to_csr
from src.kahip_utils import partition_graph
from src.models import train_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Build Neural LSH index")

    parser.add_argument("-d", required=True, help="Input dataset file (input.dat)")
    parser.add_argument("-i", "--index_path", required=True,
                        help="Output index directory (model + inverted file)")
    parser.add_argument("-type", required=True, choices=["sift", "mnist", "bio"],
                        help="Type of dataset (sift, mnist, bio)")
    parser.add_argument("-graph", help="Optional path to precomputed k-NN graph file")

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("-m", type=int, default=100, help="Number of partitions")
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
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

    # Convert to numpy matrix for training make mnist float32 to train on
    X_np = np.vstack(X.data).astype(np.float32)

    # Build or load k-NN graph
    search_executable = "./search" 
    graph_file = ensure_knn_graph(
        search_executable,
        args.d,
        args.type,
        args.knn,
        graph_path=args.graph
    )

    knn = read_knn_file(graph_file, max_neighbors=args.knn)

    # Symmetrize graph and convert to CSR
    graph = symmetrize_graph(knn)
    vwgt, xadj, adjncy, adjcwgt = graph_to_csr(graph)

    # KaHIP partitioning
    print(f"Running KaHIP with m={args.m}, imbalance={args.imbalance}, mode={args.kahip_mode}")
    blocks, edgecut = partition_graph(
        vwgt, xadj, adjncy, adjcwgt,
        nblocks=args.m,
        imbalance=args.imbalance,
        seed=args.seed,
        mode=args.kahip_mode
    )

    # Train MLP classifier
    print("Training PyTorch classifier...")
    model = train_classifier(
        X_np,
        blocks,
        m=args.m,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.nodes,     
        layers=args.layers
    )

    # Save model and inverted file
    os.makedirs(args.index_path, exist_ok=True)
    model_path = os.path.join(args.index_path, "model.pth")
    invfile_path = os.path.join(args.index_path, "inverted_file.npz")

    torch.save({
        "model_state": model.state_dict(),
        "d": d,
        "m": args.m,
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
