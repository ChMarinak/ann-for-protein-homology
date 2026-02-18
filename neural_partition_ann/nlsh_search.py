"""
Neural LSH Search Implementation
Performs approximate nearest neighbor search using a trained Neural LSH model.
"""

import argparse
import os
import time
import numpy as np
import torch

from src.file_parser import read_mnist, read_sift, read_bio, load_ground_truth, load_model, load_inverted_file
from src.search import search_neural_lsh
from src.evaluator import compute_ground_truth, compute_af, compute_recall
from src.writer import write_bio_output, write_standard_output


def parse_args():
    parser = argparse.ArgumentParser(description="Neural LSH Search")

    parser.add_argument("-d", required=True, help="Dataset file")
    parser.add_argument("-q", required=True, help="Query file")
    parser.add_argument("-i", "--index_path", required=True, help="Index directory")
    parser.add_argument("-o", required=True, help="Output file")
    parser.add_argument("-type", required=True, choices=["sift", "mnist", "bio"])

    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-R", type=float, default=None)
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-range", type=str, default="false")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--num_queries", type=int, default=None)

    return parser.parse_args()


def load_dataset(path, dtype):
    if dtype == "sift":
        obj = read_sift(path)
    elif dtype == "mnist":
        obj = read_mnist(path)
    else:
        obj = read_bio(path)

    return np.vstack(obj.data).astype(np.float32)


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default radius
    if args.R is None:
        args.R = 2000.0 if args.type == "mnist" else 0.5

    is_bio = args.type == "bio"

    # Load dataset and queries
    X = load_dataset(args.d, args.type)
    queries = load_dataset(args.q, args.type)

    if args.num_queries:
        queries = queries[:args.num_queries]

    X_gpu = torch.from_numpy(X).to(device)

    # Load model and inverted file
    model, _, _ = load_model(
        os.path.join(args.index_path, "model.pth"),
        device
    )

    inverted = load_inverted_file(
        os.path.join(args.index_path, "inverted_file.npz")
    )

    # Ground truth handling
    ground_truths = []
    true_times = []
    use_precomputed_gt = False
    overall_ttrue_from_file = 0.0

    if (
        not is_bio
        and args.ground_truth
        and os.path.exists(args.ground_truth)
    ):
        gt_results, overall_ttrue_from_file = load_ground_truth(
            args.ground_truth,
            max_queries=len(queries),
        )

        for gt in gt_results:
            ground_truths.append({
                "indices": gt.indices,
                "dists": gt.dists,
                "time": gt.time,
            })

        use_precomputed_gt = True

    results = []
    approx_times = []

    total_start = time.time()

    for q_idx, query in enumerate(queries):

        start = time.time()

        nn_idx, nn_dist, range_idx = search_neural_lsh(
            query=query,
            model=model,
            dataset_gpu=X_gpu,
            inverted=inverted,
            T=args.T,
            N=args.N,
            R=args.R,
            do_range=(args.range.lower() == "true"),
            device=device,
        )

        approx_time = time.time() - start
        approx_times.append(approx_time)

        results.append({
            "query_idx": q_idx,
            "nn_indices": nn_idx,
            "nn_dists": nn_dist,
            "range_indices": range_idx,
        })

        # Only compute ground truth if not using precomputed
        if not is_bio and not use_precomputed_gt:
            t0 = time.time()
            gt_idx, gt_dist = compute_ground_truth(query, X, args.N)
            ttrue = time.time() - t0

            true_times.append(ttrue)

            ground_truths.append({
                "indices": gt_idx,
                "dists": gt_dist,
                "time": ttrue,
            })

    total_time = time.time() - total_start
    avg_approx_time = np.mean(approx_times) if approx_times else 0.0
    qps = len(queries) / total_time if total_time > 0 else 0.0

    # BIO mode: ANN only
    if is_bio:
        write_bio_output(
            output_path=args.o,
            results=results,
            N=args.N,
            avg_time=avg_approx_time,
            qps=qps,
        )
        return

    # Compute metrics
    afs = []
    recalls = []

    for result, gt in zip(results, ground_truths):
        af = compute_af(result["nn_dists"], gt["dists"])
        recall = compute_recall(result["nn_indices"], gt["indices"])
        afs.append(af)
        recalls.append(recall)

    avg_af = np.mean(afs) if afs else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0

    if use_precomputed_gt:
        avg_true_time = overall_ttrue_from_file
    else:
        avg_true_time = np.mean(true_times) if true_times else 0.0

    # Write output
    write_standard_output(
        output_path=args.o,
        results=results,
        ground_truths=ground_truths,
        afs=afs,
        recalls=recalls,
        avg_af=avg_af,
        avg_recall=avg_recall,
        avg_approx_time=avg_approx_time,
        avg_true_time=avg_true_time,
        qps=qps,
        N=args.N,
    )


if __name__ == "__main__":
    main()
