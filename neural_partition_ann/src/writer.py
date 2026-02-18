# writer.py

import os
import numpy as np


def ensure_output_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)


def write_bio_output(
    output_path: str,
    results,
    N: int,
    avg_time: float,
    qps: float
):
    ensure_output_dir(output_path)

    with open(output_path, "w") as f:
        f.write("TYPE bio\n")
        f.write("METHOD Neural_LSH\n")
        f.write(f"N_eval {N}\n")
        f.write(f"Time_per_query {avg_time:.6f}\n")
        f.write(f"QPS {qps:.2f}\n")

        for res in results:
            f.write(f"\nQUERY {res['query_idx']}\n")
            for rank, (idx, dist) in enumerate(
                zip(res["nn_indices"], res["nn_dists"]), 1
            ):
                f.write(f"{rank} {idx} {dist:.6f}\n")


def write_standard_output(
    output_path: str,
    results,
    ground_truths,
    afs,
    recalls,
    avg_af,
    avg_recall,
    avg_approx_time,
    avg_true_time,
    qps,
    N
):
    ensure_output_dir(output_path)

    with open(output_path, "w") as f:
        f.write("Neural LSH\n")

        for result, gt in zip(results, ground_truths):
            query_idx = result["query_idx"]

            f.write(f"Query: {query_idx}\n")

            for rank, (idx, dist) in enumerate(
                zip(result["nn_indices"], result["nn_dists"]), 1
            ):
                f.write(f"Nearest neighbor-{rank}: {idx}\n")
                f.write(f"distanceApproximate: {dist:.6f}\n")

                if rank <= len(gt["dists"]):
                    f.write(f"distanceTrue: {gt['dists'][rank-1]:.6f}\n")

            if result["range_indices"]:
                f.write("R-near neighbors:\n")
                for r in result["range_indices"]:
                    f.write(f"{r}\n")

            af = afs[query_idx] if query_idx < len(afs) else 0.0
            recall = recalls[query_idx] if query_idx < len(recalls) else 0.0

            f.write(f"Average AF: {af:.6f}\n")
            f.write(f"Recall@{N}: {recall:.6f}\n")
            f.write(f"QPS: {qps:.2f}\n")
            f.write(f"tApproximateAverage: {avg_approx_time:.6f}\n")
            f.write(f"tTrueAverage: {avg_true_time:.6f}\n\n")

        f.write(f"Overall Average AF: {avg_af:.6f}\n")
        f.write(f"Overall Recall@{N}: {avg_recall:.6f}\n")
        f.write(f"Overall QPS: {qps:.2f}\n")
        f.write(f"Overall tApproximateAverage: {avg_approx_time:.6f}\n")
        f.write(f"Overall tTrueAverage: {avg_true_time:.6f}\n")
