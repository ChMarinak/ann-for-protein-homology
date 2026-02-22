import argparse
from protein_pipeline.ann import embed_queries, run_ann, parse_ann
from protein_pipeline.blast import run_blast, parse_blast
from protein_pipeline.report import generate_report

ALL_METHODS = ["lsh", "hypercube", "ivfflat", "ivfpq", "kahip", "louvain"]

def expand_methods(method_arg):
    if method_arg == "all":
        return ALL_METHODS
    if method_arg == "neural":
        return "kahip"
    return [method_arg]


def load_ids(path):
    with open(path) as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db_vecs", required=True)
    parser.add_argument("--db_fasta", required=True)
    parser.add_argument("--query_fasta", required=True)
    parser.add_argument("--db_ids", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-N", type=int, default=50)
    parser.add_argument(
        "-method",
        choices=["all", "neural", "lsh", "hypercube", "ivfflat", "ivfpq", "kahip", "louvain"],
        default="all",
        help="ANN method to run"
    )
    args = parser.parse_args()

    # Embed queries
    embed_queries(args.query_fasta, "data/queries.dat")

    query_vecs = "data/queries.dat"
    query_ids = load_ids("data/queries_ids.txt")
    db_ids = load_ids(args.db_ids)

    # BLAST
    run_blast(args.db_fasta, args.query_fasta, "results/blast_results.tsv")
    blast_hits = parse_blast("results/blast_results.tsv")

    # ANN
    methods = expand_methods(args.method)
    ann_outputs = {}

    for m in methods:
        out_file = f"results/ann_{m}.txt"
        run_ann(m, args.db_vecs, query_vecs, out_file, args.N)

        parsed = parse_ann(out_file)

        ann_outputs[m] = parsed

    # Report
    report = generate_report(
        methods,
        ann_outputs,
        blast_hits,
        db_ids,
        query_ids,
        N_eval=args.N,
        N_print=20,
    )

    with open(args.out, "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()