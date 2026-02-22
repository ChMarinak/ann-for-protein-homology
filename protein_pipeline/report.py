# report.py

from lib.bio_comment import has_comments, bio_comment, load_pfam_map, load_go_map, load_notes

def recall_at_n(ann_ids, blast_ids, N):
    """Compute Recall@N vs BLAST Top-N."""
    if not blast_ids:
        return 0.0
    return len(set(ann_ids[:N]) & set(blast_ids[:N])) / N


def generate_report(methods, ann_outputs, blast_hits,
                    db_ids, query_ids,
                    N_eval=50, N_print=10):

    method_names = {
        "lsh": "Euclidean LSH",
        "hypercube": "Hypercube",
        "ivfflat": "IVF-Flat",
        "ivfpq": "IVF-PQ",
        "kahip": "NLSH with Kahip",
        "louvain": "NLSH Louvain",
        "blast": "BLAST (Ref)"
    }

    # Load comments data once (if they exist)
    if has_comments:
        query_pfam = load_pfam_map("data/comments/targets.pfam_map.tsv")
        db_pfam = load_pfam_map("data/comments/swissprot_50k.pfam_map.tsv")

        query_go = load_go_map("data/comments/targets.interpro_go.tsv")
        db_go = load_go_map("data/comments/swissprot_50k.interpro_go.tsv")

        query_notes = load_notes("data/comments/targets.notes.tsv")
        db_notes = load_notes("data/comments/swissprot_50k.notes.tsv")

    # column widths (terminal + file)
    W_METHOD = 20
    W_RECALL = 14
    W_TIME = 16
    W_QPS = 8

    def summary_header():
        header = (
            f"{'Method':<{W_METHOD}} | "
            f"{'Avg Recall@N':<{W_RECALL}} | "
            f"{'Time/query (s)':<{W_TIME}} | "
            f"{'QPS':<{W_QPS}}"
        )
        return header, "-" * len(header)

    def summary_row(method, recall, t, qps):
        return (
            f"{method:<{W_METHOD}} | "
            f"{recall:<{W_RECALL}.2f} | "
            f"{t:<{W_TIME}.3f} | "
            f"{qps:<{W_QPS}.1f}"
        )

    out = []

    # Per-query report (file)
    for qi, qid in enumerate(query_ids):
        out.append(f"Query Protein: {qid}")

        out.append(f"N = {N_eval}")
        out.append("[1] Method comparison summary")
        out.append("-" * 70)
        out.append(
            f"{'Method':<{W_METHOD}} | "
            f"{'Time/query (s)':<{W_TIME}} | "
            f"{'QPS':<{W_QPS}} | "
            f"{'Recall@N':<{W_RECALL}}"
        )
        out.append("-" * 70)

        blast_list = blast_hits.get(qid, [])
        blast_ids = [x[0] for x in blast_list]

        for m in methods:
            ann_data = ann_outputs[m]
            ann = ann_data["results"][qi]

            ann_ids = [db_ids[i] for i, _ in ann]
            recall = recall_at_n(ann_ids, blast_ids, N_eval)

            t = ann_data["time_per_query"]
            qps = ann_data["qps"]

            out.append(
                f"{method_names.get(m,m):<{W_METHOD}} | "
                f"{t:<{W_TIME}.3f} | "
                f"{qps:<{W_QPS}.1f} | "
                f"{recall:<{W_RECALL}.2f}"
            )

        out.append(
            f"{method_names['blast']:<{W_METHOD}} | "
            f"{0.0:<{W_TIME}.3f} | "
            f"{0.0:<{W_QPS}.1f} | "
            f"{1.00:<{W_RECALL}.2f}"
        )
        out.append("-" * 70)

        # Top-N neighbors
        out.append(f"[2] Top-N neighbors per method (N = {N_print})")

        # BLAST Top-N only (for Yes / No)
        blast_topn_set = {sid for sid, *_ in blast_list[:N_eval]}

        # ALL BLAST results (for identity percentage)
        blast_identity_all = {sid: identity for sid, identity, *_ in blast_list}

        # Define column widths
        W_RANK = 5
        W_ID = 25
        W_DIST = 8
        W_BLAST = 12
        W_INBLAST = 15
        W_COMMENT = 35

        for m in methods:
            ann = ann_outputs[m]["results"][qi]

            out.append(f"\nMethod: {method_names.get(m,m)}")

            # Header
            out.append(
                f"{'Rank':<{W_RANK}} | "
                f"{'Neighbor ID':<{W_ID}} | "
                f"{'L2 Dist':<{W_DIST}} | "
                f"{'BLAST Identity':<{W_BLAST}} | "
                f"{'In BLAST Top-N?':<{W_INBLAST}} | "
                f"{'Bio comment':<{W_COMMENT}}"
            )
            out.append("-" * (W_RANK + W_ID + W_DIST + W_BLAST + W_INBLAST + W_COMMENT + 10))

            for r, (idx, dist) in enumerate(ann[:N_print], 1):
                nid = db_ids[idx]
                blast_identity = blast_identity_all.get(nid, 0.0)
                in_blast = "Yes" if nid in blast_topn_set else "No"

                if has_comments:
                    comment = bio_comment(
                        qid=qid,
                        nid=nid,
                        dist=dist,
                        blast_identity=blast_identity,
                        in_blast=in_blast,
                        query_pfam=query_pfam,
                        db_pfam=db_pfam,
                        query_go=query_go,
                        db_go=db_go,
                        query_notes=query_notes,
                        db_notes=db_notes
                    )
                else:
                  if in_blast == "Yes":
                      if blast_identity >= 30:
                          comment = "Clear indication of homology"
                      elif 20 <= blast_identity < 30:
                          comment = "Twilight Zone"
                      else:
                          comment = "Unable to distinguish from random similarity"
                  else:
                      if in_blast == "No" and blast_identity > 0:
                          comment = "Possible remote homolog"
                      else:
                          comment = "--"

                out.append(
                    f"{r:<{W_RANK}} | "
                    f"{nid:<{W_ID}} | "
                    f"{dist:<{W_DIST}.3f} | "
                    f"{blast_identity:<{W_BLAST}.1f}% | "
                    f"{in_blast:<{W_INBLAST}} | "
                    f"{comment:<{W_COMMENT}}"
                )

            out.append(f"\n")

    # Terminal output (averages)
    print("\n" + "=" * 70)
    print("AVERAGES PER METHOD (ACROSS ALL QUERIES)")
    print("=" * 70)

    header, sep = summary_header()
    print(header)
    print(sep)

    for m in methods:
        recalls = []

        for qi, qid in enumerate(query_ids):
            blast_list = blast_hits.get(qid, [])
            blast_ids = [x[0] for x in blast_list]

            ann = ann_outputs[m]["results"][qi]
            ann_ids = [db_ids[i] for i, _ in ann]

            recalls.append(recall_at_n(ann_ids, blast_ids, N_eval))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        t = ann_outputs[m]["time_per_query"]
        qps = ann_outputs[m]["qps"]

        print(summary_row(
            method_names.get(m,m),
            avg_recall,
            t,
            qps
        ))

    print("=" * len(header) + "\n")

    return "\n".join(out)
