# bio_comment

import os

COMMENTS_FILES = [
    "data/comments/targets.pfam_map.tsv",
    "data/comments/swissprot_50k.pfam_map.tsv",
    "data/comments/targets.interpro_go.tsv",
    "data/comments/swissprot_50k.interpro_go.tsv",
    "data/comments/targets.notes.tsv",
    "data/comments/swissprot_50k.notes.tsv"
]

has_comments = all(os.path.exists(f) for f in COMMENTS_FILES)

def load_pfam_map(path):
    pfam_map = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            acc, pfam = parts[:2]
            pfam_map.setdefault(acc, set()).add(pfam)
    return pfam_map

def load_go_map(path):
    go_map = {}
    with open(path) as f:
        next(f)  
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:  
                continue
            acc, go = parts[:2]
            go_map.setdefault(acc, set()).add(go)
    return go_map


def load_notes(path):
    notes = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            acc, note = parts
            notes[acc] = note
    return notes


def bio_comment(
    qid,
    nid,
    dist,
    blast_identity,
    in_blast,
    query_pfam,
    db_pfam,
    query_go,
    db_go,
    query_notes,
    db_notes
):
    def normalize_acc(acc):
        if "|" in acc:
            return acc.split("|")[1]
        return acc

    nid_norm = normalize_acc(nid)

    q_pfam = set(query_pfam.get(qid, []))
    n_pfam = set(db_pfam.get(nid_norm, []))

    q_go = set(query_go.get(qid, []))
    n_go = set(db_go.get(nid_norm, []))

    q_note = query_notes.get(qid, "")
    n_note = db_notes.get(nid_norm, "")

    shared_pfam = q_pfam & n_pfam
    shared_go = q_go & n_go

    if in_blast == "Yes":
        if blast_identity >= 30:
            return "Clear homology (BLAST)"
        if 20 <= blast_identity < 30:
            if shared_pfam:
                return f"Remote homolog (shared Pfam: {', '.join(shared_pfam)})"
            return "Twilight zone BLAST hit"
        if shared_pfam or shared_go:
            return "BLAST hit with supporting functional/structural similarity"
        else:
            return "Possible local similarity (BLAST hit)"

    if in_blast == "No":

        # Prints notes for comparison
        # if shared_pfam and shared_go:
        #     return (
        #         "Possible functional homology (shared Pfam: {', '.join(shared_pfam)}) | "
        #         f"\nQuery: {q_note} | \n"
        #         f"\nNeighbor: {n_note}\n"
        #     )

        # if shared_pfam and shared_go:
        #     return (
        #         "Possible functional homology | "
        #         f"\nQuery: {q_note} | \n"
        #         f"\nNeighbor: {n_note}\n"
        #     )

        if shared_pfam and shared_go:
            return f"True positive remote homolog (shared Pfam: {', '.join(shared_pfam)})(shared GO terms)"
        if shared_pfam:
            return f"True positive remote homolog (shared Pfam: {', '.join(shared_pfam)})"
        if shared_go:
            return "Possible functional homology (shared GO terms)"
        if blast_identity > 0:
            return "Embedding-based similarity χωρίς λειτουργική συνάφεια (πιθανό FP)"

    return "--"

