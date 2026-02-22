# blast.py

import subprocess
from pathlib import Path
from collections import defaultdict

def run_blast(db_fasta, query_fasta, out_tsv):
    # Where the swissprot_db should be
    db_name = "data/db/swissprot_db"

    if not Path(db_name + ".pin").exists():
        subprocess.run([
            "makeblastdb",
            "-in", db_fasta,
            "-dbtype", "prot",
            "-out", db_name
        ], check=True)

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "blastp",
        "-db", db_name,
        "-query", query_fasta,
        "-outfmt", "6",
        "-out", out_tsv
    ], check=True)

def parse_blast(path, evalue_thr=1):
    hits = defaultdict(list)

    with open(path) as f:
        for line in f:
            c = line.rstrip().split("\t")
            qid, sid = c[0], c[1]
            pident = float(c[2])
            evalue = float(c[10])
            bitscore = float(c[11])

            if evalue <= evalue_thr:
                hits[qid].append((sid, pident, bitscore))

    for qid in hits:
        hits[qid].sort(key=lambda x: -x[2])

    return hits