import argparse
import torch
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm


MAX_LEN = 1022  # for <cls> and <eos>


def parse_fasta(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        sequences.append((record.id, seq))
    return sequences


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    # Load sequences
    sequences = parse_fasta(args.input)

    embeddings = []
    ids = []

    batch_size = args.batch_size

    # Batch Processing
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i + batch_size]
        batch_ids, batch_seqs = zip(*batch)

        data = list(zip(batch_ids, batch_seqs))
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        # Inference
        with torch.no_grad():
            results = model(tokens, repr_layers=[6])

        token_embeddings = results["representations"][6]

        # Mean Pooling (ignoring padding)
        for j, seq in enumerate(batch_seqs):
            length = len(seq) + 2  # <cls> + <eos>
            emb = token_embeddings[j, :length].mean(dim=0)
            embeddings.append(emb.cpu().numpy())
            ids.append(batch_ids[j])

    # Save
    embeddings = np.vstack(embeddings).astype(np.float32)
    embeddings.tofile(args.output)

    with open(args.output.replace(".dat", "_ids.txt"), "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Embedding with ESM-2")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output .dat file")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
