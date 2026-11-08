# -*- coding: utf-8 -*-
"""Phase 4.1: index aligned VP1 sequences and tag master pairs."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FASTA = str(ROOT / "step3_alignment/aligned_vp1_protein_final.fasta")
PAIRS = str(ROOT / "dataset/master_pairs.csv")


def load_fasta_full_header(fasta_path):
    """Map the first header field to the aligned sequence."""
    seq_dict = {}
    current_name = None
    current_seq = []
    with open(fasta_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_name:
                    seq_dict[current_name] = "".join(current_seq)
                current_name = line[1:].split("|")[0]
                current_seq = []
            elif line:
                current_seq.append(line.upper())
        if current_name:
            seq_dict[current_name] = "".join(current_seq)
    return seq_dict


def main():
    seq_dict = load_fasta_full_header(FASTA)
    print(f"sequences loaded: {len(seq_dict)}")
    lengths = {len(seq) for seq in seq_dict.values()}
    print(f"aligned lengths: {lengths}")

    mp = pd.read_csv(PAIRS, encoding="utf-8-sig")
    print(f"master pairs: {len(mp)}")
    print(f"source: {PAIRS}")

    missing_virus = [v for v in mp["virus_name"].unique() if v not in seq_dict]
    missing_serum = [s for s in mp["serum_name"].unique() if s not in seq_dict]
    print(f"\nviruses missing from FASTA: {len(missing_virus)}")
    for name in missing_virus:
        print(f"  {name}")
    print(f"\nsera missing from FASTA: {len(missing_serum)}")
    for name in missing_serum:
        n = int((mp["serum_name"] == name).sum())
        print(f"  {name}  pairs={n}")

    mp["serum_in_fasta"] = mp["serum_name"].isin(seq_dict)
    mp["virus_in_fasta"] = mp["virus_name"].isin(seq_dict)
    mp["can_compute"] = mp["serum_in_fasta"] & mp["virus_in_fasta"]
    print(f"\ncomputable pairs: {int(mp['can_compute'].sum())}")
    print(f"not computable: {int((~mp['can_compute']).sum())}")


if __name__ == "__main__":
    main()
