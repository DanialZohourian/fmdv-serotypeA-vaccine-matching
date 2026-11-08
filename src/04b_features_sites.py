# -*- coding: utf-8 -*-
"""Phase 4.2: Poisson-corrected pairwise VP1 amino-acid distance."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FASTA = ROOT / "step3_alignment/aligned_vp1_protein_final.fasta"
PAIRS = ROOT / "dataset/master_pairs.csv"
OUT = ROOT / "step4_features/pair_features.csv"


def load_fasta_full_header(fasta_path):
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


def poisson_corrected_distance(seq1, seq2):
    """Poisson-corrected p-distance. Gaps and X are ignored."""
    skip = {"-", "X"}
    comparable = [(a, b) for a, b in zip(seq1, seq2) if a not in skip and b not in skip]
    if len(comparable) == 0:
        return np.nan
    n_diff = sum(1 for a, b in comparable if a != b)
    p = n_diff / len(comparable)
    if p == 0:
        return 0.0
    if p >= 1.0:
        return np.nan
    return float(-np.log(1.0 - p))


def main():
    seq_dict = load_fasta_full_header(FASTA)
    mp = pd.read_csv(PAIRS, encoding="utf-8-sig")
    mp["serum_in_fasta"] = mp["serum_name"].isin(seq_dict)
    mp["virus_in_fasta"] = mp["virus_name"].isin(seq_dict)
    mp["can_compute"] = mp["serum_in_fasta"] & mp["virus_in_fasta"]

    distances = []
    compared = []
    for row in mp.itertuples(index=False):
        if not row.can_compute:
            distances.append(np.nan)
            compared.append(0)
            continue
        seq_v = seq_dict[row.virus_name]
        seq_s = seq_dict[row.serum_name]
        skip = {"-", "X"}
        n_comp = sum(1 for a, b in zip(seq_v, seq_s) if a not in skip and b not in skip)
        distances.append(poisson_corrected_distance(seq_v, seq_s))
        compared.append(n_comp)

    mp["pairwise_aa_distance"] = distances
    mp["comparable_sites"] = compared

    valid = mp.loc[mp["can_compute"], "pairwise_aa_distance"]
    print("=== Poisson-corrected distance ===")
    print(f"mean:  {valid.mean():.4f}")
    print(f"std:   {valid.std():.4f}")
    print(f"min:   {valid.min():.4f}")
    print(f"max:   {valid.max():.4f}")
    print(f"NaN:   {int(valid.isna().sum())}")
    print(f"zero:  {int((valid == 0).sum())}")
    print(f"comparable sites min/max: {mp.loc[mp['can_compute'], 'comparable_sites'].min()} / {mp.loc[mp['can_compute'], 'comparable_sites'].max()}")

    computable = mp.loc[mp["can_compute"]].copy()
    print("\nmean distance by r1_binary:")
    print(computable.groupby("r1_binary")["pairwise_aa_distance"].mean().round(4))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
        "virus_in_fasta", "serum_in_fasta", "can_compute",
        "pairwise_aa_distance", "comparable_sites",
    ]
    mp[keep].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved {OUT} rows={len(mp)}")


if __name__ == "__main__":
    main()
