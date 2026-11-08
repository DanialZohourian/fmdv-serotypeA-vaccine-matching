# -*- coding: utf-8 -*-
"""Phase 4.3: one boolean difference column per aligned VP1 site."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FASTA = ROOT / "step3_alignment/aligned_vp1_protein_final.fasta"
PAIRS = ROOT / "dataset/master_pairs.csv"
OUT = ROOT / "step4_features/site_differences.csv"
ALN_LEN = 213
SKIP = {"-", "X"}


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


def main():
    seq_dict = load_fasta_full_header(FASTA)
    bad_len = [name for name, seq in seq_dict.items() if len(seq) != ALN_LEN]
    if bad_len:
        raise SystemExit(f"alignment length is not {ALN_LEN}: {bad_len[:5]}")

    mp = pd.read_csv(PAIRS, encoding="utf-8-sig").reset_index(drop=True)
    mp["can_compute"] = mp["virus_name"].isin(seq_dict) & mp["serum_name"].isin(seq_dict)

    site_cols = [f"site_{i + 1:03d}_diff" for i in range(ALN_LEN)]
    site_matrix = np.full((len(mp), ALN_LEN), np.nan, dtype=np.float32)

    for row_idx, row in mp.iterrows():
        if not row["can_compute"]:
            continue
        seq_v = seq_dict[row["virus_name"]]
        seq_s = seq_dict[row["serum_name"]]
        for pos in range(ALN_LEN):
            aa_v = seq_v[pos]
            aa_s = seq_s[pos]
            if aa_v in SKIP or aa_s in SKIP:
                site_matrix[row_idx, pos] = 0.0
            else:
                site_matrix[row_idx, pos] = 0.0 if aa_v == aa_s else 1.0

    site_df = pd.DataFrame(site_matrix, columns=site_cols)
    computable = mp["can_compute"]
    site_comp = site_df.loc[computable]

    print("=== Site-specific polymorphisms ===")
    print(f"matrix: {site_df.shape}")
    print(f"mean difference per site, across pairs: {site_comp.mean().mean():.3f}")

    top_variable = site_comp.mean().nlargest(10)
    print("\nmost variable sites:")
    for pos, freq in top_variable.items():
        print(f"  {pos}: {freq:.3f} ({freq * 100:.1f}% of pairs differ)")

    constant_cols = [col for col in site_cols if site_comp[col].nunique(dropna=True) <= 1]
    print(f"\nconstant sites: {len(constant_cols)}")
    print(f"variable sites: {ALN_LEN - len(constant_cols)}")
    print(f"computable rows with any NaN: {int(site_comp.isna().any(axis=1).sum())}")
    print(f"non-computable rows: {int((~computable).sum())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    saved = pd.concat([mp[["pair_id", "virus_name", "serum_name", "can_compute"]], site_df], axis=1)
    saved.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
