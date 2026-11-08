# -*- coding: utf-8 -*-
"""Phase 4.4: N-glycosylation site differences between virus and serum."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FASTA = ROOT / "step3_alignment/aligned_vp1_protein_final.fasta"
PAIRS = ROOT / "dataset/master_pairs.csv"
FEATURES = ROOT / "step4_features/pair_features.csv"
MOTIF = re.compile(r"N[^P][ST]")


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


def find_nglyc_sites(seq_aligned):
    """1-based motif positions after gaps and X are removed."""
    seq_nogap = seq_aligned.replace("-", "").replace("X", "")
    return [match.start() + 1 for match in MOTIF.finditer(seq_nogap)]


def main():
    seq_dict = load_fasta_full_header(FASTA)
    mp = pd.read_csv(PAIRS, encoding="utf-8-sig")
    mp["can_compute"] = mp["virus_name"].isin(seq_dict) & mp["serum_name"].isin(seq_dict)

    count_diff = []
    shared = []
    virus_only = []
    serum_only = []
    for row in mp.itertuples(index=False):
        if not row.can_compute:
            count_diff.append(np.nan)
            shared.append(np.nan)
            virus_only.append(np.nan)
            serum_only.append(np.nan)
            continue
        sites_v = set(find_nglyc_sites(seq_dict[row.virus_name]))
        sites_s = set(find_nglyc_sites(seq_dict[row.serum_name]))
        count_diff.append(abs(len(sites_v) - len(sites_s)))
        shared.append(len(sites_v & sites_s))
        virus_only.append(len(sites_v - sites_s))
        serum_only.append(len(sites_s - sites_v))

    mp["nglyc_count_diff"] = count_diff
    mp["nglyc_shared"] = shared
    mp["nglyc_virus_only"] = virus_only
    mp["nglyc_serum_only"] = serum_only

    comp = mp.loc[mp["can_compute"]]
    print("=== N-Glycosylation Features ===")
    for col in ["nglyc_count_diff", "nglyc_shared", "nglyc_virus_only", "nglyc_serum_only"]:
        print(f"{col}: mean={comp[col].mean():.3f}  min={comp[col].min():.0f}  max={comp[col].max():.0f}")
    print("\nby r1_binary:")
    print(comp.groupby("r1_binary")[["nglyc_count_diff", "nglyc_shared"]].mean().round(3))

    site_counts = {name: len(find_nglyc_sites(seq)) for name, seq in seq_dict.items()}
    used = set(mp.loc[mp["can_compute"], "virus_name"]) | set(mp.loc[mp["can_compute"], "serum_name"])
    used_counts = [site_counts[name] for name in used]
    print(f"\nmotif count among used sequences: min={min(used_counts)} max={max(used_counts)}")
    print("sequences with 0 motifs:", [name for name in sorted(used) if site_counts[name] == 0])

    features = pd.read_csv(FEATURES, encoding="utf-8-sig")
    extra = mp[["pair_id", "nglyc_count_diff", "nglyc_shared", "nglyc_virus_only", "nglyc_serum_only"]]
    merged = features.drop(columns=[c for c in extra.columns if c != "pair_id" and c in features.columns])
    merged = merged.merge(extra, on="pair_id", how="left", validate="one_to_one")
    merged.to_csv(FEATURES, index=False, encoding="utf-8-sig")
    print(f"saved columns onto {FEATURES} rows={len(merged)}")


if __name__ == "__main__":
    main()
