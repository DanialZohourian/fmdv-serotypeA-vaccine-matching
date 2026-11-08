# -*- coding: utf-8 -*-
"""Phase 4.5: archive feature matrix and the ML-ready matrix."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEAT = ROOT / "step4_features"
PAIRS = ROOT / "dataset/master_pairs.csv"
META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated", "can_compute",
]
ML_META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
]
NGLYC = ["nglyc_count_diff", "nglyc_shared", "nglyc_virus_only", "nglyc_serum_only"]


def main():
    master = pd.read_csv(PAIRS, encoding="utf-8-sig")
    features = pd.read_csv(FEAT / "pair_features.csv", encoding="utf-8-sig")
    sites = pd.read_csv(FEAT / "site_differences.csv", encoding="utf-8-sig")
    if not master["pair_id"].equals(features["pair_id"]) or not master["pair_id"].equals(sites["pair_id"]):
        raise SystemExit("pair_id order differs between master, distances, and site matrix")

    site_cols = [col for col in sites.columns if col.startswith("site_")]
    if len(site_cols) != 213:
        raise SystemExit(f"expected 213 site columns, found {len(site_cols)}")

    meta = master[META[:-1]].copy()
    meta["can_compute"] = features["can_compute"].astype(bool)
    numeric = pd.concat(
        [features[["pairwise_aa_distance", *NGLYC]], sites[site_cols]],
        axis=1,
    )
    feature_cols = ["pairwise_aa_distance", *site_cols, *NGLYC]
    full = pd.concat([meta.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
    full_path = FEAT / "feature_matrix_full.csv"
    full.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"full matrix: {full.shape}")
    print(f"saved {full_path}")

    ml_ready = full.loc[full["can_compute"]].copy()
    ml_values = ml_ready[feature_cols]
    constant = [col for col in feature_cols if ml_values[col].nunique(dropna=True) <= 1]
    variable = [col for col in feature_cols if col not in constant]
    print(f"\nconstant features removed: {len(constant)}")
    print(f"variable features kept: {len(variable)}")

    ml = ml_ready[ML_META + variable].reset_index(drop=True)
    ml_path = FEAT / "feature_matrix_ml.csv"
    ml.to_csv(ml_path, index=False, encoding="utf-8-sig")
    print(f"\nML-ready matrix: {ml.shape}")
    print(f"saved {ml_path}")
    print("\nr1_binary:")
    print(ml["r1_binary"].value_counts().sort_index().to_string())
    print(f"missing feature values: {int(ml[variable].isna().sum().sum())}")


if __name__ == "__main__":
    main()
