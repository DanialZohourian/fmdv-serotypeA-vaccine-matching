# -*- coding: utf-8 -*-
"""Phase 4.6: per-feature QC report for the ML matrix."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ML_PATH = ROOT / "step4_features/feature_matrix_ml.csv"
OUT = ROOT / "step4_features/feature_engineering_report.csv"
META = {
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
}


def feature_type(name):
    if name == "pairwise_aa_distance":
        return "distance"
    if "nglyc" in name:
        return "glycan"
    return "site"


def main():
    ml = pd.read_csv(ML_PATH, encoding="utf-8-sig")
    variable = [col for col in ml.columns if col not in META]
    x = ml[variable]
    y = ml["r1_binary"]
    rows = []
    for col in variable:
        values = x[col].dropna()
        match_mean = x.loc[y == 1, col].mean()
        no_match_mean = x.loc[y == 0, col].mean()
        rows.append({
            "feature": col,
            "type": feature_type(col),
            "n_valid": int(values.count()),
            "n_nan": int(x[col].isna().sum()),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "pct_zero": round(float((values == 0).mean()) * 100, 1),
            "pct_one": round(float((values == 1).mean()) * 100, 1),
            "mean_in_match": round(float(match_mean), 4),
            "mean_in_nomatch": round(float(no_match_mean), 4),
            "diff_by_class": round(abs(float(match_mean) - float(no_match_mean)), 4),
        })
    report = pd.DataFrame(rows)
    report.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("=== feature report ===")
    print(f"features: {len(report)}")
    print("\nby type:")
    print(report["type"].value_counts().to_string())
    print("\ntop 10 by diff_by_class:")
    print(report.nlargest(10, "diff_by_class")[
        ["feature", "type", "mean_in_match", "mean_in_nomatch", "diff_by_class"]
    ].to_string(index=False))
    print(f"\nsaved {OUT}")
    print(f"rows with any NaN: {int((report['n_nan'] > 0).sum())}")


if __name__ == "__main__":
    main()
