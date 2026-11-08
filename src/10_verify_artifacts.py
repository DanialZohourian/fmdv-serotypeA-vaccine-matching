# -*- coding: utf-8 -*-
"""Stage 9 checks: artifacts, reproduced AUC, SHAP, validation, and the two examples."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from utils.sequence_processing import (
    alignment_cache,
    extract_features,
    features_to_array,
    match_shap,
    place_sequences,
    read_fasta,
    validate_sequence,
)

ART = ROOT / "app" / "model_artifacts"
REQUIRED = [
    "app/model_artifacts/rf_model.pkl",
    "app/model_artifacts/shap_explainer.pkl",
    "app/model_artifacts/model_meta.json",
    "app/model_artifacts/selected_features.txt",
    "app/model_artifacts/reference_alignment.fasta",
    "app/model_artifacts/alignment_to_pdb_mapping.csv",
    "app/model_artifacts/shap_global_importance.csv",
    "app/app.py",
    "app/utils/sequence_processing.py",
    "app/requirements.txt",
    "app/.streamlit/config.toml",
]


def main():
    print("=== stage 9 checks ===\n")
    for relative in REQUIRED:
        path = ROOT / relative
        ok = path.exists()
        size = path.stat().st_size / 1024 if ok else 0
        print(f"  {'OK' if ok else 'FAIL'} {relative} ({size:.1f} KB)")

    model = joblib.load(ART / "rf_model.pkl")
    meta = json.loads((ART / "model_meta.json").read_text(encoding="utf-8"))
    params = model.get_params()
    checks = {
        "n_estimators=300": model.n_estimators == 300,
        "max_depth=10": model.max_depth == 10,
        "max_features=sqrt": params["max_features"] == "sqrt",
        "min_samples_leaf=2": model.min_samples_leaf == 2,
        "threshold=0.23": meta["decision_threshold"] == 0.23,
    }
    print("\n--- model ---")
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}")

    selected = [line.strip() for line in (ART / "selected_features.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = pd.read_csv(ROOT / "step5_boruta" / "feature_matrix_selected.csv", encoding="utf-8-sig")
    split = pd.read_csv(ROOT / "step5_boruta" / "train_test_split.csv", encoding="utf-8-sig")
    test_ids = set(split.loc[split["split"] == "test", "pair_id"])
    test_df = matrix[matrix["pair_id"].isin(test_ids)]
    x_test = test_df[selected].to_numpy(dtype=np.float64)
    y_test = test_df["r1_binary"].to_numpy(dtype=int)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    saved = pd.read_csv(ROOT / "step6_model" / "rf_test_predictions.csv", encoding="utf-8-sig")
    merged = test_df[["pair_id"]].merge(saved[["pair_id", "prob_match"]], on="pair_id")
    max_gap = float(np.max(np.abs(probabilities - merged["prob_match"].to_numpy())))
    print(f"\nrebuilt AUC {auc:.4f}  max |probability gap vs stage 6| {max_gap:.6f}")
    print(f"  {'OK' if abs(auc - 0.8429) < 0.001 else 'FAIL'} AUC within 0.001 of 0.8429")
    print(f"  {'OK' if max_gap < 1e-6 else 'FAIL'} probabilities match the saved test file")

    explainer = joblib.load(ART / "shap_explainer.pkl")
    row, base_value = match_shap(explainer, x_test[:1])
    rebuilt = base_value + row.sum()
    print(f"\nSHAP base {base_value:.4f}  row shape {row.shape}  reconstruction gap {abs(rebuilt - probabilities[0]):.6f}")
    print(f"  {'OK' if abs(base_value - 0.4995) < 0.01 else 'FAIL'} base value near 0.4995")
    print(f"  {'OK' if abs(rebuilt - probabilities[0]) < 1e-4 else 'FAIL'} SHAP adds up to the match probability")

    ok_long, _msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY" * 11)
    ok_short, _msg = validate_sequence("ACDEFG" * 5)
    ok_empty, _msg = validate_sequence("  \n")
    ok_bad, bad_msg = validate_sequence(("ACDEFGHIKLMNPQRSTVWY" * 11) + "1")
    print("\n--- validation ---")
    print(f"  {'OK' if ok_long else 'FAIL'} 220 aa accepted")
    print(f"  {'OK' if not ok_short else 'FAIL'} 30 aa rejected")
    print(f"  {'OK' if not ok_empty else 'FAIL'} empty rejected")
    print(f"  {'OK' if (not ok_bad and '1' in bad_msg) else 'FAIL'} illegal character rejected ({bad_msg})")

    records = read_fasta(ART / "reference_alignment.fasta")
    by_name = {header.split("|")[0]: seq for header, seq in records}
    reference = by_name["A22/IRQ/24/64"]
    cache = alignment_cache(records)
    profile = [seq for _header, seq in records]
    examples = [
        ("A/IRN/10/2018", "A22/IRQ/24/64", "Match", 0.8226428217998326),
        ("A/IRN/5/2006", "A/IRN/1996", "No-match", 0.001107123870281765),
        ("A/IRN/23/2018", "A/Iran-05", "Match", 0.5849470896281243),
    ]
    print("\n--- examples through the app feature path ---")
    example_ok = True
    for virus_name, serum_name, expected, saved_prob in examples:
        virus_ok, virus_seq = validate_sequence(by_name[virus_name].replace("-", ""))
        serum_ok, serum_seq = validate_sequence(by_name[serum_name].replace("-", ""))
        aligned, note = place_sequences([virus_seq, serum_seq], reference, cache, profile)
        features = extract_features(aligned[0], aligned[1], selected)
        matrix_row = features_to_array(features, selected)
        stored = matrix.loc[(matrix["virus_name"] == virus_name) & (matrix["serum_name"] == serum_name), selected]
        feature_gap = float(np.max(np.abs(matrix_row - stored.to_numpy(dtype=float))))
        probability = float(model.predict_proba(matrix_row)[0, 1])
        label = "Match" if probability >= 0.23 else "No-match"
        ok = virus_ok and serum_ok and label == expected and feature_gap < 1e-6 and abs(probability - saved_prob) < 1e-6 and note == ""
        example_ok = example_ok and ok
        print(
            f"  {'OK' if ok else 'FAIL'} {virus_name} vs {serum_name}: "
            f"{label} prob={probability:.4f} feature_gap={feature_gap:.3g} note={note!r}"
        )

    requirements = (ROOT / "app" / "requirements.txt").read_text(encoding="utf-8")
    print("\n--- requirements ---")
    for package in ["streamlit", "scikit-learn", "shap", "biopython", "joblib"]:
        print(f"  {'OK' if package in requirements else 'FAIL'} {package}")
    ready = all(checks.values()) and abs(auc - 0.8429) < 0.001 and abs(base_value - 0.4995) < 0.01 and example_ok
    print(f"\nREADY {ready}")


if __name__ == "__main__":
    main()
