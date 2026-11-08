# -*- coding: utf-8 -*-
"""Save the final random forest and the files the Streamlit app loads."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "app" / "model_artifacts"
MATRIX = ROOT / "step5_boruta" / "feature_matrix_selected.csv"
SPLIT = ROOT / "step5_boruta" / "train_test_split.csv"
SELECTED = ROOT / "step5_boruta" / "selected_features.txt"
FASTA = ROOT / "step3_alignment" / "aligned_vp1_protein_final.fasta"
MAPPING = ROOT / "step8_shap" / "alignment_to_pdb_mapping.csv"
IMPORTANCE = ROOT / "step8_shap" / "shap_global_importance.csv"
CV = ROOT / "step6_model" / "cv_best_summary.csv"
METRICS = ROOT / "step6_model" / "rf_test_metrics.csv"


def main():
    ART.mkdir(parents=True, exist_ok=True)
    selected = [line.strip() for line in SELECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = pd.read_csv(MATRIX, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT, encoding="utf-8-sig")
    train_ids = set(split.loc[split["split"] == "train", "pair_id"])
    test_ids = set(split.loc[split["split"] == "test", "pair_id"])
    train_df = matrix[matrix["pair_id"].isin(train_ids)].reset_index(drop=True)
    test_df = matrix[matrix["pair_id"].isin(test_ids)].reset_index(drop=True)
    x_train_raw = train_df[selected].to_numpy(dtype=np.float64)
    y_train_raw = train_df["r1_binary"].to_numpy(dtype=int)
    x_test = test_df[selected].to_numpy(dtype=np.float64)
    y_test = test_df["r1_binary"].to_numpy(dtype=int)
    x_train, y_train = SMOTE(k_neighbors=5, random_state=42).fit_resample(x_train_raw, y_train_raw)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    print(f"rebuilt test AUC {auc:.4f}  train {x_train.shape}", flush=True)
    joblib.dump(model, ART / "rf_model.pkl")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    joblib.dump(explainer, ART / "shap_explainer.pkl")
    cv = pd.read_csv(CV, encoding="utf-8-sig")
    rf_row = cv.loc[cv["model"] == "rf"].iloc[0]
    metrics = pd.read_csv(METRICS, encoding="utf-8-sig")
    official = metrics.loc[np.isclose(metrics["threshold"], 0.23)].iloc[0]
    meta = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "decision_threshold": 0.23,
        "cv_auc": float(rf_row["cv_auc_mean"]),
        "cv_auc_std": float(rf_row["cv_auc_std"]),
        "test_auc": float(official["auc_roc"]),
        "test_pr_auc": float(official["auc_pr"]),
        "test_f1_opt": float(official["f1"]),
        "test_sensitivity": float(official["sensitivity"]),
        "test_specificity": float(official["specificity"]),
        "n_features": len(selected),
        "n_train_pairs": int(len(train_df)),
        "train_match": int((y_train_raw == 1).sum()),
        "train_nomatch": int((y_train_raw == 0).sum()),
        "n_test_pairs": int(len(test_df)),
        "n_dataset_pairs": int(len(matrix)),
        "dataset_match": int((matrix["r1_binary"] == 1).sum()),
        "dataset_nomatch": int((matrix["r1_binary"] == 0).sum()),
        "smote_k": 5,
        "random_state": 42,
        "alignment_length": 213,
        "serotype": "A",
        "region": "Pool 3 (Middle East/Asia)",
        "reference_strain": "A22/IRQ/24/64",
        "pdb_structure": "4GH4 chain A",
        "paper_reference": "Azimi et al. (in preparation)",
        "date_trained": "2026-10-05",
        "rebuilt_test_auc": auc,
    }
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.copy(SELECTED, ART / "selected_features.txt")
    shutil.copy(FASTA, ART / "reference_alignment.fasta")
    shutil.copy(MAPPING, ART / "alignment_to_pdb_mapping.csv")
    shutil.copy(IMPORTANCE, ART / "shap_global_importance.csv")
    print("artifacts:", flush=True)
    for path in sorted(ART.iterdir()):
        print(f"  {path.name}: {path.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
