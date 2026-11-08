# -*- coding: utf-8 -*-
"""Recover held-out predictions from the 96-feature random-forest baseline."""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "step4_features" / "feature_matrix_ml.csv"
SPLIT_PATH = ROOT / "step5_boruta" / "train_test_split.csv"
OUT = ROOT / "step6_model" / "rf_all96_test_predictions.csv"
META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
]
THRESHOLD = 0.23


def main():
    full = pd.read_csv(FULL_PATH, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    features = [column for column in full.columns if column not in META]
    if len(features) != 96:
        raise SystemExit(f"expected 96 features, found {len(features)}")
    train_ids = set(split.loc[split["split"] == "train", "pair_id"])
    test_ids = set(split.loc[split["split"] == "test", "pair_id"])
    train = full[full["pair_id"].isin(train_ids)].reset_index(drop=True)
    test = full[full["pair_id"].isin(test_ids)].reset_index(drop=True)
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = train["r1_binary"].to_numpy(dtype=int)
    x_test = test[features].to_numpy(dtype=np.float64)
    y_test = test["r1_binary"].to_numpy(dtype=int)
    x_resampled, y_resampled = SMOTE(k_neighbors=5, random_state=42).fit_resample(x_train, y_train)
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    model.fit(x_resampled, y_resampled)
    probability = model.predict_proba(x_test)[:, 1]
    pred_05 = (probability >= 0.5).astype(int)
    pred_opt = (probability >= THRESHOLD).astype(int)
    table = test[META].copy()
    table["prob_match"] = probability
    table["pred_05"] = pred_05
    table["pred_opt"] = pred_opt
    table["correct_05"] = (pred_05 == y_test).astype(int)
    table["correct_opt"] = (pred_opt == y_test).astype(int)
    table.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"rows {len(table)}  AUC {roc_auc_score(y_test, probability):.10f}")
    print(f"saved {OUT}")
    print(table[["pair_id", "virus_name", "serum_name", "r1_value", "r1_binary", "prob_match", "pred_opt", "correct_opt"]].to_string(index=False))


if __name__ == "__main__":
    main()
