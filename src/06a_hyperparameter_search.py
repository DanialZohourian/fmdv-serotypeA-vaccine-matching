# -*- coding: utf-8 -*-
"""Stage 6.1-6.2: split check and stratified CV for RF, XGBoost, and LightGBM."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "step6_model"
SELECTED_PATH = ROOT / "step5_boruta" / "selected_features.txt"
MATRIX_PATH = ROOT / "step5_boruta" / "feature_matrix_selected.csv"
SPLIT_PATH = ROOT / "step5_boruta" / "train_test_split.csv"


def load_split():
    selected = [line.strip() for line in SELECTED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    missing = [name for name in selected if name not in matrix.columns]
    if missing:
        raise SystemExit(f"selected features missing from matrix: {missing}")
    train_ids = set(split.loc[split["split"] == "train", "pair_id"])
    test_ids = set(split.loc[split["split"] == "test", "pair_id"])
    if train_ids & test_ids:
        raise SystemExit("train and test pair_ids overlap")
    train_df = matrix[matrix["pair_id"].isin(train_ids)].reset_index(drop=True)
    test_df = matrix[matrix["pair_id"].isin(test_ids)].reset_index(drop=True)
    x_train_raw = train_df[selected].to_numpy(dtype=np.float64)
    y_train_raw = train_df["r1_binary"].to_numpy(dtype=int)
    x_test = test_df[selected].to_numpy(dtype=np.float64)
    y_test = test_df["r1_binary"].to_numpy(dtype=int)
    print(f"features: {len(selected)}", flush=True)
    print(f"Train: {x_train_raw.shape} | match={int(y_train_raw.sum())}", flush=True)
    print(f"Test:  {x_test.shape} | match={int(y_test.sum())}", flush=True)
    smote = SMOTE(k_neighbors=5, random_state=42)
    x_train, y_train = smote.fit_resample(x_train_raw, y_train_raw)
    print(
        f"Train after SMOTE: {x_train.shape} | match={int(y_train.sum())} | no-match={int((y_train == 0).sum())}",
        flush=True,
    )
    if x_train.shape != (228, 23) or x_test.shape != (38, 23) or int(y_test.sum()) != 10:
        raise SystemExit("split sizes do not match the stage-6 gate")
    return x_train_raw, y_train_raw, selected


def search(model_name, grid, builder, x_train_raw, y_train_raw):
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"\n{model_name}: {len(combos)} combinations x 10 folds", flush=True)
    folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_aucs = []
        for train_idx, val_idx in folder.split(x_train_raw, y_train_raw):
            x_tr, y_tr = x_train_raw[train_idx], y_train_raw[train_idx]
            x_val, y_val = x_train_raw[val_idx], y_train_raw[val_idx]
            k = min(5, int(y_tr.sum()) - 1)
            x_tr_s, y_tr_s = SMOTE(k_neighbors=k, random_state=42).fit_resample(x_tr, y_tr)
            x_tr_s = np.ascontiguousarray(x_tr_s, dtype=np.float64)
            y_tr_s = np.ascontiguousarray(y_tr_s, dtype=np.int32)
            model = builder(params)
            model.fit(x_tr_s, y_tr_s)
            probability = model.predict_proba(x_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, probability))
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        rows.append({**params, "cv_auc_mean": mean_auc, "cv_auc_std": std_auc})
        print(f"  {params} -> AUC={mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    results = pd.DataFrame(rows).sort_values(["cv_auc_mean", "cv_auc_std"], ascending=[False, True])
    path = OUT / f"{model_name.lower()}_cv_results.csv"
    results.to_csv(path, index=False, encoding="utf-8-sig")
    best = results.iloc[0]
    print(f"BEST {model_name}: AUC={best['cv_auc_mean']:.4f} +/- {best['cv_auc_std']:.4f}", flush=True)
    print(best.drop(labels=["cv_auc_mean", "cv_auc_std"]).to_dict(), flush=True)
    return results


def rf_builder(params):
    return RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1, random_state=42)


def xgb_builder(params):
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


def lgb_builder(params):
    return LGBMClassifier(
        **params,
        objective="binary",
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=42,
        force_row_wise=True,
        verbose=-1,
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x_train_raw, y_train_raw, _selected = load_split()
    grids = {
        "rf": (
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", 0.5],
                "min_samples_leaf": [1, 2],
            },
            rf_builder,
        ),
        "xgb": (
            {
                "n_estimators": [300, 500],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.05, 0.1],
                "min_child_weight": [1, 5],
            },
            xgb_builder,
        ),
        "lgbm": (
            {
                "n_estimators": [300, 500],
                "max_depth": [-1, 6, 10],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31],
            },
            lgb_builder,
        ),
    }
    import subprocess
    import sys

    summaries = []
    for name, (grid, builder) in grids.items():
        if name == "lgbm":
            subprocess.run([sys.executable, str(ROOT / "step6_lgbm_only.py")], check=True)
            table = pd.read_csv(OUT / "lgbm_cv_results.csv", encoding="utf-8-sig")
        else:
            table = search(name, grid, builder, x_train_raw, y_train_raw)
        best = table.iloc[0].to_dict()
        best["model"] = name
        summaries.append(best)
    pd.DataFrame(summaries).to_csv(OUT / "cv_best_summary.csv", index=False, encoding="utf-8-sig")
    print("\nsaved step6_model CV tables", flush=True)
