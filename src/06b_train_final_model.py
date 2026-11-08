# -*- coding: utf-8 -*-
"""Train the final random forest and evaluate it once on the held-out test set."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "step6_model"
SELECTED_PATH = ROOT / "step5_boruta" / "selected_features.txt"
MATRIX_PATH = ROOT / "step5_boruta" / "feature_matrix_selected.csv"
FULL_PATH = ROOT / "step4_features" / "feature_matrix_ml.csv"
SPLIT_PATH = ROOT / "step5_boruta" / "train_test_split.csv"
CV_PATH = OUT / "rf_cv_results.csv"
META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
]


def compute_metrics(y_true, y_pred, y_prob, label):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    constant = np.unique(y_prob).size < 2
    auc = 0.5 if constant else roc_auc_score(y_true, y_prob)
    ap = float(y_true.mean()) if constant else average_precision_score(y_true, y_prob)
    return {
        "threshold": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": auc,
        "auc_pr": ap,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_test": int(len(y_true)),
        "n_match": int(y_true.sum()),
        "n_no_match": int((y_true == 0).sum()),
    }


def make_forest(n_estimators, max_depth, max_features, min_samples_leaf):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def main():
    selected = [line.strip() for line in SELECTED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")
    full = pd.read_csv(FULL_PATH, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    cv = pd.read_csv(CV_PATH, encoding="utf-8-sig")
    best = cv.iloc[0]
    n_estimators = int(best["n_estimators"])
    max_depth = None if pd.isna(best["max_depth"]) else int(best["max_depth"])
    max_features = best["max_features"]
    if isinstance(max_features, float):
        max_features = float(max_features)
    min_leaf = int(best["min_samples_leaf"])

    train_ids = set(split.loc[split["split"] == "train", "pair_id"])
    test_ids = set(split.loc[split["split"] == "test", "pair_id"])
    train_df = matrix[matrix["pair_id"].isin(train_ids)].reset_index(drop=True)
    test_df = matrix[matrix["pair_id"].isin(test_ids)].reset_index(drop=True)
    x_train_raw = train_df[selected].to_numpy(dtype=np.float64)
    y_train_raw = train_df["r1_binary"].to_numpy(dtype=int)
    x_test = test_df[selected].to_numpy(dtype=np.float64)
    y_test = test_df["r1_binary"].to_numpy(dtype=int)

    oof = np.zeros(len(y_train_raw), dtype=float)
    folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in folder.split(x_train_raw, y_train_raw):
        k = min(5, int(y_train_raw[train_idx].sum()) - 1)
        x_res, y_res = SMOTE(k_neighbors=k, random_state=42).fit_resample(
            x_train_raw[train_idx], y_train_raw[train_idx]
        )
        fold_model = make_forest(n_estimators, max_depth, max_features, min_leaf)
        fold_model.fit(x_res, y_res)
        oof[val_idx] = fold_model.predict_proba(x_train_raw[val_idx])[:, 1]

    best_thresh = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (oof >= threshold).astype(int)
        score = f1_score(y_train_raw, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(threshold)
    print(f"OOF threshold: {best_thresh:.2f}  training F1={best_f1:.4f}", flush=True)

    x_train, y_train = SMOTE(k_neighbors=5, random_state=42).fit_resample(x_train_raw, y_train_raw)
    model = make_forest(n_estimators, max_depth, max_features, min_leaf)
    model.fit(x_train, y_train)
    print(
        f"final model n_estimators={n_estimators} max_depth={max_depth} "
        f"max_features={max_features} min_samples_leaf={min_leaf} n={len(x_train)}",
        flush=True,
    )

    probs_test = model.predict_proba(x_test)[:, 1]
    preds_05 = (probs_test >= 0.5).astype(int)
    preds_opt = (probs_test >= best_thresh).astype(int)
    metrics_05 = compute_metrics(y_test, preds_05, probs_test, "0.50")
    metrics_opt = compute_metrics(y_test, preds_opt, probs_test, f"{best_thresh:.2f}")
    pd.DataFrame([metrics_05, metrics_opt]).to_csv(OUT / "rf_test_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = test_df[META].copy()
    predictions["prob_match"] = probs_test
    predictions["pred_05"] = preds_05
    predictions["pred_opt"] = preds_opt
    predictions["correct_05"] = (preds_05 == y_test).astype(int)
    predictions["correct_opt"] = (preds_opt == y_test).astype(int)
    predictions.to_csv(OUT / "rf_test_predictions.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, preds_opt, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual: No-match", "Actual: Match"],
        columns=["Predicted: No-match", "Predicted: Match"],
    ).to_csv(OUT / "rf_confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No-match", "Match"], yticklabels=["No-match", "Match"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold={best_thresh:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "rf_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fpr, tpr, roc_thresholds = roc_curve(y_test, probs_test)
    auc_val = roc_auc_score(y_test, probs_test)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        OUT / "rf_roc_data.csv", index=False, encoding="utf-8-sig"
    )
    finite = np.isfinite(roc_thresholds)
    opt_idx = int(np.argmin(np.abs(np.where(finite, roc_thresholds, 1e9) - best_thresh)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f4e79", linewidth=2, label=f"Random forest (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Chance (AUC = 0.500)")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], s=80, c="#c0392b", zorder=5, label=f"Threshold {best_thresh:.2f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — FMDV serotype A VP1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rf_roc_curve.png", dpi=200)
    plt.close(fig)

    importances = model.feature_importances_
    spread = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    importance = pd.DataFrame({
        "feature": selected,
        "importance": importances,
        "std": spread,
        "type": [
            "distance" if name == "pairwise_aa_distance" else "glycan" if "nglyc" in name else "site"
            for name in selected
        ],
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_imp = importance.sort_values("importance", ascending=True)
    colors = {"distance": "#3498db", "site": "#2ecc71", "glycan": "#e67e22"}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        range(len(plot_imp)), plot_imp["importance"], xerr=plot_imp["std"],
        color=[colors[kind] for kind in plot_imp["type"]], edgecolor="white", capsize=3,
    )
    ax.set_yticks(range(len(plot_imp)))
    ax.set_yticklabels(plot_imp["feature"], fontsize=8)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Final random forest feature importance")
    ax.legend(handles=[mpatches.Patch(color=color, label=label) for label, color in colors.items()], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "rf_feature_importance.png", dpi=200)
    plt.close(fig)

    baselines = {}
    majority = DummyClassifier(strategy="most_frequent", random_state=42)
    majority.fit(x_train_raw, y_train_raw)
    baselines["Always No-match"] = compute_metrics(
        y_test, majority.predict(x_test), majority.predict_proba(x_test)[:, 1], "majority"
    )
    stratified = DummyClassifier(strategy="stratified", random_state=42)
    stratified.fit(x_train_raw, y_train_raw)
    baselines["Stratified Random"] = compute_metrics(
        y_test, stratified.predict(x_test), stratified.predict_proba(x_test)[:, 1], "stratified"
    )
    distance_col = selected.index("pairwise_aa_distance")
    distance_model = LogisticRegression(random_state=42, class_weight="balanced")
    distance_model.fit(x_train_raw[:, [distance_col]], y_train_raw)
    distance_prob = distance_model.predict_proba(x_test[:, [distance_col]])[:, 1]
    baselines["Distance Only (LR)"] = compute_metrics(
        y_test, distance_model.predict(x_test[:, [distance_col]]), distance_prob, "lr"
    )

    all_features = [col for col in full.columns if col not in META]
    full_train = full[full["pair_id"].isin(train_ids)].reset_index(drop=True)
    full_test = full[full["pair_id"].isin(test_ids)].reset_index(drop=True)
    if not full_train["pair_id"].equals(train_df["pair_id"]) or not full_test["pair_id"].equals(test_df["pair_id"]):
        full_train = train_df[["pair_id"]].merge(full, on="pair_id", how="left", validate="one_to_one")
        full_test = test_df[["pair_id"]].merge(full, on="pair_id", how="left", validate="one_to_one")
    x_all_train = full_train[all_features].to_numpy(dtype=np.float64)
    x_all_test = full_test[all_features].to_numpy(dtype=np.float64)
    x_all_s, y_all_s = SMOTE(k_neighbors=5, random_state=42).fit_resample(x_all_train, y_train_raw)
    all_model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    all_model.fit(x_all_s, y_all_s)
    all_prob = all_model.predict_proba(x_all_test)[:, 1]
    baselines["RF All 96 Features"] = compute_metrics(
        y_test, (all_prob >= best_thresh).astype(int), all_prob, "rf_all96"
    )

    comparison = pd.DataFrame([{"model": name, **values} for name, values in baselines.items()])
    comparison = pd.concat(
        [comparison, pd.DataFrame([{"model": "RF Boruta-23 (Final)", **metrics_opt}])],
        ignore_index=True,
    )
    comparison.to_csv(OUT / "rf_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    iran = predictions[predictions["lineage"].fillna("").str.contains("Iran-05")].copy()
    iran.to_csv(OUT / "rf_case_study_iran05.csv", index=False, encoding="utf-8-sig")
    print(f"Iran-05 test pairs: {len(iran)}", flush=True)
    if len(iran):
        print(iran[["virus_name", "serum_name", "year_isolated", "r1_value", "r1_binary", "prob_match", "pred_opt", "correct_opt"]].to_string(index=False), flush=True)
        y_iran = iran["r1_binary"].to_numpy(dtype=int)
        if y_iran.sum() > 0 and (y_iran == 0).sum() > 0:
            print(f"Iran-05 AUC {roc_auc_score(y_iran, iran['prob_match']):.4f}", flush=True)
            print(f"Iran-05 F1 {f1_score(y_iran, iran['pred_opt'], zero_division=0):.4f}", flush=True)

    summary = [
        "=== RF Model Summary — FMDV Serotype A VP1 ===",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "--- Dataset ---",
        f"Total pairs: 190 (train={len(train_df)}, test={len(test_df)})",
        f"Features: {len(selected)} (Boruta-selected from 96)",
        f"Test match: {int(y_test.sum())}/{len(y_test)} ({y_test.mean() * 100:.1f}%)",
        "",
        "--- Best Hyperparameters ---",
        f"n_estimators:     {n_estimators}",
        f"max_depth:        {max_depth}",
        f"max_features:     {max_features}",
        f"min_samples_leaf: {min_leaf}",
        "class_weight:     balanced",
        f"CV AUC (best):    {best['cv_auc_mean']:.4f} +/- {best['cv_auc_std']:.4f}",
        "Decision threshold was chosen by maximizing F1 on out-of-fold training predictions.",
        f"OOF threshold:    {best_thresh:.2f}",
        f"OOF F1:           {best_f1:.4f}",
        "",
        "--- Test Set Results (threshold=0.50) ---",
        f"AUC-ROC:     {metrics_05['auc_roc']:.4f}",
        f"AUC-PR:      {metrics_05['auc_pr']:.4f}",
        f"Accuracy:    {metrics_05['accuracy']:.4f}",
        f"Sensitivity: {metrics_05['sensitivity']:.4f}",
        f"Specificity: {metrics_05['specificity']:.4f}",
        f"PPV:         {metrics_05['ppv']:.4f}",
        f"NPV:         {metrics_05['npv']:.4f}",
        f"F1:          {metrics_05['f1']:.4f}",
        "",
        f"--- Test Set Results (threshold={best_thresh:.2f}, OOF F1) ---",
        f"AUC-ROC:     {metrics_opt['auc_roc']:.4f}",
        f"AUC-PR:      {metrics_opt['auc_pr']:.4f}",
        f"Accuracy:    {metrics_opt['accuracy']:.4f}",
        f"Sensitivity: {metrics_opt['sensitivity']:.4f}",
        f"Specificity: {metrics_opt['specificity']:.4f}",
        f"PPV:         {metrics_opt['ppv']:.4f}",
        f"NPV:         {metrics_opt['npv']:.4f}",
        f"F1:          {metrics_opt['f1']:.4f}",
        f"TP={metrics_opt['tp']} FP={metrics_opt['fp']} FN={metrics_opt['fn']} TN={metrics_opt['tn']}",
        "",
        "--- Comparison with Makau 2025 (Serotype O) ---",
        "Makau test accuracy: 0.75 | F1: 0.86 | AUC: 0.92",
        f"This model accuracy: {metrics_opt['accuracy']:.2f} | F1: {metrics_opt['f1']:.2f} | AUC: {metrics_opt['auc_roc']:.2f}",
        "",
        "--- Top 5 Features ---",
    ]
    for _, row in importance.head(5).iterrows():
        summary.append(f"  {row['feature']:30s} importance={row['importance']:.4f}")
    (OUT / "rf_model_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\nTEST threshold 0.50", flush=True)
    for key in ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "auc_roc", "auc_pr"):
        print(f"  {key}: {metrics_05[key]:.4f}", flush=True)
    print(f"\nTEST threshold {best_thresh:.2f}", flush=True)
    for key in ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "auc_roc", "auc_pr"):
        print(f"  {key}: {metrics_opt[key]:.4f}", flush=True)
    print(f"  TP={metrics_opt['tp']} FP={metrics_opt['fp']} FN={metrics_opt['fn']} TN={metrics_opt['tn']}", flush=True)
    print("\nBASELINES", flush=True)
    print(comparison[["model", "auc_roc", "f1", "sensitivity", "specificity", "accuracy"]].to_string(index=False), flush=True)
    print("\nTOP FEATURES", flush=True)
    print(importance.head(10)[["feature", "type", "importance", "std"]].to_string(index=False), flush=True)

    best_baseline_auc = max(row["auc_roc"] for row in baselines.values())
    checks = {
        "rf_cv_results.csv": CV_PATH.exists(),
        "rf_test_metrics.csv": (OUT / "rf_test_metrics.csv").exists(),
        "rf_test_predictions.csv": (OUT / "rf_test_predictions.csv").exists(),
        "rf_confusion_matrix.csv": (OUT / "rf_confusion_matrix.csv").exists(),
        "rf_roc_data.csv": (OUT / "rf_roc_data.csv").exists(),
        "rf_feature_importance.csv": (OUT / "rf_feature_importance.csv").exists(),
        "rf_baseline_comparison.csv": (OUT / "rf_baseline_comparison.csv").exists(),
        "rf_case_study_iran05.csv": (OUT / "rf_case_study_iran05.csv").exists(),
        f"AUC-ROC >= 0.70 ({metrics_opt['auc_roc']:.3f})": metrics_opt["auc_roc"] >= 0.70,
        f"F1 >= 0.50 ({metrics_opt['f1']:.3f})": metrics_opt["f1"] >= 0.50,
        f"Sensitivity >= 0.50 ({metrics_opt['sensitivity']:.3f})": metrics_opt["sensitivity"] >= 0.50,
        f"Specificity >= 0.60 ({metrics_opt['specificity']:.3f})": metrics_opt["specificity"] >= 0.60,
        "AUC above every baseline": metrics_opt["auc_roc"] > best_baseline_auc,
    }
    print("\nCHECKS", flush=True)
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}", flush=True)
    print(f"READY {all(checks.values())}", flush=True)


if __name__ == "__main__":
    main()
