# -*- coding: utf-8 -*-
"""Stage 5: stratified split, SMOTE on train only, then Boruta."""
import inspect
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from boruta import BorutaPy
from boruta.boruta_py import BorutaPy as _BorutaPy
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "step5_boruta"
ML_PATH = ROOT / "step4_features" / "feature_matrix_ml.csv"
REPORT_PATH = ROOT / "step4_features" / "feature_engineering_report.csv"
META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
]
BIO_MAP = {
    range(40, 55): "Site 1 (BC loop)",
    range(55, 70): "C-D loop",
    range(70, 110): "Site 2 (EF loop)",
    range(132, 162): "G-H loop",
    range(195, 214): "C-terminus",
}
O_IMPORTANT = [48, 100, 135, 150, 151]


def _fit_without_builtin_rough_fix(self, X, y):
    """Boruta _fit that keeps every unresolved feature tentative.

    Boruta 0.4.3 otherwise drops tentatives whose median importance does not
    beat the median shadow maximum. Those features are resolved afterwards
    with a Mann-Whitney test against the shadow-maximum history.
    """
    source = textwrap.dedent(inspect.getsource(_BorutaPy._fit))
    old = """\
    # we automatically apply R package's rough fix for tentative ones
    confirmed = np.where(dec_reg == 1)[0]
    tentative = np.where(dec_reg == 0)[0]
    # ignore the first row of zeros
    tentative_median = np.median(imp_history[1:, tentative], axis=0)
    # which tentative to keep
    tentative_confirmed = np.where(tentative_median
                                   > np.median(sha_max_history))[0]
    tentative = tentative[tentative_confirmed]
"""
    new = """\
    confirmed = np.where(dec_reg == 1)[0]
    tentative = np.where(dec_reg == 0)[0]
    self.sha_max_history_ = np.asarray(sha_max_history, dtype=float)
"""
    if old not in source:
        raise RuntimeError("Boruta _fit source no longer matches the expected rough-fix block")
    local_namespace = {"np": np}
    exec(source.replace(old, new), _BorutaPy._fit.__globals__, local_namespace)
    return local_namespace["_fit"](self, X, y)


class BorutaPyTracked(BorutaPy):
    _fit = _fit_without_builtin_rough_fix


def feature_type(name):
    if name == "pairwise_aa_distance":
        return "distance"
    if "nglyc" in name:
        return "glycan"
    return "site"


def region_of(position):
    for positions, name in BIO_MAP.items():
        if position in positions:
            return name
    return "Other"


def tentative_rough_fix(boruta, feature_names):
    history = boruta.importance_history_[1:]
    shadow_max = boruta.sha_max_history_
    resolved = {}
    for idx in np.where(boruta.support_weak_)[0]:
        feat_imp = history[:, idx]
        feat_imp = feat_imp[~np.isnan(feat_imp)]
        try:
            _, p_value = mannwhitneyu(feat_imp, shadow_max, alternative="greater")
        except ValueError:
            p_value = 1.0
        if np.isnan(p_value):
            p_value = 1.0
        decision = "Confirmed" if p_value < 0.05 else "Rejected"
        resolved[feature_names[idx]] = {"p_value": float(p_value), "decision": decision}
        print(f"  {feature_names[idx]}: p={p_value:.4f} -> {decision}", flush=True)
    return resolved


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ml = pd.read_csv(ML_PATH, encoding="utf-8-sig").reset_index(drop=True)
    feature_cols = [col for col in ml.columns if col not in META]
    site_cols = [col for col in feature_cols if col.startswith("site_")]
    nglyc_cols = [col for col in feature_cols if col.startswith("nglyc_")]
    print(f"features: {len(feature_cols)}", flush=True)
    print(f"  distance: {int('pairwise_aa_distance' in feature_cols)}", flush=True)
    print(f"  site:     {len(site_cols)}", flush=True)
    print(f"  nglyc:    {len(nglyc_cols)}", flush=True)

    x = ml[feature_cols].to_numpy(dtype=np.float64)
    y = ml["r1_binary"].to_numpy(dtype=int)
    feature_names = np.array(feature_cols)
    if np.isnan(x).any():
        raise SystemExit("X contains NaN")
    print(f"X {x.shape}  y {y.shape}", flush=True)
    print(f"class 0: {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)", flush=True)
    print(f"class 1: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)", flush=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y))
    x_train_raw, x_test = x[train_idx], x[test_idx]
    y_train_raw, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}", flush=True)
    print(f"Train no-match {(y_train_raw == 0).sum()} match {(y_train_raw == 1).sum()}", flush=True)
    print(f"Test  no-match {(y_test == 0).sum()} match {(y_test == 1).sum()}", flush=True)

    split = np.full(len(ml), "train", dtype=object)
    split[test_idx] = "test"
    split_info = ml[META].copy()
    split_info["split"] = split
    split_info[["pair_id", "virus_name", "serum_name", "r1_binary", "split"]].to_csv(
        OUT / "train_test_split.csv", index=False, encoding="utf-8-sig"
    )

    smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
    x_train, y_train = smote.fit_resample(x_train_raw, y_train_raw)
    print(f"After SMOTE: {len(x_train)}  no-match {(y_train == 0).sum()}  match {(y_train == 1).sum()}", flush=True)
    pd.DataFrame({
        "set": ["train_before", "train_before", "train_after", "train_after"],
        "class": ["no-match", "match", "no-match", "match"],
        "count": [
            int((y_train_raw == 0).sum()), int((y_train_raw == 1).sum()),
            int((y_train == 0).sum()), int((y_train == 1).sum()),
        ],
        "percent": [
            float((y_train_raw == 0).mean() * 100), float((y_train_raw == 1).mean() * 100),
            float((y_train == 0).mean() * 100), float((y_train == 1).mean() * 100),
        ],
    }).to_csv(OUT / "smote_summary.csv", index=False, encoding="utf-8-sig")

    rf = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", max_depth=5, random_state=42
    )
    boruta = BorutaPyTracked(
        estimator=rf, n_estimators="auto", perc=100, max_iter=150,
        verbose=2, random_state=42,
    )
    print(f"Running Boruta on {x_train.shape}", flush=True)
    boruta.fit(x_train, y_train)
    # If nothing is rejected, BorutaPy marks every feature confirmed, including
    # features that were still tentative. Put those back in the tentative set.
    marked_both = boruta.support_ & boruta.support_weak_
    boruta.support_[marked_both] = False
    n_confirmed = int(boruta.support_.sum())
    n_tentative = int(boruta.support_weak_.sum())
    n_rejected = int((~boruta.support_ & ~boruta.support_weak_).sum())
    print(f"Confirmed {n_confirmed}  Tentative {n_tentative}  Rejected {n_rejected}", flush=True)

    print("Resolving tentative features", flush=True)
    resolved = tentative_rough_fix(boruta, feature_names) if n_tentative else {}
    if not n_tentative:
        print("No tentative features", flush=True)

    decisions = []
    for i, name in enumerate(feature_names):
        if boruta.support_[i]:
            decision = "Confirmed"
        elif boruta.support_weak_[i]:
            decision = resolved.get(name, {}).get("decision", "Tentative")
        else:
            decision = "Rejected"
        decisions.append({
            "feature": name,
            "type": feature_type(name),
            "boruta_ranking": int(boruta.ranking_[i]),
            "boruta_decision": decision,
            "tentative_p_value": resolved.get(name, {}).get("p_value", np.nan),
        })
    decisions_df = pd.DataFrame(decisions)
    engineering = pd.read_csv(REPORT_PATH, encoding="utf-8-sig")
    decisions_df = decisions_df.merge(
        engineering[["feature", "diff_by_class", "mean_in_match", "mean_in_nomatch"]],
        on="feature", how="left",
    )
    decisions_df.to_csv(OUT / "boruta_feature_selection.csv", index=False, encoding="utf-8-sig")

    selected = decisions_df.loc[decisions_df["boruta_decision"] == "Confirmed", "feature"].tolist()
    distance_forced = False
    if "pairwise_aa_distance" not in selected:
        selected = ["pairwise_aa_distance"] + selected
        distance_forced = True
        print("pairwise_aa_distance was not Confirmed and was kept because it is the overall VP1 distance", flush=True)
    (OUT / "selected_features.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"selected: {len(selected)}", flush=True)
    for kind in ("distance", "site", "glycan"):
        print(f"  {kind}: {sum(feature_type(name) == kind for name in selected)}", flush=True)
    for name in selected:
        diff = decisions_df.loc[decisions_df["feature"] == name, "diff_by_class"].iloc[0]
        print(f"  {name}  diff={diff:.3f}", flush=True)

    print("Selected site regions (trimmed alignment columns)", flush=True)
    region_features = {}
    for name in selected:
        if not name.startswith("site_"):
            continue
        region_features.setdefault(region_of(int(name.split("_")[1])), []).append(name)
    for region, names in sorted(region_features.items(), key=lambda item: -len(item[1])):
        avg = decisions_df.loc[decisions_df["feature"].isin(names), "diff_by_class"].mean()
        print(f"  {region}: {len(names)} sites (avg diff={avg:.3f})", flush=True)

    print("Serotype O positions from Makau 2025, looked up on these alignment columns", flush=True)
    known = set(feature_names)
    for pos in O_IMPORTANT:
        feat = f"site_{pos:03d}_diff"
        if feat in selected and decisions_df.loc[decisions_df["feature"] == feat, "boruta_decision"].iloc[0] == "Confirmed":
            diff = decisions_df.loc[decisions_df["feature"] == feat, "diff_by_class"].iloc[0]
            print(f"  position {pos}: Confirmed in A (diff={diff:.3f})", flush=True)
        elif feat in known:
            decision = decisions_df.loc[decisions_df["feature"] == feat, "boruta_decision"].iloc[0]
            print(f"  position {pos}: {decision} in A", flush=True)
        else:
            print(f"  position {pos}: constant in this dataset, removed before Boruta", flush=True)

    selected_matrix = ml[META + selected].copy()
    selected_matrix["split"] = split
    selected_matrix.to_csv(OUT / "feature_matrix_selected.csv", index=False, encoding="utf-8-sig")
    print(f"selected matrix {selected_matrix.shape}", flush=True)
    print(selected_matrix["split"].value_counts().to_string(), flush=True)
    print(selected_matrix.groupby("split")["r1_binary"].value_counts().to_string(), flush=True)

    history = boruta.importance_history_[1:]
    plot_df = decisions_df.copy()
    plot_df["median_importance"] = np.nanmedian(history, axis=0)
    plot_df = plot_df.sort_values("median_importance", ascending=True)
    color_map = {"Confirmed": "#2ecc71", "Tentative": "#f39c12", "Rejected": "#e74c3c"}
    colors = [color_map.get(decision, "#95a5a6") for decision in plot_df["boruta_decision"]]
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(range(len(plot_df)), plot_df["median_importance"], color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"], fontsize=6)
    ax.set_xlabel("Median feature importance (Boruta)")
    ax.set_title("Boruta feature selection — FMDV serotype A VP1")
    ax.legend(handles=[mpatches.Patch(color=color, label=label) for label, color in color_map.items()], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "boruta_importance_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    checks = {
        "train_test_split.csv has split": "split" in pd.read_csv(OUT / "train_test_split.csv", encoding="utf-8-sig").columns,
        "test set size is 38": int((split == "test").sum()) == 38,
        "boruta file has 96 features": len(decisions_df) == 96,
        "at least one Confirmed": int((decisions_df["boruta_decision"] == "Confirmed").sum()) > 0,
        "selected matrix has 190 rows": len(selected_matrix) == 190,
        "selected matrix has split": "split" in selected_matrix.columns,
        "selected_features.txt exists": (OUT / "selected_features.txt").exists(),
        "importance plot exists": (OUT / "boruta_importance_plot.png").exists(),
    }
    print("FINAL CHECKS", flush=True)
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}", flush=True)
    n_final_confirmed = int((decisions_df["boruta_decision"] == "Confirmed").sum())
    print(f"Confirmed after tentative resolution: {n_final_confirmed}", flush=True)
    print(f"distance forced in: {distance_forced}", flush=True)
    print(f"READY {all(checks.values()) and n_final_confirmed >= 3}", flush=True)


if __name__ == "__main__":
    main()
