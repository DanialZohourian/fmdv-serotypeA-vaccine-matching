# -*- coding: utf-8 -*-
"""SHAP explanations for the final serotype A random forest."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "step7_shap"
SELECTED_PATH = ROOT / "step5_boruta" / "selected_features.txt"
MATRIX_PATH = ROOT / "step5_boruta" / "feature_matrix_selected.csv"
SPLIT_PATH = ROOT / "step5_boruta" / "train_test_split.csv"
PREDS_PATH = ROOT / "step6_model" / "rf_test_predictions.csv"
BORUTA_PATH = ROOT / "step5_boruta" / "boruta_feature_selection.csv"
CASE_PATH = ROOT / "step6_model" / "rf_case_study_iran05.csv"
META = [
    "pair_id", "virus_name", "serum_name", "r1_value", "r1_binary",
    "lineage", "sub_lineage", "country", "year_isolated",
]
# Importance numbers supplied in the stage-7 brief for the serotype O comparison.
MAKAU_SUPPLIED = {
    "site_048_diff": 0.0250,
    "site_100_diff": 0.0220,
    "site_135_diff": 0.0200,
    "site_150_diff": 0.0180,
    "site_151_diff": 0.0170,
}
VP1_REGIONS = [
    (1, 40, "N-terminal", "#bdc3c7"),
    (40, 55, "BC loop (S1)", "#3498db"),
    (55, 70, "CD loop", "#95a5a6"),
    (70, 110, "EF loop (S2)", "#e67e22"),
    (110, 132, "FG region", "#95a5a6"),
    (132, 162, "GH loop (S3)", "#e74c3c"),
    (162, 195, "Other", "#bdc3c7"),
    (195, 213, "C-terminus", "#9b59b6"),
]


def feature_type(name):
    if name == "pairwise_aa_distance":
        return "distance"
    if "nglyc" in name:
        return "glycan"
    return "site"


def region_name(position):
    if 132 <= position <= 161:
        return "G-H loop"
    if 70 <= position <= 110:
        return "EF loop"
    if 40 <= position <= 55:
        return "BC loop"
    if position >= 195:
        return "C-terminus"
    return "Other"


def region_color(position):
    for start, end, _name, color in VP1_REGIONS:
        if start <= position < end or (end == 213 and position == 213):
            return color
    return "#d0d0d0"


def positive_shap(values, expected):
    """Return SHAP for the match class and its base value."""
    if hasattr(values, "values"):
        expected = values.base_values
        values = values.values
    array = np.asarray(values)
    base = np.asarray(expected)
    if array.ndim == 2:
        base_value = float(base.reshape(-1)[-1]) if base.size else float(base)
        return array, base_value
    if array.ndim == 3 and array.shape[-1] == 2:
        base_value = float(base[..., 1].reshape(-1)[0]) if base.ndim else float(base[1])
        return array[:, :, 1], base_value
    if array.ndim == 3 and array.shape[1] == 2:
        base_value = float(np.asarray(base)[1])
        return array[:, 1, :], base_value
    raise RuntimeError(f"unexpected SHAP shape {array.shape}")


def value_label(name, value):
    if name == "pairwise_aa_distance":
        return f"{value:.3f}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = [line.strip() for line in SELECTED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    predictions = pd.read_csv(PREDS_PATH, encoding="utf-8-sig")
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
    print(f"model rebuilt  train {x_train.shape}  test {x_test.shape}", flush=True)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    shap_test, base_value = positive_shap(explainer.shap_values(x_test, check_additivity=False), explainer.expected_value)
    x_all = np.vstack([x_train_raw, x_test])
    y_all = np.concatenate([y_train_raw, y_test])
    shap_all, _base_all = positive_shap(explainer.shap_values(x_all, check_additivity=False), explainer.expected_value)
    reconstructed = base_value + shap_test.sum(axis=1)
    predicted = model.predict_proba(x_test)[:, 1]
    gap = float(np.max(np.abs(reconstructed - predicted)))
    print(f"base value {base_value:.4f}  max |SHAP sum - probability| {gap:.4f}", flush=True)
    if gap > 0.05:
        raise SystemExit("SHAP values do not add up to the match probability")

    shap_table = pd.DataFrame(shap_test, columns=[f"shap_{name}" for name in selected])
    shap_table.insert(0, "r1_binary", y_test)
    shap_table.insert(0, "pair_id", test_df["pair_id"].to_numpy())
    shap_table.to_csv(OUT / "shap_test_values.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame({
        "feature": selected,
        "mean_abs_shap": np.abs(shap_all).mean(axis=0),
        "type": [feature_type(name) for name in selected],
    }).sort_values("mean_abs_shap", ascending=True)
    colors = {"distance": "#3498db", "site": "#2ecc71", "glycan": "#e67e22"}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(importance)), importance["mean_abs_shap"], color=[colors[kind] for kind in importance["type"]], edgecolor="white")
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance["feature"], fontsize=8)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global feature importance — SHAP")
    ax.legend(handles=[mpatches.Patch(color=color, label=label) for label, color in colors.items()], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "shap_bar_global.png", dpi=200)
    plt.close(fig)
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(
        OUT / "shap_global_importance.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_all, x_all, feature_names=selected, show=False, max_display=len(selected),
    )
    plt.title("SHAP beeswarm — FMDV serotype A VP1")
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    top5 = importance.sort_values("mean_abs_shap", ascending=False)["feature"].head(5).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat_axes = axes.flatten()
    point_colors = np.where(y_all == 1, "#e74c3c", "#3498db")
    rng = np.random.default_rng(42)
    for axis, name in zip(flat_axes, top5):
        index = selected.index(name)
        jitter = rng.uniform(-0.03, 0.03, len(y_all)) if feature_type(name) != "distance" else 0
        axis.scatter(x_all[:, index] + jitter, shap_all[:, index], c=point_colors, alpha=0.65, s=28, edgecolors="none")
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("Feature value")
        axis.set_ylabel("SHAP value")
        axis.legend(handles=[
            mpatches.Patch(color="#e74c3c", label="Match"),
            mpatches.Patch(color="#3498db", label="No-match"),
        ], fontsize=7)
    flat_axes[-1].set_visible(False)
    fig.suptitle("SHAP dependence — five strongest features")
    fig.tight_layout()
    fig.savefig(OUT / "shap_dependence_top5.png", dpi=200)
    plt.close(fig)

    merged = test_df.merge(predictions[["pair_id", "prob_match", "pred_opt", "correct_opt"]], on="pair_id", validate="one_to_one")
    examples = []
    true_pos = merged[(merged["r1_binary"] == 1) & (merged["correct_opt"] == 1)].sort_values("prob_match", ascending=False)
    true_neg = merged[(merged["r1_binary"] == 0) & (merged["correct_opt"] == 1)].sort_values("prob_match", ascending=True)
    false_neg = merged[(merged["r1_binary"] == 1) & (merged["correct_opt"] == 0)]
    iran_fp = merged[(merged["virus_name"] == "A/IRN/23/2018") & (merged["serum_name"] == "A/Iran-05")]
    if len(true_pos):
        examples.append(("True match", true_pos.iloc[0]))
    if len(true_neg):
        examples.append(("True no-match", true_neg.iloc[0]))
    if len(false_neg):
        examples.append(("Missed match", false_neg.iloc[0]))
    if len(iran_fp):
        examples.append(("Iran-05 false match", iran_fp.iloc[0]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for axis, (label, row) in zip(axes.flatten(), examples):
        local_idx = int(test_df.index[test_df["pair_id"] == row["pair_id"]][0])
        order = np.argsort(np.abs(shap_test[local_idx]))[::-1][:10]
        signed = shap_test[local_idx, order]
        axis.barh(
            range(len(order)), signed[::-1],
            color=["#e74c3c" if value > 0 else "#3498db" for value in signed[::-1]],
            edgecolor="white",
        )
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(
            [f"{selected[i]} ({value_label(selected[i], x_test[local_idx, i])})" for i in order[::-1]],
            fontsize=7,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("SHAP value")
        axis.set_title(
            f"{label}\n{row['virus_name']} vs {row['serum_name']}\n"
            f"r1={row['r1_value']:.2f} | actual={'match' if int(row['r1_binary']) == 1 else 'no-match'} | "
            f"probability={predicted[local_idx]:.2f}",
            fontsize=8,
        )
    for axis in axes.flatten()[len(examples):]:
        axis.set_visible(False)
    fig.suptitle("SHAP contributions for four test pairs")
    fig.tight_layout()
    fig.savefig(OUT / "shap_waterfall_examples.png", dpi=200)
    plt.close(fig)

    shap_per_site = {
        int(name.split("_")[1]): float(np.abs(shap_all[:, selected.index(name)]).mean())
        for name in selected if name.startswith("site_")
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 2]})
    for start, end, name, color in VP1_REGIONS:
        ax1.barh(0, end - start, left=start, color=color, edgecolor="white", height=0.6)
        ax1.text((start + end) / 2, 0, name, ha="center", va="center", fontsize=7)
    ax1.set_xlim(1, 213)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axis("off")
    ax1.set_title("VP1 regions on the trimmed alignment")
    positions = list(range(1, 214))
    ax2.bar(positions, [shap_per_site.get(pos, 0) for pos in positions], color=[region_color(pos) for pos in positions], width=1.0)
    for pos, value in shap_per_site.items():
        ax2.annotate(str(pos), xy=(pos, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
    ax2.set_xlim(1, 213)
    ax2.set_xlabel("Trimmed alignment column")
    ax2.set_ylabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(OUT / "vp1_position_map.png", dpi=200)
    plt.close(fig)

    def residue_list(items):
        return "+".join(str(pos) for pos in sorted(items))

    high = {pos: value for pos, value in shap_per_site.items() if value > 0.02}
    medium = {pos: value for pos, value in shap_per_site.items() if 0.01 <= value <= 0.02}
    low = {pos: value for pos, value in shap_per_site.items() if value < 0.01}
    pml = [
        "# Alignment columns from the trimmed 213-position VP1 alignment.",
        "# These numbers are not a verified mapping onto PDB 2WZR residue numbers.",
        "fetch 2WZR",
        "hide everything",
        "show cartoon, chain D",
        "color gray80, chain D",
        "",
    ]
    for label, group, color in (("high_importance", high, "red"), ("med_importance", medium, "orange"), ("low_importance", low, "yellow")):
        if group:
            pml += [
                f"select {label}, chain D and resi {residue_list(group)}",
                f"show sticks, {label}",
                f"color {color}, {label}",
                "",
            ]
    pml += [
        "select gh_loop, chain D and resi 132-161",
        "color salmon, gh_loop",
        "zoom chain D",
    ]
    (OUT / "color_vp1_positions.pml").write_text("\n".join(pml) + "\n", encoding="utf-8")

    decisions = pd.read_csv(BORUTA_PATH, encoding="utf-8-sig")
    decision_of = dict(zip(decisions["feature"], decisions["boruta_decision"]))
    rows = []
    for feature, makau_value in MAKAU_SUPPLIED.items():
        position = int(feature.split("_")[1])
        if feature in selected:
            status = "Confirmed"
            our_value = float(np.abs(shap_all[:, selected.index(feature)]).mean())
        elif feature in decision_of:
            status = decision_of[feature]
            our_value = 0.0
        else:
            status = "Constant — removed before Boruta"
            our_value = 0.0
        rows.append({
            "position": position,
            "region": region_name(position),
            "important_in_O": "yes",
            "makau_importance_supplied": makau_value,
            "status_in_A": status,
            "our_shap_A": our_value,
        })
    for feature in selected:
        if not feature.startswith("site_") or feature in MAKAU_SUPPLIED:
            continue
        position = int(feature.split("_")[1])
        rows.append({
            "position": position,
            "region": region_name(position),
            "important_in_O": "no",
            "makau_importance_supplied": np.nan,
            "status_in_A": "Confirmed",
            "our_shap_A": float(np.abs(shap_all[:, selected.index(feature)]).mean()),
        })
    comparison = pd.DataFrame(rows).sort_values("position")
    comparison.to_csv(OUT / "serotype_comparison_table.csv", index=False, encoding="utf-8-sig")

    case = pd.read_csv(CASE_PATH, encoding="utf-8-sig")
    iran_idx = test_df.index[test_df["pair_id"].isin(case["pair_id"])].to_numpy()
    shap_iran = shap_test[iran_idx]
    y_iran = y_test[iran_idx]
    case_table = pd.DataFrame({
        "feature": selected,
        "mean_shap_match": shap_iran[y_iran == 1].mean(axis=0),
        "mean_shap_nomatch": shap_iran[y_iran == 0].mean(axis=0),
    })
    case_table["diff"] = case_table["mean_shap_match"] - case_table["mean_shap_nomatch"]
    case_table = case_table.sort_values("diff", key=np.abs, ascending=False)
    case_table.to_csv(OUT / "shap_case_study_iran05.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 7))
    xpos = np.arange(len(case_table))
    ax.bar(xpos - 0.2, case_table["mean_shap_match"], 0.4, label="Match", color="#e74c3c")
    ax.bar(xpos + 0.2, case_table["mean_shap_nomatch"], 0.4, label="No-match", color="#3498db")
    ax.set_xticks(xpos)
    ax.set_xticklabels(case_table["feature"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean SHAP value")
    ax.set_title(f"A/ASIA/Iran-05 test pairs (n={len(iran_idx)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "shap_case_study_iran05.png", dpi=200)
    plt.close(fig)

    fp = test_df[(test_df["virus_name"] == "A/IRN/23/2018") & (test_df["serum_name"] == "A/Iran-05")]
    fp_lines = []
    if len(fp):
        fp_idx = int(fp.index[0])
        fp_table = pd.DataFrame({"feature": selected, "shap": shap_test[fp_idx], "value": x_test[fp_idx]})
        fp_prob = float(predictions.loc[predictions["pair_id"] == fp.iloc[0]["pair_id"], "prob_match"].iloc[0])
        fp_lines = [
            "",
            "--- A/IRN/23/2018 vs A/Iran-05 ---",
            f"Actual r1={fp.iloc[0]['r1_value']} (no-match) | model probability={fp_prob:.3f}",
            "Largest positive SHAP (toward match):",
        ]
        for _, item in fp_table.nlargest(5, "shap").iterrows():
            fp_lines.append(f"  {item['feature']}: shap={item['shap']:.4f}, value={value_label(item['feature'], item['value'])}")
        fp_lines.append("Largest negative SHAP (toward no-match):")
        for _, item in fp_table.nsmallest(5, "shap").iterrows():
            fp_lines.append(f"  {item['feature']}: shap={item['shap']:.4f}, value={value_label(item['feature'], item['value'])}")
        print("\n".join(fp_lines), flush=True)

    classic, inverse = [], []
    for name in selected:
        if not name.startswith("site_"):
            continue
        index = selected.index(name)
        values = x_all[:, index]
        differs = values >= 0.5
        if differs.any() and (~differs).any():
            effect = float(shap_all[differs, index].mean() - shap_all[~differs, index].mean())
        else:
            effect = float(shap_all[:, index].mean())
        (inverse if effect > 0 else classic).append((name, effect))
    ranked = importance.sort_values("mean_abs_shap", ascending=False)
    report = [
        "=== SHAP Interpretability Report ===",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        f"Base value on the probability scale: {base_value:.4f}",
        "SHAP values are the change in predicted match probability.",
        "Global values use the 190 real pairs, not the synthetic SMOTE rows.",
        "",
        "--- Global ranking (mean absolute SHAP) ---",
    ]
    for _, row in ranked.head(10).iterrows():
        report.append(f"  {row['feature']:30s} {row['mean_abs_shap']:.5f}  [{row['type']}]")
    report += ["", "A site difference lowers the match probability:"]
    report += [f"  {name}  effect={effect:.4f}" for name, effect in sorted(classic, key=lambda item: item[1])]
    report += ["", "A site difference raises the match probability:"]
    report += [f"  {name}  effect={effect:.4f}" for name, effect in sorted(inverse, key=lambda item: -item[1])]
    report += [
        "",
        "--- Iran-05 test subset ---",
        f"Pairs: {len(case)}",
        f"False matches: {int(((case['r1_binary'] == 0) & (case['pred_opt'] == 1)).sum())}",
    ]
    report += fp_lines
    report += [
        "",
        "Makau serotype O importance numbers in serotype_comparison_table.csv",
        "are the values supplied for this comparison, not re-extracted from the PDF here.",
        "Position numbers are trimmed alignment columns.",
    ]
    (OUT / "shap_interpretation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    required = [
        "shap_test_values.csv", "shap_global_importance.csv", "shap_bar_global.png",
        "shap_beeswarm.png", "shap_dependence_top5.png", "shap_waterfall_examples.png",
        "vp1_position_map.png", "color_vp1_positions.pml", "serotype_comparison_table.csv",
        "shap_case_study_iran05.csv", "shap_case_study_iran05.png", "shap_interpretation_report.txt",
    ]
    top1 = ranked.iloc[0]["feature"]
    checks = {name: (OUT / name).exists() for name in required}
    checks["shap_test_values rows = 38"] = len(shap_table) == 38
    checks["global importance has 23 features"] = len(ranked) == 23
    checks[f"top feature is interpretable ({top1})"] = ("site_149" in top1) or ("pairwise" in top1) or ("171" in top1)
    print("\nCHECKS", flush=True)
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}", flush=True)
    print(f"READY {all(checks.values())}", flush=True)
    print("\nTOP", flush=True)
    print(ranked.head(8).to_string(index=False), flush=True)
    print("\nCLASSIC", classic, flush=True)
    print("INVERSE", inverse, flush=True)


if __name__ == "__main__":
    main()
