"""Fig. 5 — A/ASIA/Iran-05 case study (10 held-out pairs).

Panel A : measured r1 against predicted probability of antigenic match for
          the 10 Iran-05 test pairs; the two discordant pairs are named.
Panel B : mean SHAP value per feature in true-match vs true-no-match
          Iran-05 pairs.
Panel C : additive SHAP decomposition (waterfall) of the discordant pair
          A/IRN/23/2018 vs A/Iran-05 antiserum.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

UP = "/mnt/user-data/uploads"
fs.apply_style()

shap = pd.read_csv(os.path.join(UP, "shap_test_values.csv"), encoding="utf-8-sig")
pred = pd.read_csv(os.path.join(UP, "rf_test_predictions.csv"), encoding="utf-8-sig")
case = pd.read_csv(os.path.join(UP, "shap_case_study_iran05.csv"), encoding="utf-8-sig")
cmp_tab = pd.read_csv("/home/claude/build/serotype_comparison_table_v2.csv", encoding="utf-8-sig")
region_lut = dict(zip(cmp_tab["position"], cmp_tab["region"]))

BASE = 0.4995          # SHAP base value, step-8 output
THR_OPT, THR_DEF = 0.23, 0.50
R1_CUT = 0.30

shap_cols = [c for c in shap.columns if c.startswith("shap_")]
features = [c[5:] for c in shap_cols]

merged = shap.merge(pred[["pair_id", "virus_name", "serum_name", "lineage",
                          "r1_value", "prob_match"]], on="pair_id", validate="one_to_one")
iran = merged[merged["lineage"] == "A/ASIA/Iran-05"].copy()
assert len(iran) == 10, len(iran)

PRETTY = {
    "pairwise_aa_distance": "Pairwise aa distance",
    "nglyc_count_diff": "|Δ sequon count|",
    "nglyc_shared": "Shared sequons",
    "nglyc_virus_only": "Virus-only sequons",
    "nglyc_serum_only": "Serum-only sequons",
}


def label_for(f: str) -> str:
    if f in PRETTY:
        return PRETTY[f]
    pos = int(f.split("_")[1])
    reg = region_lut.get(pos, fs.region_of(pos))
    return f"VP1 {pos}" if reg == "Other" else f"VP1 {pos} ({reg})"


fig = plt.figure(figsize=(fs.W_DOUBLE, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[1.0, 1.18],
                      wspace=0.42, hspace=0.46)
axA = fig.add_subplot(gs[0, 0])
axC = fig.add_subplot(gs[1, 0])
axB = fig.add_subplot(gs[:, 1])

# ------------------------------------------------------------------ Panel A
axA.axhspan(THR_OPT, 1.03, color="#F2F2F2", zorder=0)
axA.axvline(R1_CUT, color=fs.C_NEUTRAL, lw=0.7, ls=(0, (4, 2)), zorder=2)
axA.axhline(THR_OPT, color=fs.C_NOMATCH, lw=0.8, ls=(0, (1.5, 1.5)), zorder=2)
axA.axhline(THR_DEF, color=fs.C_NEUTRAL, lw=0.8, ls=(0, (4, 2)), zorder=2)

for cls, col, mk, lab in ((1, fs.C_MATCH, "o", "$r_1$ ≥ 0.3 (match)"),
                          (0, fs.C_NOMATCH, "s", "$r_1$ < 0.3 (no match)")):
    sub = iran[iran["r1_value"].ge(R1_CUT) == bool(cls)]
    axA.plot(sub["r1_value"], sub["prob_match"], marker=mk, ls="none", ms=5.0,
             mfc=col, mec="white", mew=0.6, zorder=4, label=lab)

disc = iran[(iran["prob_match"] >= THR_OPT) & (iran["r1_value"] < R1_CUT)]
for _, r in disc.iterrows():
    axA.annotate(r["virus_name"], (r["r1_value"], r["prob_match"]),
                 textcoords="offset points", xytext=(7, -1), fontsize=6.4,
                 va="center", ha="left", color="#3A3A3A", zorder=6)
axA.text(R1_CUT + 0.012, 0.035, "WOAH $r_1$ = 0.3", fontsize=6.4, color=fs.C_NEUTRAL)
axA.text(0.012, THR_OPT + 0.03, "Model threshold 0.23", fontsize=6.4, color=fs.C_NOMATCH)

axA.set_xlim(0, 0.62)
axA.set_ylim(-0.03, 1.03)
axA.set_xlabel("Measured $r_1$ value (2D-VNT)")
axA.set_ylabel("Predicted probability\nof antigenic match")
axA.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.26), ncol=2, handletextpad=0.3,
           columnspacing=1.2, borderaxespad=0.0)
fs.panel_label(axA, "A", dx=-0.21, dy=1.30)

# ------------------------------------------------------------------ Panel B
cs = case.sort_values("diff", ascending=True).reset_index(drop=True)
y = np.arange(len(cs))
h = 0.38
axB.barh(y + h / 2, cs["mean_shap_nomatch"], height=h, color=fs.C_NOMATCH,
         edgecolor="white", linewidth=0.3, zorder=3, label="True no match (n = 7)")
axB.barh(y - h / 2, cs["mean_shap_match"], height=h, color=fs.C_MATCH, hatch="///",
         edgecolor="white", linewidth=0.3, zorder=3, label="True match (n = 3)")
axB.axvline(0, color=fs.C_NEUTRAL, lw=0.7, zorder=4)
axB.set_yticks(y)
axB.set_yticklabels([label_for(f) for f in cs["feature"]])
axB.set_ylim(-0.8, len(cs) - 0.2)
axB.set_xlabel("Mean SHAP value in Iran-05 test pairs")
axB.tick_params(axis="y", length=0)
axB.xaxis.grid(True, zorder=0)
axB.set_axisbelow(True)
axB.spines["left"].set_visible(False)
axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, handlelength=1.4,
           handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)
fs.panel_label(axB, "B", dx=-0.37, dy=1.065)

# ------------------------------------------------------------------ Panel C
row = merged[merged["pair_id"] == "P0502"].iloc[0]
contrib = pd.Series({f: float(row[f"shap_{f}"]) for f in features})
contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
TOP = 9
top = contrib.iloc[:TOP]
rest = contrib.iloc[TOP:]
items = [(label_for(f), v) for f, v in top.items()]
items.append((f"{len(rest)} remaining features", float(rest.sum())))

running = BASE
lefts, widths, cols = [], [], []
for _, v in items:
    lefts.append(running if v >= 0 else running + v)
    widths.append(abs(v))
    cols.append(fs.C_MATCH if v >= 0 else fs.C_NOMATCH)
    running += v
final = running
assert abs(final - row["prob_match"]) < 1e-4, (final, row["prob_match"])

yy = np.arange(len(items))[::-1]
axC.barh(yy, widths, left=lefts, height=0.62, color=cols, edgecolor="white",
         linewidth=0.3, zorder=3)
# connecting step lines
run2 = BASE
for k, (_, v) in enumerate(items):
    if k < len(items) - 1:
        axC.plot([run2 + v, run2 + v], [yy[k] - 0.31, yy[k] - 0.69],
                 color=fs.C_LIGHT, lw=0.6, zorder=2)
    run2 += v
for k, (_, v) in enumerate(items):
    xa = lefts[k] + widths[k] if v >= 0 else lefts[k]
    axC.annotate(f"{v:+.3f}", (xa, yy[k]), textcoords="offset points",
                 xytext=(4 if v >= 0 else -4, 0), ha="left" if v >= 0 else "right",
                 va="center", fontsize=6.0, color="#3A3A3A", zorder=6)

axC.axvline(BASE, color=fs.C_NEUTRAL, lw=0.8, ls=(0, (4, 2)), zorder=4)
axC.axvline(final, color="black", lw=0.9, zorder=4)
axC.axvline(THR_OPT, color=fs.C_NOMATCH, lw=0.8, ls=(0, (1.5, 1.5)), zorder=4)
axC.set_yticks(yy)
axC.set_yticklabels([lab for lab, _ in items])
axC.set_ylim(-0.75, len(items) - 0.25)
axC.set_xlim(0.10, 0.78)
axC.set_xlabel("Predicted probability of antigenic match")
axC.tick_params(axis="y", length=0)
axC.xaxis.grid(True, zorder=0)
axC.set_axisbelow(True)
axC.spines["left"].set_visible(False)
axC.text(BASE - 0.006, len(items) - 0.18, f"Base value {BASE:.3f}", ha="right",
         va="bottom", fontsize=6.3, color=fs.C_NEUTRAL)
axC.text(final + 0.006, len(items) - 0.18, f"Prediction {final:.3f}", ha="left",
         va="bottom", fontsize=6.3, color="black")
axC.text(THR_OPT, -0.70, "0.23", ha="center", va="bottom", fontsize=6.3,
         color=fs.C_NOMATCH)
fs.panel_label(axC, "C", dx=-0.37, dy=1.18)

written = fs.save(fig, "Fig5")
print("\n".join(written))
print("discordant pairs:", list(disc["virus_name"]))
print("waterfall final=%.4f predicted=%.4f" % (final, row["prob_match"]))
print("iran05 match/no-match:", int((iran['r1_value'] >= R1_CUT).sum()),
      int((iran['r1_value'] < R1_CUT).sum()))
