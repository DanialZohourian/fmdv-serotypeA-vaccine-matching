"""Fig. 2 — Performance of the Boruta-23 random-forest classifier on the
held-out test set (n = 38 virus/antiserum pairs).

Panel A : ROC curve, both operating points marked.
Panel B : precision-recall curve against the no-skill prevalence line.
Panel C : predicted match probability by true class, both thresholds shown.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

UP = "/mnt/user-data/uploads"
fs.apply_style()

pred = pd.read_csv(os.path.join(UP, "rf_test_predictions.csv"), encoding="utf-8-sig")
y = pred["r1_binary"].to_numpy(int)
p = pred["prob_match"].to_numpy(float)

THR_OPT, THR_DEF = 0.23, 0.50

fpr, tpr, roc_thr = roc_curve(y, p)
auc_roc = roc_auc_score(y, p)
prec, rec, pr_thr = precision_recall_curve(y, p)
auc_pr = average_precision_score(y, p)
auc_pr_trap = auc(rec, prec)
prevalence = y.mean()

print(f"AUC-ROC={auc_roc:.4f}  AP={auc_pr:.4f}  AUC-PR(trapz)={auc_pr_trap:.4f}  prev={prevalence:.4f}")

# published values to cross-check against
assert abs(auc_roc - 0.8428571428571429) < 1e-9, auc_roc

fig, axes = plt.subplots(1, 3, figsize=(fs.W_DOUBLE, 2.45),
                         gridspec_kw={"wspace": 0.42})
axA, axB, axC = axes

# ------------------------------------------------------------------ Panel A
axA.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=fs.C_LIGHT, lw=0.8, zorder=1)
axA.step(fpr, tpr, where="post", color=fs.C_MATCH, lw=1.4, zorder=3)
axA.fill_between(fpr, tpr, step="post", color=fs.C_MATCH, alpha=0.10, zorder=2)

for thr, marker, col, lab in ((THR_DEF, "o", fs.C_NEUTRAL, "0.50 (default)"),
                              (THR_OPT, "s", fs.C_NOMATCH, "0.23 (F$_1$-optimised)")):
    yhat = (p >= thr).astype(int)
    tp_ = int(((yhat == 1) & (y == 1)).sum()); fn_ = int(((yhat == 0) & (y == 1)).sum())
    fp_ = int(((yhat == 1) & (y == 0)).sum()); tn_ = int(((yhat == 0) & (y == 0)).sum())
    sens, spec = tp_ / (tp_ + fn_), tn_ / (tn_ + fp_)
    axA.plot(1 - spec, sens, marker=marker, ms=4.6, mfc="white", mec=col, mew=1.2,
             ls="none", zorder=5, label=f"Threshold {lab}")
    print(f"thr={thr}: sens={sens:.3f} spec={spec:.3f} TP={tp_} FP={fp_} FN={fn_} TN={tn_}")

axA.set_xlabel("1 − specificity (false-positive rate)")
axA.set_ylabel("Sensitivity (true-positive rate)")
axA.set_xlim(-0.02, 1.02)
axA.set_ylim(-0.02, 1.02)
axA.set_xticks(np.arange(0, 1.01, 0.25))
axA.set_yticks(np.arange(0, 1.01, 0.25))
axA.text(0.97, 0.06, f"AUC-ROC = {auc_roc:.3f}", ha="right", va="bottom", fontsize=7.5)
axA.legend(loc="lower right", bbox_to_anchor=(1.02, 0.14), handletextpad=0.4,
           labelspacing=0.25, borderaxespad=0.2)
axA.set_aspect("equal", adjustable="box")
fs.panel_label(axA, "A", dx=-0.23, dy=1.10)

# ------------------------------------------------------------------ Panel B
axB.axhline(prevalence, ls=(0, (3, 3)), color=fs.C_LIGHT, lw=0.8, zorder=1)
axB.step(rec, prec, where="post", color=fs.C_ACCENT, lw=1.4, zorder=3)
axB.fill_between(rec, prec, step="post", color=fs.C_ACCENT, alpha=0.10, zorder=2)
for thr, marker, col in ((THR_DEF, "o", fs.C_NEUTRAL), (THR_OPT, "s", fs.C_NOMATCH)):
    yhat = (p >= thr).astype(int)
    tp_ = int(((yhat == 1) & (y == 1)).sum())
    fp_ = int(((yhat == 1) & (y == 0)).sum())
    fn_ = int(((yhat == 0) & (y == 1)).sum())
    axB.plot(tp_ / (tp_ + fn_), tp_ / (tp_ + fp_), marker=marker, ms=4.6, mfc="white",
             mec=col, mew=1.2, ls="none", zorder=5)
axB.text(0.03, prevalence + 0.035, f"No-skill = {prevalence:.2f}", fontsize=6.8,
         color=fs.C_NEUTRAL)
axB.set_xlabel("Recall (sensitivity)")
axB.set_ylabel("Precision (positive predictive value)")
axB.set_xlim(-0.02, 1.02)
axB.set_ylim(-0.02, 1.02)
axB.set_xticks(np.arange(0, 1.01, 0.25))
axB.set_yticks(np.arange(0, 1.01, 0.25))
axB.text(0.97, 0.06, f"AUC-PR = {auc_pr:.3f}", ha="right", va="bottom", fontsize=7.5)
axB.set_aspect("equal", adjustable="box")
fs.panel_label(axB, "B", dx=-0.23, dy=1.10)

# ------------------------------------------------------------------ Panel C
rng = np.random.default_rng(7)
groups = ((0, 1, fs.C_MATCH, "Match\n$r_1$ ≥ 0.3"),
          (1, 0, fs.C_NOMATCH, "No match\n$r_1$ < 0.3"))
for xpos, cls, col, _ in groups:
    vals = p[y == cls]
    off = fs.beeswarm_offsets(vals, max_offset=0.26, n_bins=14)
    axC.scatter(xpos + off, vals, s=16, facecolor=col, edgecolor="white",
                linewidth=0.35, alpha=0.9, zorder=4)
    axC.hlines(np.median(vals), xpos - 0.33, xpos + 0.33, color="black", lw=1.2, zorder=5)

axC.axhline(THR_DEF, color=fs.C_NEUTRAL, ls=(0, (4, 2)), lw=0.9, zorder=2)
axC.axhline(THR_OPT, color=fs.C_NOMATCH, ls=(0, (1.5, 1.5)), lw=0.9, zorder=2)
axC.text(1.58, THR_DEF + 0.015, "0.50", fontsize=6.8, color=fs.C_NEUTRAL, ha="right")
axC.text(1.58, THR_OPT + 0.015, "0.23", fontsize=6.8, color=fs.C_NOMATCH, ha="right")
axC.set_xticks([0, 1])
axC.set_xticklabels([g[3] for g in groups])
axC.set_ylabel("Predicted probability of antigenic match")
axC.set_ylim(-0.03, 1.03)
axC.set_xlim(-0.6, 1.6)
axC.yaxis.grid(True, zorder=0)
axC.set_axisbelow(True)
fs.panel_label(axC, "C", dx=-0.23, dy=1.10)

written = fs.save(fig, "Fig2")
print("\n".join(written))
