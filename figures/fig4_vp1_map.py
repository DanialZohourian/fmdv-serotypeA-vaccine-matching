"""Fig. 4 — Where the predictive signal sits on VP1.

Panel A : linear map of VP1 (213 aligned positions, numbered as in the
          A22/IRQ/24/64 reference and identical to PDB 4GH4 chain A).
          Stems give the global mean |SHAP| of each confirmed position;
          open triangles mark the positions reported as important for
          serotype O by Makau et al. (2025).
Panel B : signed mean effect of an amino-acid difference at each confirmed
          position on the predicted probability of antigenic match.
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

glob = pd.read_csv(os.path.join(UP, "shap_global_importance.csv"), encoding="utf-8-sig")
cmp_tab = pd.read_csv("/home/claude/build/serotype_comparison_table_v2.csv", encoding="utf-8-sig")
mapping = pd.read_csv(os.path.join(UP, "alignment_to_pdb_mapping.csv"), encoding="utf-8-sig")
region_lut = dict(zip(cmp_tab["position"], cmp_tab["region"]))

ALIGN_LEN = int(mapping["alignment_col"].max())

# Signed mean effect of a difference at each position, taken verbatim from
# step-8 output shap_interpretation_report.txt (computed over all 190 pairs).
SIGNED_EFFECT = {
    149: -0.1741, 173: -0.1053, 197: -0.0938, 193: -0.0858, 171: -0.0769,
    101: -0.0571, 167: -0.0547, 83: -0.0453, 164: -0.0391, 143: -0.0339,
    142: -0.0299, 3: -0.0220, 140: -0.0219,
    96: 0.0650, 46: 0.0520, 95: 0.0431, 93: 0.0304, 40: 0.0194,
}

sites = glob[glob["type"] == "site"].copy()
sites["position"] = sites["feature"].str.extract(r"site_(\d+)_diff").astype(int)
sites["region"] = sites["position"].map(lambda p: region_lut.get(p, fs.region_of(p)))
assert set(sites["position"]) == set(SIGNED_EFFECT), (
    set(sites["position"]) ^ set(SIGNED_EFFECT))

# Every position plotted here is assigned a structural region in
# serotype_comparison_table.csv (Table 4), so region membership is taken from
# that file rather than from residue ranges asserted in the figure.
assert set(sites["position"]) <= set(region_lut), "position missing a region assignment"

fig = plt.figure(figsize=(fs.W_DOUBLE, 4.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.52)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ------------------------------------------------------------------ Panel A
ymax = sites["mean_abs_shap"].max() * 1.30
axA.hlines(0, 0.5, ALIGN_LEN + 0.5, color=fs.C_NEUTRAL, lw=0.7, zorder=3)
for _, r in sites.iterrows():
    axA.vlines(r["position"], 0, r["mean_abs_shap"], color=fs.C_NEUTRAL, lw=0.8, zorder=4)
    axA.plot(r["position"], r["mean_abs_shap"], "o", ms=4.2,
             mfc=fs.REGION_COLORS[r["region"]], mec=fs.C_NEUTRAL, mew=0.7, zorder=5)

# label the six strongest positions (nudged apart where stems are adjacent)
LABEL_NUDGE = {171: (-7.5, 0.5), 173: (5.5, 5.0), 193: (-5.5, 2.0), 197: (5.0, 4.5)}
for _, r in sites.nlargest(6, "mean_abs_shap").iterrows():
    dx, dy = LABEL_NUDGE.get(int(r["position"]), (0.0, 4.5))
    axA.annotate(int(r["position"]), (r["position"], r["mean_abs_shap"]),
                 textcoords="offset points", xytext=(dx, dy), ha="center",
                 fontsize=6.6, zorder=6)

o_pos = cmp_tab[cmp_tab["important_in_O"] == "yes"]
axA.plot(o_pos["position"], np.full(len(o_pos), -ymax * 0.075), marker="v", ls="none",
         ms=4.6, mfc="white", mec="black", mew=0.8, zorder=6,
         label="Important for serotype O [Makau et al., 2025]")
o_sorted = o_pos.sort_values("position").reset_index(drop=True)
for k, r in o_sorted.iterrows():
    prev = o_sorted.loc[k - 1, "position"] if k > 0 else -99
    dy = -16.0 if (r["position"] - prev) < 8 else -8.5
    axA.annotate(int(r["position"]), (r["position"], -ymax * 0.075),
                 textcoords="offset points", xytext=(0, dy), ha="center",
                 fontsize=6.3, color="#3A3A3A", zorder=6)

axA.set_xlim(0.5, ALIGN_LEN + 0.5)
axA.set_ylim(-ymax * 0.20, ymax)
axA.set_xlabel("VP1 amino-acid position (A22/IRQ/24/64 numbering)")
axA.set_ylabel("Mean |SHAP|")
axA.set_xticks(list(range(0, ALIGN_LEN + 1, 20)) + [ALIGN_LEN])
axA.set_yticks(np.arange(0, 0.101, 0.025))
axA.spines["bottom"].set_visible(False)
reg_handles = [plt.Line2D([], [], marker="o", ls="none", ms=4.2,
                          mfc=fs.REGION_COLORS[r], mec=fs.C_NEUTRAL, mew=0.7)
               for r in fs.REGION_ORDER]
tri_handle = plt.Line2D([], [], marker="v", ls="none", ms=4.6, mfc="white",
                        mec="black", mew=0.8)
axA.legend(reg_handles + [tri_handle],
           fs.REGION_ORDER + ["Most informative for serotype O"],
           loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=6,
           handletextpad=0.3, columnspacing=1.1, borderaxespad=0.0)
fs.panel_label(axA, "A", dx=-0.055, dy=1.20)

# ------------------------------------------------------------------ Panel B
eff = (pd.DataFrame({"position": list(SIGNED_EFFECT), "effect": list(SIGNED_EFFECT.values())})
       .sort_values("effect"))
eff["region"] = eff["position"].map(lambda p: region_lut.get(p, fs.region_of(p)))
x = np.arange(len(eff))
colors = [fs.C_NOMATCH if v < 0 else fs.C_MATCH for v in eff["effect"]]
axB.bar(x, eff["effect"], width=0.66, color=colors, edgecolor="white", linewidth=0.3,
        zorder=3)
axB.axhline(0, color=fs.C_NEUTRAL, lw=0.7, zorder=4)
axB.set_xticks(x)
axB.set_xticklabels(eff["position"])
axB.set_xlim(-0.8, len(eff) - 0.2)
axB.set_xlabel("VP1 amino-acid position", labelpad=12)
axB.set_ylabel("Mean change in predicted\nprobability of match")
axB.yaxis.grid(True, zorder=0)
axB.set_axisbelow(True)

# region colour chips under each bar label
for i, reg in enumerate(eff["region"]):
    axB.add_patch(plt.Rectangle((i - 0.33, -0.085), 0.66, 0.030,
                                transform=axB.get_xaxis_transform(which="grid"),
                                clip_on=False, facecolor=fs.REGION_COLORS[reg],
                                edgecolor="none", zorder=6))

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fs.REGION_COLORS[r], edgecolor="none")
           for r in fs.REGION_ORDER]
axB.legend(handles, fs.REGION_ORDER, loc="lower center", bbox_to_anchor=(0.5, 1.01),
           ncol=5, handlelength=1.1, handleheight=0.9, handletextpad=0.4,
           columnspacing=1.2)
fs.panel_label(axB, "B", dx=-0.055, dy=1.24)

written = fs.save(fig, "Fig4")
print("\n".join(written))
print("alignment length:", ALIGN_LEN)
print("n negative:", (eff["effect"] < 0).sum(), "n positive:", (eff["effect"] > 0).sum())
print(sites.nlargest(6, "mean_abs_shap")[["position", "region", "mean_abs_shap"]].to_string(index=False))
