"""Fig. 1 — Boruta feature selection: which VP1 descriptors discriminate
antigenic match from no-match.

Panel A : the 18 confirmed VP1 site features, frequency of an amino-acid
          difference among no-match vs match pairs.
Panel B : the four confirmed N-glycosylation features, mean per class.
Panel C : pairwise amino-acid distance distribution per class.
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

bor = pd.read_csv(os.path.join(UP, "boruta_feature_selection.csv"), encoding="utf-8-sig")
fm = pd.read_csv(os.path.join(UP, "feature_matrix_selected.csv"), encoding="utf-8-sig")
cmp_tab = pd.read_csv("/home/claude/build/serotype_comparison_table_v2.csv", encoding="utf-8-sig")
region_lut = dict(zip(cmp_tab["position"], cmp_tab["region"]))

conf = bor[bor["boruta_decision"] == "Confirmed"].copy()
sites = conf[conf["type"] == "site"].copy()
sites["position"] = sites["feature"].str.extract(r"site_(\d+)_diff").astype(int)
sites["region"] = sites["position"].map(lambda p: region_lut.get(p, fs.region_of(p)))
sites = sites.sort_values("diff_by_class", ascending=True)

n_match = int((fm["r1_binary"] == 1).sum())
n_nomatch = int((fm["r1_binary"] == 0).sum())

fig = plt.figure(figsize=(fs.W_ONEHALF, 5.95))
outer = fig.add_gridspec(2, 1, height_ratios=[0.50, 1.92], hspace=0.40)
axA = fig.add_subplot(outer[0])
inner = outer[1].subgridspec(2, 2, width_ratios=[1.0, 0.82],
                             height_ratios=[1.0, 0.92], wspace=0.52, hspace=0.46)
axB = fig.add_subplot(inner[:, 0])
axC = fig.add_subplot(inner[0, 1])
axD = fig.add_subplot(inner[1, 1])

# ------------------------------------------------------------------ Panel A
# Boruta outcome across all 96 candidate features. boruta_feature_selection.csv
# stores the final ranking and decision only, not the per-iteration importance
# distributions, so the selection profile is shown rather than a box plot of
# shadow-vs-real importances.
rank_df = bor.sort_values(["boruta_ranking", "feature"]).reset_index(drop=True)
assert len(rank_df) == 96, len(rank_df)
n_conf = int((rank_df["boruta_decision"] == "Confirmed").sum())
n_rej = int((rank_df["boruta_decision"] == "Rejected").sum())
xi = np.arange(1, len(rank_df) + 1)
is_conf = (rank_df["boruta_decision"] == "Confirmed").to_numpy()

axA.scatter(xi[is_conf], rank_df.loc[is_conf, "boruta_ranking"], s=13,
            facecolor=fs.C_ACCENT, edgecolor="white", linewidth=0.3, zorder=4,
            label=f"Confirmed (n = {n_conf})")
axA.scatter(xi[~is_conf], rank_df.loc[~is_conf, "boruta_ranking"], s=11,
            facecolor=fs.C_LIGHT, edgecolor="white", linewidth=0.3, zorder=3,
            label=f"Rejected (n = {n_rej})")
axA.axhline(2.5, color=fs.C_NEUTRAL, lw=0.7, ls=(0, (4, 2)), zorder=2)

# the one feature resolved from Tentative to Confirmed
tent = rank_df[rank_df["tentative_p_value"].notna()]
for _, r in tent.iterrows():
    k = int(xi[rank_df["feature"] == r["feature"]][0])
    axA.annotate(f"{r['feature'].replace('_diff', '')} (resolved from Tentative,\n"
                 f"p = {float(r['tentative_p_value']):.3f})",
                 (k, r["boruta_ranking"]), textcoords="offset points",
                 xytext=(13, 19), fontsize=6.2, color="#3A3A3A", zorder=6,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.C_NEUTRAL,
                                 shrinkA=0, shrinkB=2))
axA.set_xlabel("Candidate feature, ordered by Boruta ranking")
axA.set_ylabel("Boruta ranking")
axA.set_xlim(-1, len(rank_df) + 2)
axA.set_ylim(0, rank_df["boruta_ranking"].max() * 1.12)
axA.yaxis.grid(True, zorder=0)
axA.set_axisbelow(True)
axA.legend(loc="upper left", bbox_to_anchor=(0.01, 1.02), handletextpad=0.3,
           labelspacing=0.25, borderaxespad=0.0)
fs.panel_label(axA, "A", dx=-0.075, dy=1.20)

# ------------------------------------------------------------------ Panel B
y = np.arange(len(sites))
h = 0.38
axB.barh(y + h / 2, sites["mean_in_nomatch"], height=h, color=fs.C_NOMATCH,
         edgecolor="white", linewidth=0.3, label=f"No match, $r_1$ < 0.3 (n = {n_nomatch})",
         zorder=3)
axB.barh(y - h / 2, sites["mean_in_match"], height=h, color=fs.C_MATCH,
         edgecolor="white", linewidth=0.3, hatch="///",
         label=f"Match, $r_1$ ≥ 0.3 (n = {n_match})", zorder=3)
axB.set_yticks(y)
axB.set_yticklabels([f"{p}" for p in sites["position"]])
axB.set_ylim(-0.75, len(sites) - 0.25)
axB.set_xlim(0, 1.0)
axB.set_xlabel("Proportion of pairs with an amino-acid difference")
axB.set_ylabel("VP1 position", labelpad=14)
axB.xaxis.grid(True, zorder=0)
axB.set_axisbelow(True)
axB.tick_params(axis="y", length=0, pad=2)

# region colour strip left of the y labels
for i, reg in enumerate(sites["region"]):
    axB.add_patch(plt.Rectangle((-0.155, i - 0.42), 0.045, 0.84,
                                transform=axB.get_yaxis_transform(which="grid"),
                                clip_on=False, facecolor=fs.REGION_COLORS[reg],
                                edgecolor="none", zorder=4))
# class legend inside the axes, region key above it
cls_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=fs.C_MATCH, hatch="///",
                  edgecolor="white", linewidth=0.3),
    plt.Rectangle((0, 0), 1, 1, facecolor=fs.C_NOMATCH, edgecolor="white", linewidth=0.3),
]
leg_cls = axB.legend(cls_handles,
                     [f"Match, $r_1$ ≥ 0.3 (n = {n_match})",
                      f"No match, $r_1$ < 0.3 (n = {n_nomatch})"],
                     loc="lower right", handlelength=1.4, handletextpad=0.5,
                     borderaxespad=0.4, labelspacing=0.3)
axB.add_artist(leg_cls)

present = [r for r in fs.REGION_ORDER if r in set(sites["region"])]
reg_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fs.REGION_COLORS[r], edgecolor="none")
               for r in present]
axB.legend(reg_handles, present, loc="lower center", bbox_to_anchor=(0.45, 1.005),
           ncol=2, handlelength=1.1, handleheight=0.9, handletextpad=0.4,
           columnspacing=0.9, labelspacing=0.25)
fs.panel_label(axB, "B", dx=-0.26, dy=1.105)

# ------------------------------------------------------------------ Panel C
gly = conf[conf["type"] == "glycan"].copy()
gly_label = {
    "nglyc_count_diff": "|$\\Delta$ count|",
    "nglyc_shared": "Shared",
    "nglyc_virus_only": "Virus-only",
    "nglyc_serum_only": "Serum-only",
}
gly["label"] = gly["feature"].map(gly_label)
gly = gly.set_index("feature").loc[list(gly_label)].reset_index()
x = np.arange(len(gly))
w = 0.38
axC.bar(x - w / 2, gly["mean_in_match"], width=w, color=fs.C_MATCH, hatch="///",
        edgecolor="white", linewidth=0.3, zorder=3)
axC.bar(x + w / 2, gly["mean_in_nomatch"], width=w, color=fs.C_NOMATCH,
        edgecolor="white", linewidth=0.3, zorder=3)
axC.set_xticks(x)
axC.set_xticklabels(gly["label"], rotation=20, ha="right")
axC.set_ylabel("Mean number of\nN-glycosylation sequons")
axC.yaxis.grid(True, zorder=0)
axC.set_axisbelow(True)
axC.set_ylim(0, max(gly["mean_in_nomatch"].max(), gly["mean_in_match"].max()) * 1.22)
fs.panel_label(axC, "C", dx=-0.24, dy=1.14)

# ------------------------------------------------------------------ Panel D
d_match = fm.loc[fm["r1_binary"] == 1, "pairwise_aa_distance"].to_numpy(float)
d_nomatch = fm.loc[fm["r1_binary"] == 0, "pairwise_aa_distance"].to_numpy(float)
bp = axD.boxplot([d_match, d_nomatch], positions=[0, 1], widths=0.5,
                 showfliers=False, patch_artist=True, zorder=3)
for patch, col, hatch in zip(bp["boxes"], [fs.C_MATCH, fs.C_NOMATCH], ["///", ""]):
    patch.set_facecolor(col)
    patch.set_alpha(0.35)
    patch.set_edgecolor(col)
    patch.set_linewidth(0.8)
    patch.set_hatch(hatch)
for key in ("whiskers", "caps"):
    for ln in bp[key]:
        ln.set_color(fs.C_NEUTRAL)
        ln.set_linewidth(0.7)
for ln in bp["medians"]:
    ln.set_color("black")
    ln.set_linewidth(1.1)

rng = np.random.default_rng(42)
for pos, data, col in ((0, d_match, fs.C_MATCH), (1, d_nomatch, fs.C_NOMATCH)):
    jit = pos + rng.uniform(-0.17, 0.17, data.size)
    axD.scatter(jit, data, s=4.5, facecolor=col, edgecolor="none", alpha=0.55, zorder=4)
axD.set_xticks([0, 1])
axD.set_xticklabels([f"Match\n(n = {n_match})", f"No match\n(n = {n_nomatch})"])
axD.set_ylabel("Pairwise amino-acid distance\n(substitutions per site)")
axD.yaxis.grid(True, zorder=0)
axD.set_axisbelow(True)
axD.set_xlim(-0.6, 1.6)
fs.panel_label(axD, "D", dx=-0.26, dy=1.10)

written = fs.save(fig, "Fig1")
print("\n".join(written))
print(f"n_match={n_match} n_nomatch={n_nomatch} n_site_conf={len(sites)} n_gly={len(gly)}")
print("median distance match=%.4f nomatch=%.4f" % (np.median(d_match), np.median(d_nomatch)))
