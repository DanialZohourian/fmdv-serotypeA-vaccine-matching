"""Fig. 3 — SHAP summary (beeswarm) for the 23 Boruta-selected features
across the 38 held-out test pairs.

Each point is one virus/antiserum pair. Horizontal position is the SHAP
value (change in predicted match probability); colour encodes the feature
value for that pair.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

UP = "/mnt/user-data/uploads"
fs.apply_style()

shap = pd.read_csv(os.path.join(UP, "shap_test_values.csv"), encoding="utf-8-sig")
feat = pd.read_csv(os.path.join(UP, "feature_matrix_selected.csv"), encoding="utf-8-sig")
glob = pd.read_csv(os.path.join(UP, "shap_global_importance.csv"), encoding="utf-8-sig")
cmp_tab = pd.read_csv("/home/claude/build/serotype_comparison_table_v2.csv", encoding="utf-8-sig")
region_lut = dict(zip(cmp_tab["position"], cmp_tab["region"]))

features = [c[len("shap_"):] for c in shap.columns if c.startswith("shap_")]
merged = shap.merge(feat[["pair_id"] + features], on="pair_id", how="left",
                    validate="one_to_one")
assert merged[features].notna().all().all(), "missing feature values after merge"
assert len(merged) == 38, len(merged)

# order rows by global mean |SHAP| (computed on all 190 pairs)
order = (glob.set_index("feature").loc[features, "mean_abs_shap"]
         .sort_values(ascending=True).index.tolist())

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


fig, ax = plt.subplots(figsize=(fs.W_ONEHALF, 5.1))
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
rng = np.random.default_rng(11)

for i, f in enumerate(order):
    sv = merged[f"shap_{f}"].to_numpy(float)
    val = merged[f].to_numpy(float)
    lo, hi = np.nanmin(val), np.nanmax(val)
    cval = np.full_like(val, 0.5) if hi == lo else (val - lo) / (hi - lo)
    off = fs.beeswarm_offsets(sv, max_offset=0.33, n_bins=26)
    off = off + rng.uniform(-0.015, 0.015, off.size)
    ax.scatter(sv, i + off, c=cval, cmap=fs.SHAP_CMAP, norm=norm, s=12,
               edgecolor="white", linewidth=0.25, zorder=3)

ax.axvline(0.0, color=fs.C_NEUTRAL, lw=0.7, zorder=2)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([label_for(f) for f in order])
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel("SHAP value (change in predicted probability of antigenic match)")
ax.tick_params(axis="y", length=0)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=fs.SHAP_CMAP), ax=ax,
                  pad=0.015, fraction=0.030, aspect=34)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.set_label("Feature value", labelpad=-9)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(length=0)

written = fs.save(fig, "Fig3")
print("\n".join(written))
print("top of plot:", order[-1], "bottom:", order[0])
print("SHAP range: %.3f to %.3f" % (
    min(merged[f"shap_{f}"].min() for f in order),
    max(merged[f"shap_{f}"].max() for f in order)))
