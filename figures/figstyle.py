"""Shared plotting style for Vaccine (Elsevier) submission figures.

Elsevier artwork requirements applied here
------------------------------------------
* Sans-serif typeface throughout (Arial; Liberation Sans is metrically identical).
* Minimum effective font size 7 pt at final printed size.
* Column widths: 90 mm (single), 140 mm (1.5 column), 190 mm (double).
* No figure number, article title or author names inside the artwork.
* Fonts embedded as TrueType (fonttype 42) in all vector output.
* Colour defined in RGB; all series also separable in greyscale by
  hatching / marker shape / position.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- column widths
MM = 1.0 / 25.4
W_SINGLE = 90 * MM     # 3.543 in
W_ONEHALF = 140 * MM   # 5.512 in
W_DOUBLE = 190 * MM    # 7.480 in

# ---------------------------------------------------------------------- palette
C_MATCH = "#2166AC"      # r1 >= 0.3  (antigenic match)
C_NOMATCH = "#B2182B"    # r1 <  0.3  (no match)
C_NEUTRAL = "#4D4D4D"
C_LIGHT = "#BDBDBD"
C_ACCENT = "#1A7F6B"

# VP1 structural regions
REGION_COLORS = {
    "BC loop": "#A6CEE3",
    "EF loop": "#B2DF8A",
    "G-H loop": "#FDBF6F",
    "C-terminus": "#CAB2D6",
    "Other": "#E8E8E8",
}
REGION_ORDER = ["BC loop", "EF loop", "G-H loop", "C-terminus", "Other"]

# SHAP diverging colour map (low feature value -> blue, high -> red)
SHAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "shap_bluered", ["#2166AC", "#8FA7C4", "#D6BFC2", "#D6604D", "#B2182B"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "grid.color": "#D9D9D9",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # embed TrueType fonts in vector output (Elsevier requirement)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.045) -> None:
    """Bold upper-case panel letter, Elsevier house style."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def save(fig, stem: str, outdir: str = "/home/claude/figures/out") -> list[str]:
    """Write vector PDF + EPS (preferred by Elsevier) and 600 dpi TIFF/PNG."""
    import os
    os.makedirs(outdir, exist_ok=True)
    written = []
    base = os.path.join(outdir, stem)
    # Vector PDF is kept rather than EPS: the PostScript backend flattens the
    # transparency used in the point overlays. Elsevier accepts TIFF, EPS, PDF
    # and MS Office artwork, so PDF (vector) + 900 dpi TIFF covers every route.
    # 900 dpi leaves >500 dpi (Elsevier's combination-artwork minimum) even
    # after the typesetter scales the trimmed artwork up to full column width.
    fig.savefig(base + ".pdf", facecolor="white")
    written.append(base + ".pdf")

    tmp_png = base + ".png"
    fig.savefig(tmp_png, dpi=900, facecolor="white")
    written.append(tmp_png)

    # flatten the alpha channel onto white: Elsevier wants plain RGB rasters
    from PIL import Image
    with Image.open(tmp_png) as im:
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
        flat.save(base + ".tif", compression="tiff_lzw", dpi=(900, 900))
        flat.save(tmp_png, dpi=(900, 900))
    written.append(base + ".tif")

    plt.close(fig)
    return written


def beeswarm_offsets(values: np.ndarray, max_offset: float = 0.34,
                     n_bins: int = 36) -> np.ndarray:
    """Symmetric density-proportional vertical offsets for a 1-D strip."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    offsets = np.zeros_like(values)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or hi == lo:
        return offsets
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = np.where(idx == b)[0]
        k = sel.size
        if k == 0:
            continue
        if k == 1:
            offsets[sel] = 0.0
            continue
        # interleave outwards from the centre line
        order = sel[np.argsort(values[sel])]
        spread = min(max_offset, 0.05 * k)
        pos = np.linspace(-spread, spread, k)
        # alternate so the strip fills from the middle out
        mid = np.argsort(np.abs(pos))
        offsets[order] = pos[mid][np.argsort(np.argsort(np.arange(k)))]
    return offsets


def region_of(position: int) -> str:
    """VP1 structural region for a (trimmed-alignment == A22/IRQ/24/64) position."""
    if 38 <= position <= 60:
        return "BC loop"
    if 80 <= position <= 105:
        return "EF loop"
    if 132 <= position <= 161:
        return "G-H loop"
    if position >= 195:
        return "C-terminus"
    return "Other"
