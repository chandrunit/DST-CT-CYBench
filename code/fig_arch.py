import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
C1, C2, C3, C4 = "#dbe8f8", "#fde3d7", "#d5f1e6", "#fbecc4"
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")


def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - i * 0.27, l, ha="center", va="top", fontsize=7.8, color=MUTED)


def arrow(x0, y0, x1, y1, text=None, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=7.3, color=MUTED)


box(0.1, 2.35, 2.3, 2.1, "Inputs up to issue dekad τ", ["MODIS NDVI, FPAR (dekadal)", "AgERA5 weather (8 variables)", "GLDAS soil moisture", "gaps kept as missing (mask B)"], C1)
box(0.1, 0.15, 2.3, 1.95, "Static context z", ["WISE30sec soil properties", "location, crop-area fraction", "crop calendar (sos, eos)", "yield history (past seasons)"], C1)
box(2.9, 1.2, 2.5, 2.4, "Conditional diffusion", ["ε_θ(X_s, s, c): transformer", "trained: 1000-step cosine", "sampled: DDIM, S = 50, η = 1", "M = 10 completions of", "dekads > τ + gap repair"], C2)
box(5.9, 1.2, 1.9, 2.4, "Assimilate +\nconstraints Π", ["", "observed values", "re-imposed (eq. 4)", "NDVI, FPAR bounds", "T_min ≤ T_mean ≤ T_max", "CWB = P − ET₀"], C4)
box(8.3, 1.2, 1.9, 2.4, "Digital twin", ["modality attention", "(veg, met, soil; eq. 6)", "GRU state s_t (eq. 5)", "terminal state s_K", "per scenario m"], C3)
box(10.5, 1.2, 1.4, 2.4, "Forecast", ["N(μ, σ²) head", "(eq. 7)", "mixture over m", "(eq. 8)", "isotonic", "recalibration", "(eq. 9)"], C3)
arrow(2.4, 3.4, 2.9, 2.9, None)
arrow(2.4, 1.1, 2.9, 1.8, None)
arrow(5.4, 2.4, 5.9, 2.4)
arrow(7.8, 2.4, 8.3, 2.4)
arrow(10.2, 2.4, 10.5, 2.4)
ax.annotate("", xy=(9.25, 1.2), xytext=(2.4, 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, connectionstyle="arc3,rad=0.12"))
ax.text(6.1, 0.98, "static context initializes the twin state s₀ = e(z)", fontsize=7.3, color=MUTED, ha="center")
ax.text(11.2, 0.75, "yield distribution\nand 90% interval", ha="center", fontsize=7.8, color=INK)
plt.tight_layout(); plt.savefig("figs/fig_architecture.png", dpi=600)
