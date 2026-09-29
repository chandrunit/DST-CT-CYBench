import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})
T = pd.read_csv("results2/scenario_skill.csv"); T = T[T.split == "test"]
DS = [("wheat_IN", "Wheat, India"), ("maize_IN", "Maize, India"), ("maize_US", "Maize, USA")]
M = [("Diffusion", "Conditional diffusion", "-", PAL[0]), ("Analog (kNN)", "Analog, nearest seasons", "-", PAL[2]),
     ("Analog (random)", "Analog, random seasons", "--", PAL[3]), ("Climatology", "Climatology", ":", PAL[5])]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
for j, (ds, name) in enumerate(DS):
    for row, (col, lab) in enumerate((("CRPS", "All 12 predictors"), ("rsm", "Root-zone soil moisture"))):
        ax = axs[row, j]
        for m, ml, ls, c in M:
            d = T[(T.ds == ds) & (T.method == m)].sort_values("frac")
            ax.plot(d.frac * 100, d[col], ls, color=c, marker="o", ms=4, label=ml)
        ax.set_xticks([0, 25, 50, 75]); ax.set_xlabel("Issue time (% of season elapsed)")
        ax.set_ylabel("CRPS (standardized units)")
        ax.set_title(f"({'abc'[j] if row == 0 else 'def'[j]}) {name}: {lab.lower()}", loc="left", fontsize=9.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig("figs/fig_scenario_skill.png", dpi=600); plt.close()
