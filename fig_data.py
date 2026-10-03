"""Figure: study regions (centroids coloured by mean yield) + dataset summary table (CSV)."""
import numpy as np, pandas as pd, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import load, DYN
from matplotlib.colors import LinearSegmentedColormap
BL = LinearSegmentedColormap.from_list("b", plt.cm.Blues(np.linspace(0.3, 1, 256)))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
rows = []
titles = {"wheat_IN": "(a) Wheat, India (districts)", "maize_IN": "(b) Maize, India (districts)", "maize_US": "(c) Maize, USA (counties)"}
for ax, ds in zip(axs, ["wheat_IN", "maize_IN", "maize_US"]):
    D = load(ds)
    df = pd.DataFrame(dict(adm=D["adm"], y=D["y"], lat=D["S"][:, 3], lon=D["S"][:, 4]))
    g = df.groupby("adm").agg(y=("y", "mean"), lat=("lat", "first"), lon=("lon", "first"), n=("y", "size"))
    sc = ax.scatter(g.lon, g.lat, c=g.y, cmap=BL, s=7 if ds != "maize_US" else 3, vmin=np.percentile(g.y, 2), vmax=np.percentile(g.y, 98),
                    edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Mean yield (t ha$^{-1}$)"); cb.outline.set_visible(False)
    ax.set_title(titles[ds], loc="left", fontsize=10)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)"); ax.set_aspect("equal", adjustable="datalim")
    X = D["X"]; ins = np.arange(X.shape[1])[None] < D["nd"][:, None]
    miss_ndvi = np.isnan(X[:, :, 0])[ins].mean() * 100
    miss_sm = np.isnan(X[:, :, 10])[ins].mean() * 100
    rows.append(dict(dataset=titles[ds][4:], regions=g.shape[0], seasons=len(D["y"]), years=f"{D['year'].min()}–{D['year'].max()}",
                     seasons_per_region_median=int(g.n.median()), yield_mean=round(float(D["y"].mean()), 2), yield_sd=round(float(D["y"].std()), 2),
                     yield_min=round(float(D["y"].min()), 2), yield_max=round(float(D["y"].max()), 2),
                     season_dekads_median=int(np.median(D["nd"])), ndvi_missing_pct=round(miss_ndvi, 1), soilmoist_missing_pct=round(miss_sm, 1),
                     lat_range=f"{g.lat.min():.1f}–{g.lat.max():.1f}", lon_range=f"{g.lon.min():.1f}–{g.lon.max():.1f}"))
plt.tight_layout()
plt.savefig("figs/fig_study_area.png", dpi=600)
pd.DataFrame(rows).to_csv("results/dataset_table.csv", index=False)
print(pd.DataFrame(rows).T)
