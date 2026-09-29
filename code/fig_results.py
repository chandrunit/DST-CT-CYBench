"""Result figures (print, 600 dpi). Palette: validated reference categorical order."""
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})
DS = [("wheat_IN", "(a) Wheat, India"), ("maize_IN", "(b) Maize, India"), ("maize_US", "(c) Maize, USA")]
ANCH = {"wheat_IN": "3-season mean", "maize_IN": "3-season mean", "maize_US": "Linear trend"}
FR_LAB = ["0", "25", "50", "75", "100"]


def lead_time():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (ds, title) in zip(axs, DS):
        S = pd.read_csv(f"results/summary_{ds}_temporal.csv")
        models = ["DST-CT (full)", "DST-CT (deterministic, no diffusion)", "LSTM", "Ridge", "Linear trend", "3-season mean"]
        labels = ["DST-CT", "DST-CT deterministic", "LSTM", "Ridge", "Linear trend", "3-season mean"]
        for i, (m, lab) in enumerate(zip(models, labels)):
            d = S[S.model == m].sort_values("frac")
            if d.empty:
                continue
            ls = "--" if m == "Linear trend" else (":" if m == "3-season mean" else "-")
            ax.plot(d.frac * 100, d.RMSE, ls, color=PAL[i], marker="o", ms=4, label=lab)
            ax.fill_between(d.frac * 100, d.RMSE_lo, d.RMSE_hi, color=PAL[i], alpha=0.07, lw=0) if m == "DST-CT (full)" else None
        ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("Issue time (% of season elapsed)"); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylabel("RMSE (t ha$^{-1}$)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("figs/fig_leadtime.png", dpi=600); plt.close()


def reliability(frac=0.5):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, (ds, title) in zip(axs, DS):
        R = pd.read_csv(f"results/reliability_{ds}_temporal.csv"); R = R[np.isclose(R.frac, frac)]
        ens = [m for m in R.model.unique() if m.startswith("Deep ensemble")]
        models = ["DST-CT (full)", "DST-CT (deterministic, no diffusion)", "MC-Dropout twin"] + ens + ["LSTM"]
        labels = ["DST-CT", "DST-CT deterministic", "MC dropout", "Deep ensemble", "LSTM"]
        ax.plot([20, 100], [20, 100], color=MUTED, lw=1, ls=":")
        for i, (m, lab) in enumerate(zip(models, labels)):
            d = R[R.model == m].sort_values("nominal")
            if d.empty:
                continue
            ax.plot(d.nominal, d["cov"], color=PAL[i], marker="o", ms=4, label=lab + " (recalibrated)")
            if m == "DST-CT (full)":
                ax.plot(d.nominal, d.cov_raw, color=PAL[i], marker="o", ms=4, ls="--", label="DST-CT (raw)")
        ax.set_xlim(20, 100); ax.set_ylim(20, 100); ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Nominal interval level (%)"); ax.set_ylabel("Observed coverage (%)"); ax.set_aspect("equal")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("figs/fig_reliability.png", dpi=600); plt.close()


def scatter():
    import re
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (ds, title) in zip(axs, DS):
        ys, ps = [], []
        for fn in sorted(glob.glob(f"preds/{ds}_temporal_*_s0.npz")):
            z = np.load(fn); ys.append(z["y_te"]); ps.append(z["DST-CT (full)|1.0|test|mu"])
        y, p = np.concatenate(ys), np.concatenate(ps)
        lim = (0, np.percentile(np.r_[y, p], 99.5) * 1.05)
        hb = ax.hexbin(y, p, gridsize=45, cmap="Blues", mincnt=1, extent=(*lim, *lim), linewidths=0)
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        rmse = np.sqrt(np.mean((p - y) ** 2)); bias = np.mean(p - y); rr = 100 * rmse / y.mean()
        ax.text(0.04, 0.95, f"RMSE {rmse:.2f} t ha$^{{-1}}$\nrRMSE {rr:.1f}%\nbias {bias:+.2f} t ha$^{{-1}}$\nn = {len(y):,}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Observed yield (t ha$^{-1}$)"); ax.set_ylabel("DST-CT forecast (t ha$^{-1}$)")
        cb = fig.colorbar(hb, ax=ax, shrink=0.8); cb.set_label("Region-seasons"); cb.outline.set_visible(False)
    plt.tight_layout(); plt.savefig("figs/fig_scatter.png", dpi=600); plt.close()


def deep_figs(tag_in="wheat_IN_temporal_2017_s0", tag_us="maize_US_temporal_2023_s0"):
    # attention and permutation importance by phase (sequential single hue)
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
    for k, (tag, nm) in enumerate(((tag_in, "Wheat, India"), (tag_us, "Maize, USA"))):
        try:
            d = json.load(open(f"results/deep_{tag}.json"))
        except FileNotFoundError:
            continue
        A = np.array([d["attn_by_phase"][p] for p in ("early", "mid", "late")]).T  # 3 modalities x 3 phases
        ax = axs[2 * k]
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if A[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["early", "mid", "late"]); ax.set_yticks(range(3), ["vegetation", "meteorology", "soil moisture"])
        ax.set_title(f"({'ac'[k]}) {nm}: attention α", loc="left", fontsize=9); ax.grid(False)
        P = np.array([[d["perm_importance"][f"{g}|{p}"] for p in ("early", "mid", "late")] for g in ("veg", "met", "soil")])
        ax = axs[2 * k + 1]
        vmax = max(P.max(), 1e-3)
        im = ax.imshow(np.clip(P, 0, None), cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{P[i, j]*1000:.0f}", ha="center", va="center", fontsize=8, color="white" if P[i, j] > 0.55 * vmax else INK)
        ax.set_xticks(range(3), ["early", "mid", "late"]); ax.set_yticks(range(3), ["", "", ""])
        ax.set_title(f"({'bd'[k]}) ΔRMSE (kg ha$^{{-1}}$)", loc="left", fontsize=9); ax.grid(False)
    plt.tight_layout(); plt.savefig("figs/fig_importance.png", dpi=600); plt.close()

    # case studies: NDVI scenario fan at 50% + forecast evolution
    d = json.load(open(f"results/deep_{tag_in}.json"))
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2))
    for r, cs in enumerate(d["cases"]):
        nd = cs["nd"]; t = np.arange(nd) + 1
        ax = axs[r, 0]
        sc = np.array(cs["scen_ndvi"])[:, :nd]
        o = int(round(nd * 0.5))
        lo, hi = np.percentile(sc, [5, 95], 0); q1, q3 = np.percentile(sc, [25, 75], 0)
        ax.fill_between(t, lo, hi, color=PAL[0], alpha=0.15, lw=0, label="DST-CT scenarios 5–95%")
        ax.fill_between(t, q1, q3, color=PAL[0], alpha=0.3, lw=0, label="DST-CT scenarios 25–75%")
        obs = np.array(cs["ndvi"])[:nd]; obs = np.where(obs < -1, np.nan, obs)
        ax.plot(t, np.array(cs["clim_ndvi"])[:nd], color=PAL[3], ls="--", lw=1.5, label="training climatology")
        ax.plot(t, obs, color=INK, marker="o", ms=3.5, lw=1.2, label="observed (revealed after issue)")
        ax.axvline(o + 0.5, color=MUTED, lw=1, ls=":"); ax.text(o + 0.7, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.9, "issue (50%)", fontsize=7.5, color=MUTED, va="top")
        ax.set_xlabel("Dekad since start of season"); ax.set_ylabel("NDVI")
        ax.set_title(f"({'ac'[r]}) {cs['adm']}: NDVI at 50% issue time", loc="left", fontsize=9)
        ax = axs[r, 1]
        fr = [0, 25, 50, 75, 100]
        mu = [cs["fc"][k]["mu"] for k in ("0.0", "0.25", "0.5", "0.75", "1.0")]; sd = [cs["fc"][k]["sd"] for k in ("0.0", "0.25", "0.5", "0.75", "1.0")]
        det = [cs["fc"][k]["det"] for k in ("0.0", "0.25", "0.5", "0.75", "1.0")]
        ax.errorbar(fr, mu, yerr=1.645 * np.array(sd), color=PAL[0], marker="o", ms=5, capsize=3, lw=1.8, label="DST-CT mean ± 1.645σ (raw)")
        ax.plot(fr, det, color=PAL[1], marker="s", ms=4, lw=1.5, label="DST-CT deterministic")
        ax.axhline(cs["y"], color=INK, lw=1.2, label="observed yield")
        ax.axhline(cs["trend"], color=PAL[3], lw=1.2, ls="--", label="anchor (3-season mean)")
        ax.set_xticks(fr); ax.set_xlabel("Issue time (% of season)"); ax.set_ylabel("Yield (t ha$^{-1}$)")
        ax.set_title(f"({'bd'[r]}) forecast evolution", loc="left", fontsize=9)
    h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("figs/fig_cases.png", dpi=600); plt.close()

    # gap repair example
    g = d["gap_example"]; nd = g["nd"]; t = np.arange(nd) + 1
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    truth = np.array(g["truth"])[:nd]; truth = np.where(truth < -1, np.nan, truth); hid = np.array(g["hidden"])[:nd]
    sc = np.array(g["scen"])[:, :nd]
    ax.fill_between(t, sc.min(0), sc.max(0), color=PAL[0], alpha=0.18, lw=0, label="diffusion samples (range)")
    ax.plot(t, np.array(g["diffusion"])[:nd], color=PAL[0], marker="o", ms=3, label="diffusion (mean)")
    ax.plot(t, np.array(g["locf"])[:nd], color=PAL[1], marker="s", ms=3, lw=1.5, label="last observation carried forward")
    ax.plot(t, truth, color=INK, lw=0, marker="o", ms=5, mfc="white", label="true NDVI")
    ax.plot(t[hid], truth[hid], color=INK, lw=0, marker="x", ms=7, label="hidden (artificial gap)")
    ax.set_xlabel("Dekad since start of season"); ax.set_ylabel("NDVI"); ax.legend(fontsize=7.5, loc="lower center", ncol=2)
    plt.tight_layout(); plt.savefig("figs/fig_gap.png", dpi=600); plt.close()


def steps_fig(tags):
    rows = []
    for tag in tags:
        try:
            d = json.load(open(f"results/deep_{tag}.json"))
        except FileNotFoundError:
            continue
        for r in d["steps"]:
            r = dict(r); r["tag"] = tag; rows.append(r)
    df = pd.DataFrame(rows)
    df["ds"] = df.tag.str.extract(r"^(\w+?_\w\w)_")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, (ds, lab) in enumerate((("wheat_IN", "Wheat, India"), ("maize_US", "Maize, USA"))):
        for j, f in enumerate((0.0, 0.5)):
            d = df[(df.ds == ds) & (df.frac == f)].groupby("steps").mean(numeric_only=True).reset_index()
            if d.empty:
                continue
            axs[0].plot(d.steps, d.crps, color=PAL[2 * i + j], marker="o", ms=4, label=f"{lab}, issue {int(f*100)}%")
            axs[1].plot(d.steps, d.picp90_cal, color=PAL[2 * i + j], marker="o", ms=4)
            axs[2].plot(d.steps, d.sec_per_1000_regions, color=PAL[2 * i + j], marker="o", ms=4)
    for ax, yl, t in zip(axs, ["CRPS (t ha$^{-1}$)", "PICP of 90% interval (%)", "Latency (s per 1000 regions)"], ["(a) accuracy", "(b) calibration", "(c) CPU latency, M = 10"]):
        ax.set_xscale("log"); ax.set_xlabel("DDIM sampling steps S"); ax.set_ylabel(yl); ax.set_title(t, loc="left", fontsize=9.5)
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
    axs[2].set_yscale("log")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("figs/fig_steps.png", dpi=600); plt.close()


if __name__ == "__main__":
    import sys
    for w in sys.argv[1].split(","):
        if w == "lead": lead_time()
        if w == "rel": reliability()
        if w == "scatter": scatter()
        if w == "deep": deep_figs()
        if w == "steps": steps_fig(["wheat_IN_temporal_2017_s0", "wheat_IN_temporal_2017_s1", "wheat_IN_temporal_2017_s2", "maize_US_temporal_2023_s0"])
