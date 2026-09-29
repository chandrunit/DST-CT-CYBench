"""Result tables (pandoc pipe tables) generated directly from results/*.csv|json, so every number is traceable."""
import json, glob, numpy as np, pandas as pd
from tables_static import pipe

DSN = {"wheat_IN": "Wheat, India", "maize_IN": "Maize, India", "maize_US": "Maize, USA"}
ORDER = ["Persistence", "Climatology", "Linear trend", "3-season mean", "VI regression", "Ridge", "LightGBM", "LSTM", "Transformer",
         "MC-Dropout twin", "DEEP", "DST-CT (deterministic, no diffusion)", "DST-CT (full)"]
NICE = {"MC-Dropout twin": "MC-dropout twin", "DST-CT (deterministic, no diffusion)": "DST-CT, deterministic (no diffusion)",
        "DST-CT (full)": "**DST-CT (full)**", "Linear trend": "Linear trend", "3-season mean": "3-season mean"}
PROB = ["LSTM", "Transformer", "MC-Dropout twin", "DEEP", "DST-CT (deterministic, no diffusion)", "DST-CT (full)"]


def S(ds, mode="temporal"):
    return pd.read_csv(f"results/summary_{ds}_{mode}.csv")


def mname(S_, m):
    if m == "DEEP":
        c = [x for x in S_.model.unique() if x.startswith("Deep ensemble")]
        return c[0] if c else None
    return m


def fmt_rmse(r, ci=True):
    s = f"{r.RMSE:.3f}"
    if r.n_seeds > 1 and r.RMSE_sd > 5e-4:
        s += f" ± {r.RMSE_sd:.3f}"
    if ci:
        s += f" [{r.RMSE_lo:.2f}, {r.RMSE_hi:.2f}]"
    return s


def t_main(ds, num):
    s = S(ds); rows = []
    best = {}
    for f in (0.0, 0.5, 1.0):
        best[f] = s[np.isclose(s.frac, f)].RMSE.min()
    for m in ORDER:
        mm = mname(s, m)
        if mm is None:
            continue
        row = {"Model": NICE.get(m, "Deep ensemble" if m == "DEEP" else m)}
        for f in (0.0, 0.5, 1.0):
            r = s[(s.model == mm) & np.isclose(s.frac, f)]
            if r.empty:
                row[f"RMSE, φ = {f:g}"] = "–"; continue
            r = r.iloc[0]; txt = fmt_rmse(r)
            if np.isclose(r.RMSE, best[f]):
                txt = f"**{txt}**"
            row[f"RMSE, φ = {f:g}"] = txt
        r = s[(s.model == mm) & np.isclose(s.frac, 1.0)].iloc[0]
        row["MAE, φ = 1"] = f"{r.MAE:.3f}"; row["*R*^2^, φ = 1"] = f"{r.R2:.3f}"; row["nRMSE (%), φ = 1"] = f"{r.nRMSE:.1f}"
        rows.append(row)
    n = s.n_test.iloc[0]
    return pipe(pd.DataFrame(rows), f"Table {num}. {DSN[ds]}: point accuracy on the pooled test seasons ({n:,} region-seasons) at issue fractions φ = 0 (sowing), 0.5 and 1 (end of season). RMSE in t ha^−1^: mean ± s.d. over seeds [95% region-cluster bootstrap CI]. Bold: lowest RMSE in the column. Deterministic references (persistence, climatology, trend, 3-season mean) have no seed variance.")


def t_paired(num):
    rows = []
    for ds in DSN:
        p = pd.read_csv(f"results/paired_{ds}_temporal.csv")
        for b in ["Linear trend", "3-season mean", "Ridge", "LightGBM", "LSTM", "DST-CT (deterministic, no diffusion)"]:
            row = {"Dataset": DSN[ds], "Baseline": {"DST-CT (deterministic, no diffusion)": "DST-CT, deterministic"}.get(b, b)}
            for f in (0.0, 0.5, 1.0):
                r = p[(p.baseline == b) & np.isclose(p.frac, f)]
                if r.empty:
                    row[f"φ = {f:g}"] = "–"; continue
                r = r.iloc[0]
                sig = "*" if (r.d_hi < 0 or r.d_lo > 0) and r.wilcoxon_p < 0.05 else ""
                row[f"φ = {f:g}"] = f"{r.d_rmse:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]{sig}"
            rows.append(row)
    return pipe(pd.DataFrame(rows), f"Table {num}. Paired comparison of DST-CT (full) with selected baselines: RMSE difference (DST-CT − baseline, t ha^−1^; negative = DST-CT better) of seed-averaged forecasts with 95% region-cluster bootstrap CI. \\* the CI excludes zero and a two-sided Wilcoxon signed-rank test on region-mean squared errors gives *p* < 0.05. Comparisons with all other baselines, with *p*-values, are in the supplementary file paired_tests.csv.")


def t_prob(num, f=0.5):
    rows = []
    for ds in DSN:
        s = S(ds)
        for m in PROB:
            mm = mname(s, m)
            r = s[(s.model == mm) & np.isclose(s.frac, f)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"Dataset": DSN[ds], "Model": NICE.get(m, "Deep ensemble" if m == "DEEP" else m).replace("**", ""),
                         "NLL": f"{r.NLL:.3f}", "CRPS (t ha^−1^)": f"{r.CRPS:.3f}", "PICP90 raw (%)": f"{r.PICP90_raw:.1f}",
                         "PICP90 recalibrated (%)": f"{r.PICP90:.1f}", "PIW90 (t ha^−1^)": f"{r.PIW90:.2f}"})
    return pipe(pd.DataFrame(rows), f"Table {num}. Probabilistic skill at mid-season (φ = {f:g}), mean over seeds. NLL: Gaussian negative log-likelihood; CRPS: continuous ranked probability score; PICP90/PIW90: coverage and mean width of the central 90% interval before (raw Gaussian) and after isotonic recalibration on the validation season. Nominal coverage: 90%.")


def t_ablation(num):
    V = [("DST-CT (full)", "Full model"), ("DST-CT w/o semantic constraints", "− semantic constraints"),
         ("DST-CT (point, no scenario spread)", "− scenario spread in σ̄ (eq. 8, first term only)"),
         ("DST-CT (deterministic, no diffusion)", "− diffusion (future = climatology)")]
    rows = []
    for ds in DSN:
        s = S(ds)
        for m, lab in V:
            row = {"Dataset": DSN[ds], "Variant": lab}
            for f in (0.0, 0.5, 1.0):
                r = s[(s.model == m) & np.isclose(s.frac, f)]
                if r.empty:
                    row[f"φ = {f:g}"] = "–"; continue
                r = r.iloc[0]
                row[f"φ = {f:g}"] = f"{r.RMSE:.3f} / {r.CRPS:.3f} / {r.PICP90:.0f} ({r.PICP90_raw:.0f})"
            rows.append(row)
    return pipe(pd.DataFrame(rows), f"Table {num}. Ablation of DST-CT components. Each cell: RMSE / CRPS (t ha^−1^) / PICP90 after recalibration (before recalibration, i.e. the '− isotonic recalibration' ablation), %. Mean over seeds.")


def t_spatial(num):
    rows = []
    for ds in ("wheat_IN", "maize_US"):
        try:
            s = S(ds, "spatial")
        except FileNotFoundError:
            continue
        for m in ["Climatology", "Linear trend", "3-season mean", "Ridge", "LightGBM", "LSTM", "Transformer", "DST-CT (deterministic, no diffusion)", "DST-CT (full)"]:
            row = {"Dataset": DSN[ds], "Model": NICE.get(m, m).replace("**", "")}
            for f in (0.0, 0.5, 1.0):
                r = s[(s.model == m) & np.isclose(s.frac, f)]
                row[f"RMSE, φ = {f:g}"] = f"{r.iloc[0].RMSE:.3f} [{r.iloc[0].RMSE_lo:.2f}, {r.iloc[0].RMSE_hi:.2f}]" if not r.empty else "–"
            r = s[(s.model == m) & np.isclose(s.frac, 0.5)]
            row["PICP90 (%), φ = 0.5"] = f"{r.iloc[0].PICP90:.1f}" if (not r.empty and not np.isnan(r.iloc[0].get("PICP90", np.nan))) else "–"
            rows.append(row)
    return pipe(pd.DataFrame(rows), f"Table {num}. Unseen regions (spatially blocked hold-out, one seed). The test regions and their yield history were excluded from training; 'climatology', 'trend' and '3-season mean' therefore fall back to the mean of the training regions. RMSE in t ha^−1^ [95% bootstrap CI].")


def t_transfer(num):
    fs = sorted(glob.glob("results/transfer_s*.json"))
    if not fs:
        return ""
    d = pd.DataFrame(sum([json.load(open(f)) for f in fs], []))
    g = d.groupby(["setting", "model", "frac"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), r2=("r2", "mean"), n=("seed", "nunique"),
                                                    r2123=("rmse_2021_2023", "mean")).reset_index()
    full = S("maize_US")
    rows = []
    settings = ["zero-shot", "fine-tune (1 US year)", "scratch (1 US year)", "fine-tune (3 US years)", "scratch (3 US years)"]
    def cell(r):
        if r.empty:
            return "–"
        r = r.iloc[0]
        return f"{r.rmse:.3f} ± {r.rmse_sd:.3f}" if r.n > 1 else f"{r.rmse:.3f}"
    for st in settings:
        for m in ["DST-CT", "DST-CT (deterministic)", "LSTM", "Ridge"]:
            row = {"Setting": st, "Model": m}
            for f in (0.0, 0.5, 1.0):
                row[f"RMSE, φ = {f:g}"] = cell(g[(g.setting == st) & (g.model == m) & np.isclose(g.frac, f)])
            r = g[(g.setting == st) & (g.model == m) & np.isclose(g.frac, 1.0)]
            row["RMSE, φ = 1 (2021–2023)"] = f"{r.iloc[0].r2123:.3f}" if not r.empty else "–"
            rows.append(row)
    for m in ("Linear trend (no training)", "3-season mean (no training)"):
        row = {"Setting": "reference", "Model": m}
        for f in (0.0, 0.5, 1.0):
            row[f"RMSE, φ = {f:g}"] = cell(g[(g.setting == "zero-shot") & (g.model == m) & np.isclose(g.frac, f)])
        r = g[(g.setting == "zero-shot") & (g.model == m) & np.isclose(g.frac, 1.0)]
        row["RMSE, φ = 1 (2021–2023)"] = f"{r.iloc[0].r2123:.3f}" if not r.empty else "–"
        rows.append(row)
    for m, mm in (("DST-CT", "DST-CT (full)"), ("DST-CT (deterministic)", "DST-CT (deterministic, no diffusion)"), ("LSTM", "LSTM"), ("Ridge", "Ridge")):
        row = {"Setting": "fully retrained (in-domain)", "Model": m}
        for f in (0.0, 0.5, 1.0):
            row[f"RMSE, φ = {f:g}"] = "–"
        r = full[(full.model == mm) & np.isclose(full.frac, 1.0)]
        row["RMSE, φ = 1 (2021–2023)"] = f"{r.iloc[0].RMSE:.3f}" if not r.empty else "–"
        rows.append(row)
    return pipe(pd.DataFrame(rows), f"Table {num}. Transfer from maize, India to maize, USA (test: all counties, harvest years 2019–2023). RMSE in t ha^−1^ (seed 0). Zero-shot: no parameter update and no US yield label; fine-tune/scratch: trained on the most recent one (2018) or three (2016–2018) US seasons. The last column restricts all settings to 2021–2023, the test years of the fully retrained in-domain models (expanding-window training on all US seasons up to *Y* − 2; table 7c).")
