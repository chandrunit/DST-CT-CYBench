"""Summary for revision round 2 (E1, E2, E4).

Merges, per fold (dataset, test year, seed), the stored predictions of run_fold.py (preds/) and revise2.py
(preds2/) — both share the same test indices — and writes:

  results2/yield_<ds>.csv       RMSE, R2, CRPS, conformal PICP90/PIW90 per model and issue time
                                (test seasons of all years pooled per seed, then mean +- sd over seeds)
  results2/paired_<ds>.csv      region-level paired comparison (Wilcoxon + bootstrap CI of the RMSE difference)
                                DST-CT v2 vs its ablation and vs every baseline
  results2/scenario_skill.csv   E1: CRPS of the generated rest-of-season predictors (test and validation years),
                                skill vs climatology and paired Wilcoxon Diffusion vs Analog (kNN)

E4 split-conformal intervals: for each model, issue time and fold the normalised scores |y - mu| / sd on the
validation year (Y-1) give the (1-alpha)(1+1/n) quantile q; the test interval is mu +- q sd. The validation year
is never used for training, so the intervals are finite-sample valid under exchangeability of years.
Usage: python summary2.py [ds ...]
"""
import sys, glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon

os.makedirs("results2", exist_ok=True)
DS = sys.argv[1:] or ["wheat_IN", "maize_IN", "maize_US"]
FR = ["0.0", "0.25", "0.5", "0.75", "1.0"]
REF = "DST-CT v2 (scenario-consistent twin)"


def crps_g(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def conformal(mu_v, sd_v, y_v, mu_t, sd_t, alpha=0.1):
    s = np.abs(y_v - mu_v) / sd_v
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    return mu_t - q * sd_t, mu_t + q * sd_t


def load_fold(p1, p2):
    out, meta = {}, {}
    for p in (p1, p2):
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        meta.setdefault("y_te", z["y_te"]); meta.setdefault("y_va", z["y_va"]); meta.setdefault("te", z["te"])
        for k in z.files:
            if "|" in k:
                m, f, part, what = k.split("|")
                if m.startswith("_"):
                    continue
                out.setdefault(m, {}).setdefault(str(float(f)), {})[(part, what)] = z[k]
    return out, meta


for ds in DS:
    rows, per_region = [], {}
    tags = sorted(glob.glob(f"preds2/{ds}_temporal_*_s*.npz"))
    tags = [t for t in tags if re.search(r"_s\d+\.npz$", t)]
    for p2 in tags:
        base = os.path.basename(p2)
        yr, seed = re.search(r"_(\d{4})_s(\d+)\.npz", base).groups()
        P, meta = load_fold(f"preds/{base}", p2)
        y, yv = meta["y_te"], meta["y_va"]
        adm = np.load(p2, allow_pickle=True)["adm_te"]
        for m, byf in P.items():
            for f, d in byf.items():
                if ("test", "mu") not in d:
                    continue
                mu, sd = d[("test", "mu")], d[("test", "sd")]
                lo, hi = conformal(d[("val", "mu")], d[("val", "sd")], yv, mu, sd)
                rows.append(dict(model=m, frac=float(f), year=int(yr), seed=int(seed), n=len(y),
                                 se=((mu - y) ** 2).sum(), crps=crps_g(mu, sd, y).sum(),
                                 cov=((y >= lo) & (y <= hi)).sum(), wid=(hi - lo).sum(), ysum=y.sum(), y2=(y ** 2).sum()))
                key = (m, float(f), int(seed))
                per_region.setdefault(key, []).append(pd.DataFrame({"adm": adm, "year": int(yr), "se": (mu - y) ** 2}))
    if not rows:
        print(ds, "no folds yet"); continue
    R = pd.DataFrame(rows)
    # pool test seasons over years per (model, frac, seed), then mean/sd over seeds
    g = R.groupby(["model", "frac", "seed"]).sum(numeric_only=True).reset_index()
    g["RMSE"] = np.sqrt(g.se / g.n)
    g["R2"] = 1 - g.se / (g.y2 - g.ysum ** 2 / g.n)
    g["CRPS"] = g.crps / g.n; g["PICP90_conf"] = 100 * g["cov"] / g.n; g["PIW90_conf"] = g.wid / g.n
    S = g.groupby(["model", "frac"]).agg(n_seeds=("seed", "nunique"), n_test=("n", "mean"),
                                         RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), R2=("R2", "mean"),
                                         CRPS=("CRPS", "mean"), PICP90_conf=("PICP90_conf", "mean"), PIW90_conf=("PIW90_conf", "mean")).reset_index()
    S.to_csv(f"results2/yield_{ds}.csv", index=False)
    # paired region-level comparison, seed-averaged squared errors
    pr = []
    for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        def reg(m):
            ks = [k for k in per_region if k[0] == m and k[1] == f]
            if not ks:
                return None
            d = pd.concat([pd.concat(per_region[k]).assign(seed=k[2]) for k in ks])
            return d.groupby(["adm", "year"]).se.mean()
        a = reg(REF)
        if a is None:
            continue
        for m in sorted({k[0] for k in per_region}):
            if m == REF:
                continue
            b = reg(m)
            if b is None:
                continue
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            ra = j.groupby(level=0).a.mean(); rb = j.groupby(level=0).b.mean()  # per region (cluster)
            rng = np.random.default_rng(0); regs = ra.index.values; bs = []
            for _ in range(1000):
                s_ = rng.choice(len(regs), len(regs))
                bs.append(np.sqrt(ra.values[s_].mean()) - np.sqrt(rb.values[s_].mean()))
            try:
                pval = wilcoxon(ra.values, rb.values).pvalue
            except ValueError:
                pval = np.nan
            pr.append(dict(frac=f, other=m, rmse_v2=np.sqrt(j.a.mean()), rmse_other=np.sqrt(j.b.mean()),
                           d_rmse=np.sqrt(j.a.mean()) - np.sqrt(j.b.mean()), d_lo=np.percentile(bs, 2.5), d_hi=np.percentile(bs, 97.5),
                           wilcoxon_p=pval, n_regions=len(regs)))
    pd.DataFrame(pr).to_csv(f"results2/paired_{ds}.csv", index=False)
    print(f"== {ds}: {R.seed.nunique()} seeds, years {sorted(R.year.unique())}")
    show = S[S.frac.isin([0.0, 0.5, 1.0])].sort_values(["frac", "RMSE"])
    print(show[["model", "frac", "n_seeds", "RMSE", "RMSE_sd", "CRPS", "PICP90_conf", "PIW90_conf"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- E1 scenario skill
e1rows = []
for fn in glob.glob("preds2/*_temporal_*_s*.json"):
    if "_e1_" in fn:
        continue
    d = json.load(open(fn))
    for split, key in (("test", "E1"), ("validation", "E1_val")):
        if key not in d:
            continue
        for f, res in d[key].items():
            ps = {m: np.array(v["_per_season"], float) for m, v in res.items()}
            ok = ~np.isnan(ps["Diffusion"]) & ~np.isnan(ps["Analog (kNN)"])
            try:
                pw = wilcoxon(ps["Diffusion"][ok], ps["Analog (kNN)"][ok]).pvalue
            except ValueError:
                pw = np.nan
            for m, v in res.items():
                r = dict(ds=d["ds"], year=d["year"], seed=d["seed"], split=split, frac=float(f), method=m, CRPS=v["all"],
                         p_diff_vs_knn=pw)
                r.update({k: v[k] for k in ("ndvi", "fpar", "ssm", "rsm", "prec", "tmax") if k in v})
                e1rows.append(r)
if e1rows:
    E = pd.DataFrame(e1rows)
    E.to_csv("results2/scenario_skill_folds.csv", index=False)
    T = E.groupby(["ds", "split", "frac", "method"]).mean(numeric_only=True).reset_index().drop(columns=["year", "seed"])
    clim = T[T.method == "Climatology"].set_index(["ds", "split", "frac"]).CRPS
    T["skill_vs_clim_%"] = 100 * (1 - T.CRPS / T.set_index(["ds", "split", "frac"]).index.map(clim).values)
    T.to_csv("results2/scenario_skill.csv", index=False)
    print("== E1 scenario skill (test years, mean over folds)")
    print(T[T.split == "test"][["ds", "frac", "method", "CRPS", "skill_vs_clim_%", "ndvi", "ssm"]].round(3).to_string(index=False))
