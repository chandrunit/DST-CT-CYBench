"""Aggregate fold predictions -> tables (CSV/JSON) with seed SD, region-cluster bootstrap 95% CI and paired tests."""
import glob, json, re, numpy as np, pandas as pd, sys
from collections import defaultdict
from scipy.stats import norm, wilcoxon
from lib import crps_gauss, IsoCal

rng = np.random.default_rng(0)
FR = ["0.0", "0.25", "0.5", "0.75", "1.0"]
MAIN = "DST-CT (full)"


def collect(pattern):
    """returns dict[(model, frac, seed)] -> DataFrame(adm, year, y, mu, sd, lo90, hi90) pooled over test years"""
    rows = defaultdict(list)
    files = [f for f in glob.glob(pattern) if not f.endswith("_scen.npz") and "_T" not in f.split("/")[-1]]
    meta = []
    for fn in sorted(files):
        z = np.load(fn, allow_pickle=True)
        m = re.search(r"_(\d{4})_s(\d+)", fn); year, seed = int(m.group(1)), int(m.group(2))
        j = json.load(open(fn.replace(".npz", ".json"))); meta.append(j)
        ds = j["ds"]
        from lib import load
        D = load_cached(ds)
        te, va = z["te"], z["va"]
        models = sorted({k.split("|")[0] for k in z.files if "|" in k})
        for mdl in models:
            for f in FR:
                key = f"{mdl}|{f}|test|mu"
                if key not in z.files:
                    continue
                mu, sd = z[key], z[f"{mdl}|{f}|test|sd"]
                mv, sv = z[f"{mdl}|{f}|val|mu"], z[f"{mdl}|{f}|val|sd"]
                ic = IsoCal().fit(mv, sv, z["y_va"])
                lo, hi = ic.interval(mu, sd, 0.9)
                pit_cal = {}
                rows[(mdl, f, seed)].append(pd.DataFrame(dict(adm=D["adm"][te], year=year, y=z["y_te"], mu=mu, sd=sd, lo90=lo, hi90=hi,
                                                              lo50=ic.interval(mu, sd, 0.5)[0], hi50=ic.interval(mu, sd, 0.5)[1],
                                                              lo80=ic.interval(mu, sd, 0.8)[0], hi80=ic.interval(mu, sd, 0.8)[1],
                                                              lo95=ic.interval(mu, sd, 0.95)[0], hi95=ic.interval(mu, sd, 0.95)[1],
                                                              lo70=ic.interval(mu, sd, 0.7)[0], hi70=ic.interval(mu, sd, 0.7)[1],
                                                              lo30=ic.interval(mu, sd, 0.3)[0], hi30=ic.interval(mu, sd, 0.3)[1])))
    return {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}, meta


_cache = {}
def load_cached(ds):
    if ds not in _cache:
        from lib import load
        _cache[ds] = load(ds)
    return _cache[ds]


def metrics(d, prob=True):
    e = d.mu - d.y
    r = dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), R2=1 - np.sum(e ** 2) / np.sum((d.y - d.y.mean()) ** 2),
             Bias=np.mean(e), nRMSE=100 * np.sqrt(np.mean(e ** 2)) / d.y.mean())
    if prob:
        r.update(NLL=np.mean(-norm.logpdf(d.y, d.mu, d.sd)), CRPS=np.mean(crps_gauss(d.mu.values, d.sd.values, d.y.values)),
                 PICP90=100 * np.mean((d.y >= d.lo90) & (d.y <= d.hi90)), PIW90=np.mean(d.hi90 - d.lo90),
                 PICP90_raw=100 * np.mean(np.abs(e) <= 1.645 * d.sd))
    return r


def boot_rmse(dfs, B=1000):
    """cluster bootstrap over regions; metric averaged over seeds inside each replicate"""
    regs = sorted(set(dfs[0].adm))
    ridx = {r: i for i, r in enumerate(regs)}
    sse = np.zeros((len(dfs), len(regs))); cnt = np.zeros(len(regs)); sae = np.zeros((len(dfs), len(regs)))
    for s, d in enumerate(dfs):
        g = d.assign(se=(d.mu - d.y) ** 2, ae=(d.mu - d.y).abs()).groupby("adm")
        ss = g.se.sum(); aa = g.ae.sum(); cc = g.size()
        for r, v in ss.items():
            sse[s, ridx[r]] = v; sae[s, ridx[r]] = aa[r]
            if s == 0:
                cnt[ridx[r]] = cc[r]
    w = rng.multinomial(len(regs), np.ones(len(regs)) / len(regs), size=B)
    rm = np.sqrt((sse @ w.T) / (cnt @ w.T)).mean(0)
    ma = ((sae @ w.T) / (cnt @ w.T)).mean(0)
    return np.percentile(rm, [2.5, 97.5]), np.percentile(ma, [2.5, 97.5]), sse, cnt, w


def summarize(P, models, fracs, prob_models):
    out = []
    for mdl in models:
        for f in fracs:
            seeds = sorted({s for (m, ff, s) in P if m == mdl and ff == f})
            if not seeds:
                continue
            dfs = [P[(mdl, f, s)] for s in seeds]
            ms = pd.DataFrame([metrics(d, mdl in prob_models) for d in dfs])
            ci_r, ci_m, *_ = boot_rmse(dfs)
            row = dict(model=mdl, frac=float(f), n_seeds=len(seeds), n_test=len(dfs[0]))
            for c in ms.columns:
                row[c] = ms[c].mean(); row[c + "_sd"] = ms[c].std(ddof=0)
            row["RMSE_lo"], row["RMSE_hi"] = ci_r; row["MAE_lo"], row["MAE_hi"] = ci_m
            out.append(row)
    return pd.DataFrame(out)


def paired(P, a, b, f):
    """DST-CT vs baseline b: bootstrap CI of RMSE difference (a-b) + Wilcoxon on region-mean squared-error differences."""
    sa = sorted({s for (m, ff, s) in P if m == a and ff == f}); sb = sorted({s for (m, ff, s) in P if m == b and ff == f})
    da = pd.concat([P[(a, f, s)].assign(seed=s) for s in sa]).groupby(["adm", "year"]).agg(y=("y", "first"), mu=("mu", "mean")).reset_index()
    db = pd.concat([P[(b, f, s)].assign(seed=s) for s in sb]).groupby(["adm", "year"]).agg(y=("y", "first"), mu=("mu", "mean")).reset_index()
    m = da.merge(db, on=["adm", "year", "y"], suffixes=("_a", "_b"))
    m["da"] = (m.mu_a - m.y) ** 2; m["db"] = (m.mu_b - m.y) ** 2
    reg = m.groupby("adm")[["da", "db"]].mean()
    st = wilcoxon(reg.da, reg.db) if (reg.da != reg.db).any() else None
    regs = reg.index.values
    g = m.groupby("adm")
    sa_ = g.da.sum().values; sb_ = g.db.sum().values; cn = g.size().values
    w = rng.multinomial(len(regs), np.ones(len(regs)) / len(regs), size=1000)
    diff = np.sqrt((sa_ @ w.T) / (cn @ w.T)) - np.sqrt((sb_ @ w.T) / (cn @ w.T))
    return dict(frac=float(f), baseline=b, rmse_a=float(np.sqrt(m.da.mean())), rmse_b=float(np.sqrt(m.db.mean())),
                d_rmse=float(np.sqrt(m.da.mean()) - np.sqrt(m.db.mean())), d_lo=float(np.percentile(diff, 2.5)), d_hi=float(np.percentile(diff, 97.5)),
                wilcoxon_p=float(st.pvalue) if st else 1.0, n_regions=len(regs))


def deep_ensemble(pattern):
    """Mixture of the deterministic twins across seeds (residual space -> yield)."""
    out = defaultdict(list)
    for fn in sorted(f for f in glob.glob(pattern) if not f.endswith("_scen.npz") and "_T" not in f.split("/")[-1]):
        m = re.search(r"_(\d{4})_s(\d+)", fn); out[int(m.group(1))].append(fn)
    rows = defaultdict(list)
    for year, fns in out.items():
        zs = [np.load(f) for f in fns]
        z0 = zs[0]; D = load_cached(json.load(open(fns[0].replace(".npz", ".json")))["ds"])
        for f in FR:
            parts = {}
            for part in ("val", "test"):
                mus = np.array([z[f"_twin_raw|{f}|{part}|mu"] for z in zs]); sds = np.array([z[f"_twin_raw|{f}|{part}|sd"] for z in zs])
                mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
                tr = z0["trend_te"] if part == "test" else z0["trend_va"]
                parts[part] = (tr + mu * z0["rsd"], sd * z0["rsd"])
            ic = IsoCal().fit(*parts["val"], z0["y_va"])
            mu, sd = parts["test"]; lo, hi = ic.interval(mu, sd, 0.9)
            d = dict(adm=D["adm"][z0["te"]], year=year, y=z0["y_te"], mu=mu, sd=sd, lo90=lo, hi90=hi)
            for lvl in (0.3, 0.5, 0.7, 0.8, 0.95):
                l, h = ic.interval(mu, sd, lvl); d[f"lo{int(lvl*100)}"] = l; d[f"hi{int(lvl*100)}"] = h
            rows[f].append(pd.DataFrame(d))
    return {(f"Deep ensemble ({len(fns)} twins)", f, 0): pd.concat(v, ignore_index=True) for f, v in rows.items()}


if __name__ == "__main__":
    ds = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "temporal"
    pat = f"preds/{ds}_{mode}_*_s*.npz" if mode == "temporal" else f"preds/{ds}_spatial_*.npz"
    pat_files = [f for f in glob.glob(pat) if "_T" not in f.split("/")[-1]]
    P, meta = collect(pat)
    P = {k: v for k, v in P.items() if not k[0].startswith("_")}
    if mode == "temporal":
        P.update(deep_ensemble(pat))
    models = sorted({k[0] for k in P})
    prob = [m for m in models if m not in ("Ridge", "LightGBM", "VI regression", "Persistence", "Climatology", "Linear trend", "3-season mean")]
    S = summarize(P, models, FR, prob)
    S.to_csv(f"results/summary_{ds}_{mode}.csv", index=False)
    pr = []
    for b in models:
        if b == MAIN:
            continue
        for f in FR:
            try:
                pr.append(paired(P, MAIN, b, f))
            except Exception as e:
                pass
    pd.DataFrame(pr).to_csv(f"results/paired_{ds}_{mode}.csv", index=False)
    # per-year table
    py = []
    for (mdl, f, s), d in P.items():
        for y, g in d.groupby("year"):
            r = metrics(g, False); r.update(model=mdl, frac=float(f), seed=s, year=y); py.append(r)
    pd.DataFrame(py).groupby(["model", "frac", "year"]).mean(numeric_only=True).reset_index().to_csv(f"results/peryear_{ds}_{mode}.csv", index=False)
    # reliability (calibrated coverage at several nominal levels)
    rel = []
    for (mdl, f, s), d in P.items():
        if mdl not in prob:
            continue
        for lvl in (30, 50, 70, 80, 90, 95):
            rel.append(dict(model=mdl, frac=float(f), seed=s, nominal=lvl, cov=100 * np.mean((d.y >= d[f"lo{lvl}"]) & (d.y <= d[f"hi{lvl}"])),
                            cov_raw=100 * np.mean(np.abs(d.y - d.mu) <= norm.ppf(0.5 + lvl / 200) * d.sd)))
    pd.DataFrame(rel).groupby(["model", "frac", "nominal"]).mean(numeric_only=True).reset_index().to_csv(f"results/reliability_{ds}_{mode}.csv", index=False)
    json.dump(meta, open(f"results/meta_{ds}_{mode}.json", "w"))
    print(S[S.frac.isin([0.0, 0.5, 1.0])][["model", "frac", "n_seeds", "RMSE", "RMSE_sd", "RMSE_lo", "RMSE_hi", "MAE", "R2", "CRPS", "PICP90"]].round(3).to_string())
