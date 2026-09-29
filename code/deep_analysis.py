"""Analyses on a saved fold: sampling-step sensitivity + latency, cloud-gap repair, modality attention,
permutation importance, case-study trajectories."""
import argparse, json, time, numpy as np, torch
from scipy.stats import norm
from ctx import *

p = argparse.ArgumentParser()
p.add_argument("--ds"); p.add_argument("--year", type=int); p.add_argument("--seed", type=int, default=0)
p.add_argument("--parts", default="steps,gap,attn,perm,case")
p.add_argument("--steps_list", default="5,10,25,50,100,250,1000")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
c = fold_context(a.ds, a.year)
D, tr, va, te, prep, Sst, ins, K = c["D"], c["tr"], c["va"], c["te"], c["prep"], c["Sst"], c["ins"], c["K"]
tag = f"{a.ds}_temporal_{a.year}_s{a.seed}"
twin, dif = load_models(f"preds/{tag}_models.pt", Sst.shape[1], K)
out = {}
Tt = lambda x: torch.tensor(np.asarray(x, np.float32))


def twin_mu(Z, idx):
    with torch.no_grad():
        mu, lv = twin(Tt(Z), Tt(Sst[idx]), Tt(ins[idx]))
    return mu.numpy(), np.exp(0.5 * lv.numpy())


def dstct(idx, f, steps, M=10, seed=0, return_scen=False):
    Xc, miss, cl = prep.causal_fill(D["X"][idx], D["adm"][idx])
    Zc = prep.norm(Xc).astype(np.float32)
    o = n_obs(D["nd"][idx], f); obs = (np.arange(K)[None] < o[:, None]).astype(np.float32)
    cmask = (obs[..., None] * (~miss)).astype(np.float32)
    t0 = time.time()
    sc = dif.sample(Zc, cmask, obs, ins[idx], Sst[idx], np.full(len(idx), f, np.float32), M=M, steps=steps, seed=seed)
    proj, _ = project_plausible(prep.denorm(sc), prep)
    Zs = prep.norm(proj)
    mus, sds = zip(*[twin_mu(Zs[m], idx) for m in range(M)])
    mus, sds = np.array(mus), np.array(sds)
    dt = time.time() - t0
    mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
    y_mu = c["trend"][idx] + mu * c["rsd"]; y_sd = sd * c["rsd"]
    if return_scen:
        return y_mu, y_sd, dt, proj
    return y_mu, y_sd, dt


def metr(mu, sd, y, mv=None, sv=None, yv=None):
    r = dict(rmse=float(np.sqrt(np.mean((mu - y) ** 2))), crps=float(crps_gauss(mu, sd, y).mean()),
             cov90_raw=float(np.mean(np.abs(y - mu) <= 1.645 * sd)))
    if mv is not None:
        ic = IsoCal().fit(mv, sv, yv); lo, hi = ic.interval(mu, sd, 0.9)
        r["picp90_cal"] = float(np.mean((y >= lo) & (y <= hi))); r["piw90_cal"] = float(np.mean(hi - lo))
    return r


y_te, y_va = D["y"][te], D["y"][va]
if "steps" in a.parts:
    res = []
    for S in [int(s) for s in a.steps_list.split(",")]:
        for f in (0.0, 0.5):
            mv, sv, _ = dstct(va, f, S, seed=a.seed)
            mu, sd, dt = dstct(te, f, S, seed=a.seed)
            r = metr(mu, sd, y_te, mv, sv, y_va); r.update(steps=S, frac=f, sec_per_1000_regions=dt / len(te) * 1000)
            res.append(r); print(r, flush=True)
    # deterministic twin and LSTM-like single-pass latency for reference
    Z, ob, _ = build_inputs(D, te, prep, 0.5)
    t0 = time.time(); twin_mu(Z, te); dt = time.time() - t0
    out["det_sec_per_1000_regions"] = dt / len(te) * 1000
    out["steps"] = res
    out["n_params"] = dict(twin=sum(p.numel() for p in twin.parameters()), denoiser=sum(p.numel() for p in dif.net.parameters()))

if "gap" in a.parts:
    # artificial cloud gaps: hide 30% of genuinely observed in-season NDVI dekads in full-season test data
    rng = np.random.default_rng(a.seed)
    Xraw = D["X"][te].copy()
    ndvi_ok = ~np.isnan(Xraw[:, :, 0]) & (ins[te] > 0)
    hide = ndvi_ok & (rng.random(ndvi_ok.shape) < 0.3)
    truth = Xraw[:, :, 0][hide]
    Xh = Xraw.copy(); Xh[:, :, 0][hide] = np.nan
    # (1) causal LOCF (what the baselines use)
    Xl, missh, cl = prep.causal_fill(Xh, D["adm"][te])
    # (2) climatology
    clim = cl[:, :, 0]
    # (3) linear interpolation (NON-causal reference, uses the future dekad)
    lin = Xh[:, :, 0].copy()
    for i in range(len(lin)):
        v = lin[i]; ok = ~np.isnan(v)
        if ok.sum() >= 2:
            lin[i] = np.interp(np.arange(K), np.where(ok)[0], v[ok])
        else:
            lin[i] = clim[i]
    # (4) diffusion: hidden NDVI entries are generated conditionally on everything else observed
    Zc = prep.norm(Xl).astype(np.float32)
    obs = ins[te].copy()
    cmask = (obs[..., None] * (~missh)).astype(np.float32)
    sc = dif.sample(Zc, cmask, obs, ins[te], Sst[te], np.ones(len(te), np.float32), M=10, steps=50, seed=a.seed)
    dnd = prep.denorm(sc)[..., 0].mean(0)
    rm = lambda est: float(np.sqrt(np.mean((est[hide] - truth) ** 2)))
    g = dict(n_hidden=int(hide.sum()), rmse_locf=rm(Xl[:, :, 0]), rmse_clim=rm(clim), rmse_linear_noncausal=rm(lin), rmse_diffusion=rm(dnd))
    # effect on the yield forecast at end of season
    Xd = Xl.copy(); Xd[:, :, 0] = np.where(hide, dnd, Xl[:, :, 0])
    Xo, _, _ = prep.causal_fill(Xraw, D["adm"][te])
    for name, X in (("original", Xo), ("gaps_locf", Xl), ("gaps_diffusion", Xd)):
        mu, sd = twin_mu(prep.norm(X), te)
        g[f"yield_rmse_{name}"] = float(np.sqrt(np.mean((c["trend"][te] + mu * c["rsd"] - y_te) ** 2)))
    out["gap"] = g; print(g, flush=True)
    # example series for a figure
    k = int(np.argmax(hide.sum(1)))
    out["gap_example"] = dict(adm=str(D["adm"][te][k]), truth=np.nan_to_num(Xraw[k, :, 0], nan=-9).tolist(), hidden=hide[k].tolist(),
                              locf=Xl[k, :, 0].tolist(), diffusion=dnd[k].tolist(), linear=lin[k].tolist(),
                              scen=prep.denorm(sc)[:, k, :, 0].tolist(), nd=int(D["nd"][te][k]))

if "attn" in a.parts:
    Z, ob, _ = build_inputs(D, te, prep, 1.0)
    with torch.no_grad():
        twin(Tt(Z), Tt(Sst[te]), Tt(ins[te]))
    al = twin.last_alpha.numpy()  # N,K,3
    w = ins[te][..., None]
    out["attn_by_dekad"] = ((al * w).sum(0) / w.sum(0).clip(1)).tolist()
    # by phase (relative position in season: thirds)
    rel = np.arange(K)[None] / D["nd"][te][:, None]
    ph = {}
    for nm, lo, hi in (("early", 0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("late", 2 / 3, 1.0001)):
        m = ((rel >= lo) & (rel < hi) & (ins[te] > 0))[..., None]
        ph[nm] = ((al * m).sum((0, 1)) / m.sum((0, 1)).clip(1)).tolist()
    out["attn_by_phase"] = ph; print(ph, flush=True)

if "perm" in a.parts:
    Z, ob, _ = build_inputs(D, te, prep, 1.0)
    base_mu, _ = twin_mu(Z, te)
    base = np.sqrt(np.mean((c["trend"][te] + base_mu * c["rsd"] - y_te) ** 2))
    rng = np.random.default_rng(0)
    rel = np.arange(K)[None] / D["nd"][te][:, None]
    imp = {}
    for gname, chs in GROUPS.items():
        for nm, lo, hi in (("early", 0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("late", 2 / 3, 1.0001)):
            vals = []
            for rep in range(5):
                Zp = Z.copy(); perm = rng.permutation(len(te))
                m = (rel >= lo) & (rel < hi)
                for ch in chs:
                    Zp[:, :, ch] = np.where(m, Z[perm][:, :, ch], Z[:, :, ch])
                mu, _ = twin_mu(Zp, te)
                vals.append(np.sqrt(np.mean((c["trend"][te] + mu * c["rsd"] - y_te) ** 2)) - base)
            imp[f"{gname}|{nm}"] = float(np.mean(vals))
    # static site + yield-history block
    vals = []
    for rep in range(5):
        Sp = Sst.copy(); perm = rng.permutation(len(te)); Sp[te] = Sst[te][perm]
        with torch.no_grad():
            mu, _ = twin(Tt(Z), Tt(Sp[te]), Tt(ins[te]))
        vals.append(np.sqrt(np.mean((c["trend"][te] + mu.numpy() * c["rsd"] - y_te) ** 2)) - base)
    imp["static+history"] = float(np.mean(vals))
    out["perm_importance"] = imp; out["perm_base_rmse"] = float(base); print(imp, flush=True)

if "case" in a.parts:
    # two case regions: largest negative and largest positive yield anomaly vs trend in the test year
    an = y_te - c["trend"][te]
    ks = [int(np.argmin(an)), int(np.argmax(an))]
    cases = []
    for k in ks:
        idx = te[[k]]
        rec = dict(adm=str(D["adm"][idx][0]), y=float(y_te[k]), trend=float(c["trend"][te][k]), nd=int(D["nd"][idx][0]),
                   ndvi=np.nan_to_num(D["X"][idx][0, :, 0], nan=-9).tolist(), fc={})
        for f in FRACS:
            mu, sd, _, proj = dstct(idx, f, 50, M=30, seed=a.seed, return_scen=True)
            Z, ob, _ = build_inputs(D, idx, prep, f); dm, ds_ = twin_mu(Z, idx)
            rec["fc"][str(f)] = dict(mu=float(mu[0]), sd=float(sd[0]), det=float(c["trend"][idx][0] + dm[0] * c["rsd"]),
                                     det_sd=float(ds_[0] * c["rsd"]))
            if f == 0.5:
                rec["scen_ndvi"] = proj[:, 0, :, 0].tolist()
                rec["clim_ndvi"] = prep.clim_for(D["adm"][idx])[0, :, 0].tolist()
        cases.append(rec)
    out["cases"] = cases

json.dump(out, open(f"results/deep_{tag}.json", "w"))
print("saved")
