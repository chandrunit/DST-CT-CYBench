"""One fold = one (dataset, test year, seed[, held-out spatial cluster]).
Temporal protocol: train = seasons with harvest year < Y-1, validation = Y-1 (early stopping and
calibration only), test = Y. Nothing from year >= Y is touched before prediction.
Spatial protocol: additionally, one k-means cluster of regions (lat/lon) is removed from train/val
and is the only test set (unseen regions AND unseen year); its own yield history is not used."""
import argparse, json, time, os, numpy as np, torch, lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from lib import *

p = argparse.ArgumentParser()
p.add_argument("--ds"); p.add_argument("--year", type=int); p.add_argument("--seed", type=int, default=0)
p.add_argument("--mode", default="temporal"); p.add_argument("--cluster", type=int, default=-1)
p.add_argument("--T", type=int, default=1000); p.add_argument("--steps", type=int, default=50)
p.add_argument("--M", type=int, default=10); p.add_argument("--out", default="preds")
p.add_argument("--only_dstct", action="store_true")
p.add_argument("--save_models", action="store_true")
p.add_argument("--dif_epochs", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
tag = f"{a.ds}_{a.mode}_{a.year}_s{a.seed}" + (f"_c{a.cluster}" if a.mode == "spatial" else "") + (f"_T{a.T}" if a.T != 1000 else "")
D = load(a.ds); yr = D["year"]; N = len(yr)
timing = {}

# ---------------- split
reg = np.ones(N, bool)
if a.mode == "spatial":
    lat, lon = D["S"][:, 3], D["S"][:, 4]
    u, inv = np.unique(D["adm"], return_index=True)
    km = KMeans(5, n_init=10, random_state=0).fit(np.c_[lat[inv], lon[inv]])
    lab = dict(zip(u, km.labels_)); cl = np.array([lab[x] for x in D["adm"]])
    reg = cl != a.cluster
tr = np.where((yr < a.year - 1) & reg)[0]
va = np.where((yr == a.year - 1) & reg)[0]
te = np.where((yr == a.year) & (reg if a.mode == "temporal" else ~reg))[0]

# ---------------- history features (past yields only; unseen regions never use their own)
if a.mode == "temporal":
    clim, last, trend, mean3 = history_features(D, set(yr[yr < a.year]))
else:
    clim, last, trend, mean3 = history_features(D, set(yr[yr < a.year]), True, train_mask=reg)
    c2, l2, t2, m2 = history_features(D, set(yr[yr < a.year]), False, train_mask=reg)
    clim[te], last[te], trend[te], mean3[te] = c2[te], l2[te], t2[te], m2[te]
for v in (clim, last, trend, mean3):
    v[np.isnan(v)] = np.nanmean(D["y"][tr])
prep = Prep(D, tr)
ymu, ysd = D["y"][tr].mean(), D["y"][tr].std()
H = np.c_[(clim - ymu) / ysd, (last - ymu) / ysd, (trend - ymu) / ysd, (mean3 - ymu) / ysd, (yr - 2010) / 10.0].astype(np.float32)
Sst = np.c_[prep.nstat(D["S"]), H].astype(np.float32)
lin_trend = trend.copy()
trend = mean3 if ANCHOR[D["ds"]] == "mean3" else trend  # anchor of the learned residual
res = D["y"] - trend
rsd = res[tr].std()
R = (res / rsd).astype(np.float32)  # learning target: detrended residual (trend from past yields only)
K = D["X"].shape[1]
ins_all = (np.arange(K)[None] < D["nd"][:, None]).astype(np.float32)

def to_y(mu_r, lv_r, idx):
    return trend[idx] + mu_r * rsd, np.exp(0.5 * lv_r) * rsd

P = {}  # P[model][frac] = dict(val=(mu,sd), test=(mu,sd))
def put(m, f, part, mu, sd):
    P.setdefault(m, {}).setdefault(f, {})[part] = (np.asarray(mu, np.float64), np.maximum(np.asarray(sd, np.float64), 1e-3))

inputs = {}
for f in FRACS:
    for part, idx in (("tr", tr), ("val", va), ("test", te)):
        Z, obs, miss = build_inputs(D, idx, prep, f)
        inputs[(f, part)] = (Z, obs)

# ---------------- classical reference baselines (no training)
if not a.only_dstct:
    for part, idx in (("val", va), ("test", te)):
        for name, v in (("Persistence", last), ("Climatology", clim), ("Linear trend", lin_trend), ("3-season mean", mean3)):
            s = np.sqrt(np.mean((D["y"][va] - v[va]) ** 2))
            for f in FRACS:
                put(name, f, part, v[idx], np.full(len(idx), s))

    # ---------------- feature models (trained on all issue times stacked, issue time is a feature)
    def feats(f, part, idx, veg_only=False):
        Z, obs = inputs[(f, part)]
        if veg_only:
            Zs = Z[:, :, VEG]
            fs = np.c_[Zs[:, :6].mean(1), Zs[:, 6:12].mean(1), Zs[:, 12:].mean(1), Zs.max(1), obs.mean(1, keepdims=True), H[idx][:, 2:3]]
            return fs
        return np.c_[summary_features(Z, obs, ins_all[idx]), Sst[idx]]
    # linear models: ridge penalty chosen on the validation season; LightGBM: early stopping on the validation season
    Xva_all = {}
    for name, veg in (("VI regression", True), ("Ridge", False), ("LightGBM", False)):
        t0 = time.time()
        Xtr = np.concatenate([feats(f, "tr", tr, veg) for f in FRACS]); Ytr = np.concatenate([R[tr]] * len(FRACS))
        Xv = np.concatenate([feats(f, "val", va, veg) for f in FRACS]); Yv = np.concatenate([R[va]] * len(FRACS))
        if name == "LightGBM":
            m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=a.seed, verbose=-1, n_jobs=1)
            m.fit(Xtr, Ytr, eval_set=[(Xv, Yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            timing["LightGBM_trees"] = int(m.best_iteration_ or 2000)
        else:
            best = None
            for al in (0.1, 1.0, 10.0, 100.0, 1000.0):
                mm = Ridge(al).fit(Xtr, Ytr); e = np.mean((mm.predict(Xv) - Yv) ** 2)
                if best is None or e < best[0]:
                    best = (e, al, mm)
            m = best[2]; timing[name + "_alpha"] = best[1]
        timing[name + "_train_s"] = time.time() - t0
        for f in FRACS:
            pv = m.predict(feats(f, "val", va, veg)); s = np.sqrt(np.mean((R[va] - pv) ** 2))
            for part, idx in (("val", va), ("test", te)):
                t1 = time.time(); pr = m.predict(feats(f, part, idx, veg))
                if part == "test": timing.setdefault(name + "_infer_s", 0); timing[name + "_infer_s"] += time.time() - t1
                mu, sd = to_y(pr, np.full(len(idx), 2 * np.log(s)), idx)
                put(name, f, part, mu, sd)

T = lambda x: torch.tensor(x)
St_all = T(Sst); ins_t = T(ins_all); Rt = T(R)

# ---------------- sequence baselines trained with random issue times (future = climatology)
def seq_batch(idx, frac_choice):
    Zs, obs = [], []
    for f in frac_choice:
        pass

def make_seq_trainer(net, idx_pool, part_name):
    # pre-build tensors for each issue time
    Zf = {f: T(inputs[(f, part_name)][0]) for f in FRACS}
    Of = {f: T(inputs[(f, part_name)][1]) for f in FRACS}
    rng = np.random.default_rng(a.seed)
    def fn(ib, ep):
        fsel = rng.integers(0, len(FRACS), len(ib))
        x = torch.stack([Zf[FRACS[k]][i] for k, i in zip(fsel, ib)])
        ob = torch.stack([Of[FRACS[k]][i] for k, i in zip(fsel, ib)])
        gi = idx_pool[ib]
        mu, lv = net(x, St_all[gi], ins_t[gi], ob[..., None])
        return gauss_nll(mu, lv, Rt[gi])
    return fn

def val_nll(net, fracs=FRACS):
    tot = 0
    for f in fracs:
        Z, ob = inputs[(f, "val")]
        mu, lv = net(T(Z), St_all[va], ins_t[va], T(ob)[..., None])
        tot += gauss_nll(mu, lv, Rt[va]).item()
    return tot / len(fracs)

def predict_seq(name, net):
    for f in FRACS:
        for part, idx in (("val", va), ("test", te)):
            Z, ob = inputs[(f, part)]
            t1 = time.time()
            with torch.no_grad():
                mu, lv = net(T(Z), St_all[idx], ins_t[idx], T(ob)[..., None])
            if part == "test": timing.setdefault(name + "_infer_s", 0); timing[name + "_infer_s"] += time.time() - t1
            put(name, f, part, *to_y(mu.numpy(), lv.numpy(), idx))

ep_max = 60
if not a.only_dstct:
    for name, cls in (("LSTM", LSTMNet), ("Transformer", TransNet)):
        torch.manual_seed(a.seed)
        net = cls(C, Sst.shape[1]) if name == "LSTM" else cls(C, Sst.shape[1], K=K)
        t0 = time.time()
        ne = train_net(net, make_seq_trainer(net, tr, "tr"), len(tr), ep_max, seed=a.seed, val_fn=lambda: val_nll(net))
        timing[name + "_train_s"] = time.time() - t0; timing[name + "_epochs"] = ne
        predict_seq(name, net)

# ---------------- digital twin (state-space GRU) trained on complete seasons
torch.manual_seed(a.seed)
twin = TwinGRU(C, Sst.shape[1])
Zfull_tr = T(inputs[(1.0, "tr")][0])
def twin_fn(ib, ep):
    gi = tr[ib]; mu, lv = twin(Zfull_tr[ib], St_all[gi], ins_t[gi]); return gauss_nll(mu, lv, Rt[gi])
def twin_val():
    Z = T(inputs[(1.0, "val")][0]); mu, lv = twin(Z, St_all[va], ins_t[va]); return gauss_nll(mu, lv, Rt[va]).item()
t0 = time.time(); ne = train_net(twin, twin_fn, len(tr), ep_max, seed=a.seed, val_fn=twin_val)
timing["Twin_train_s"] = time.time() - t0; timing["Twin_epochs"] = ne

def twin_pred(Z, idx, mc=0):
    with torch.no_grad():
        if mc:
            twin.train(); outs = [twin(T(Z), St_all[idx], ins_t[idx]) for _ in range(mc)]; twin.eval()
            mus = torch.stack([o[0] for o in outs]); lvs = torch.stack([o[1] for o in outs])
            m = mus.mean(0); v = (lvs.exp()).mean(0) + mus.var(0)
            return m.numpy(), torch.log(v).numpy()
        mu, lv = twin(T(Z), St_all[idx], ins_t[idx]); return mu.numpy(), lv.numpy()

for f in FRACS:
    for part, idx in (("val", va), ("test", te)):
        Z, ob = inputs[(f, part)]
        t1 = time.time(); mu, lv = twin_pred(Z, idx)
        if part == "test": timing.setdefault("Det_infer_s", 0); timing["Det_infer_s"] += time.time() - t1
        put("DST-CT (deterministic, no diffusion)", f, part, *to_y(mu, lv, idx))
        put("_twin_raw", f, part, mu, np.exp(0.5 * lv))
        if not a.only_dstct:
            mu, lv = twin_pred(Z, idx, mc=30); put("MC-Dropout twin", f, part, *to_y(mu, lv, idx))

# ---------------- conditional diffusion scenario generator
Xtr_f, miss_tr, _ = prep.causal_fill(D["X"][tr], D["adm"][tr])
Ztr = prep.norm(Xtr_f).astype(np.float32)
n = len(tr); ep_d = a.dif_epochs or int(np.clip(round(6000 / (n / 256)), 60, 150))
dif = Diffusion(Sst.shape[1], T=a.T, K=K, seed=a.seed)
t0 = time.time()
hist = dif.fit(Ztr, (~miss_tr).astype(np.float32), Sst[tr], ins_all[tr], D["nd"][tr], epochs=ep_d, seed=a.seed)
timing["Diffusion_train_s"] = time.time() - t0; timing["Diffusion_epochs"] = ep_d; timing["Diffusion_final_loss"] = hist[-1]
if a.save_models:
    torch.save({"dif": dif.net.state_dict(), "twin": twin.state_dict(), "T": a.T}, f"{a.out}/{tag}_models.pt")

viol = {}
scen_store = {}
for f in FRACS:
    for part, idx in (("val", va), ("test", te)):
        Xc, miss, cl = prep.causal_fill(D["X"][idx], D["adm"][idx])
        Zc = prep.norm(Xc).astype(np.float32)
        o = n_obs(D["nd"][idx], f); obs = (np.arange(K)[None] < o[:, None]).astype(np.float32)
        cmask = (obs[..., None] * (~miss)).astype(np.float32)  # genuine observations in the prefix only
        t1 = time.time()
        sc = dif.sample(Zc, cmask, obs, ins_all[idx], Sst[idx], np.full(len(idx), f, np.float32), M=a.M, steps=a.steps, seed=a.seed)
        t_samp = time.time() - t1
        phys = prep.denorm(sc)
        proj, vr = project_plausible(phys, prep)
        viol[(f, part)] = vr
        if f == 0.0 and part == "test":
            viol["real_test"] = rule_violations(prep.denorm(Zc))
        for name, arr in (("DST-CT (full)", prep.norm(proj)), ("DST-CT w/o semantic constraints", sc)):
            t2 = time.time()
            mus, sds = [], []
            for m in range(a.M):
                mu, lv = twin_pred(arr[m].astype(np.float32), idx); mus.append(mu); sds.append(np.exp(0.5 * lv))
            mus, sds = np.array(mus), np.array(sds)
            mu = mus.mean(0); var = (sds ** 2).mean(0) + mus.var(0)
            if part == "test" and name == "DST-CT (full)":
                timing.setdefault("DSTCT_infer_s", 0); timing["DSTCT_infer_s"] += t_samp + time.time() - t2
            put(name, f, part, *to_y(mu, np.log(var), idx))
            if name == "DST-CT (full)":
                put("DST-CT (point, no scenario spread)", f, part, *to_y(mu, np.log((sds ** 2).mean(0)), idx))
        if part == "test" and f == 0.5:
            scen_store = dict(proj=proj[:, :200].astype(np.float16), idx=idx[:200])

# ---------------- save
out = {"tag": tag, "ds": a.ds, "mode": a.mode, "year": a.year, "seed": a.seed, "cluster": a.cluster, "T": a.T,
       "steps": a.steps, "M": a.M, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
       "timing": timing, "viol": {(k if isinstance(k, str) else f"{k[0]}_{k[1]}"): v for k, v in viol.items()}, "diff_loss_hist": hist}
json.dump(out, open(f"{a.out}/{tag}.json", "w"))
arr = {"te": te, "va": va, "y_te": D["y"][te], "y_va": D["y"][va], "trend_te": trend[te], "rsd": rsd, "trend_va": trend[va]}
for m in P:
    for f in P[m]:
        for part in P[m][f]:
            mu, sd = P[m][f][part]
            arr[f"{m}|{f}|{part}|mu"] = mu; arr[f"{m}|{f}|{part}|sd"] = sd
np.savez_compressed(f"{a.out}/{tag}.npz", **arr)
if scen_store:
    np.savez_compressed(f"{a.out}/{tag}_scen.npz", **scen_store)
print(tag, json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in timing.items()}))
