"""Revision round 2: experiments that test what the diffusion component is actually for.

One fold = (dataset, test year, seed), same leakage-free temporal protocol as run_fold.py
(train < Y-1, validation = Y-1 for early stopping / calibration only, test = Y).

E1  Scenario skill of the generator itself. For every issue time < 100 %, the not-yet-observed
    in-season dekads of all 12 predictors are forecast as an ensemble and scored with the sample
    CRPS against what was later observed (normalised units, genuine observations only).
    Competitors with the same ensemble size:
      - Climatology      : region climatology from training years (deterministic, CRPS = MAE)
      - Analog (random)  : M seasons drawn at random from the same region's training years
      - Analog (kNN)     : M training seasons nearest in (observed prefix, site) space
      - Diffusion        : M conditional DDIM samples (+ physical plausibility projection)
E2  Scenario-consistent digital twin. The original twin is trained on complete seasons only and at
    inference receives partially generated seasons (train/test mismatch). Here a twin with an explicit
    observation flag is trained on seasons whose future part is filled by the diffusion generator at a
    random issue time. Ablation with an identical twin trained on climatology-filled futures separates
    the effect of the scenarios from the effect of partial-season training.
Predictions are written in the same key format as run_fold.py so conformal.py / summary2.py can
merge them with the stored baseline predictions of the same fold.
"""
import argparse, json, time, os, numpy as np, torch, torch.nn as nn
from lib import *
from ctx import fold_context

p = argparse.ArgumentParser()
p.add_argument("--ds"); p.add_argument("--year", type=int); p.add_argument("--seed", type=int, default=0)
p.add_argument("--T", type=int, default=1000); p.add_argument("--steps", type=int, default=25)
p.add_argument("--M", type=int, default=20, help="scenarios per season at test time")
p.add_argument("--M_train", type=int, default=2, help="generated completions per training season and issue time")
p.add_argument("--M_e1", type=int, default=10, help="ensemble size in the scenario-skill test")
p.add_argument("--dif_epochs", type=int, default=0); p.add_argument("--twin_epochs", type=int, default=60)
p.add_argument("--pretrained", default="", help="E3: diffusion checkpoint pretrained on other countries (fine-tuned here)")
p.add_argument("--ft_epochs", type=int, default=30)
p.add_argument("--tag_suffix", default="")
p.add_argument("--eta", type=float, default=1.0, help="DDIM stochasticity (chosen on the validation year)")
p.add_argument("--dif_iters", type=int, default=0, help="total generator optimiser steps (0 = original epoch rule)")
p.add_argument("--repaint", type=int, default=0, help="1 = re-impose forward-noised observations at every sampling step")
p.add_argument("--e1_only", action="store_true", help="only train the generator and score E1 on validation and test")
p.add_argument("--out", default="preds2")
p.add_argument("--gen_ckpt", default="", help="if the file exists, load the trained generator from it; otherwise train and save it there")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)

def _tfeat(self, t, T):
    half = self.h // 2
    fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a_ = (t.float() / T)[:, None] * 1000 * fr[None]
    return torch.cat([a_.sin(), a_.cos()], 1)
Denoiser.tfeat = _tfeat
REPAINT = False
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionDev(Diffusion):
    """Same model, objective and DDIM sampler as lib.Diffusion; only runs on the GPU when present."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.net.to(DEV); self.ab = self.ab.to(DEV)

    def fit(self, Z, zmask, S, ins, nd, epochs=40, bs=256, lr=1e-3, seed=0, drop_ndvi=0.15, **_):
        g = np.random.default_rng(seed); torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        n = len(Z); Zt = torch.tensor(Z, device=DEV); Mt = torch.tensor(zmask, dtype=torch.float32, device=DEV)
        St = torch.tensor(S, device=DEV); It = torch.tensor(ins, device=DEV); ndt = torch.tensor(nd, device=DEV); K = Z.shape[1]
        steps = epochs * math.ceil(n / bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
        fr_choices = np.array(FRACS + [0.1, 0.4, 0.6, 0.9], np.float32)
        ar = torch.arange(K, device=DEV)
        self.net.train(); hist = []
        for ep in range(epochs):
            perm = torch.tensor(g.permutation(n), device=DEV); tot = 0
            for b in range(0, n, bs):
                ib = perm[b:b + bs]; B = len(ib)
                x0 = Zt[ib]
                frac = torch.tensor(g.choice(fr_choices, B), device=DEV)
                o = torch.round(ndt[ib] * frac).long()
                obs = (ar[None] < o[:, None]).float()
                cmask = obs[..., None].repeat(1, 1, C)
                drop = (torch.tensor(g.random((B, K)), device=DEV) < drop_ndvi) & (obs > 0)
                cmask[..., 0] = cmask[..., 0] * (~drop).float()
                cmask = cmask * Mt[ib]
                t = torch.randint(0, self.T, (B,), device=DEV)
                eps = torch.randn_like(x0)
                a_ = self.ab[t][:, None, None]
                xt = a_.sqrt() * x0 + (1 - a_).sqrt() * eps
                pred = self.net(xt, t, self.T, x0 * cmask, obs, It[ib], St[ib], frac)
                w = (1 - cmask) * Mt[ib] * It[ib][..., None]
                loss = ((pred - eps) ** 2 * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step(); sched.step()
                tot += loss.item() * B
            hist.append(tot / n)
        self.net.eval()
        return hist

    @torch.no_grad()
    def sample(self, cobs, cmask, obs, ins, S, frac, M=10, steps=50, eta=1.0, seed=0, bs=8192):
        torch.manual_seed(seed)
        N, K, _ = cobs.shape
        ts = torch.linspace(self.T - 1, 0, steps).long()
        out = np.zeros((M, N, K, C), np.float32)
        cobs_t = torch.tensor(cobs, device=DEV); cm = torch.tensor(cmask, device=DEV); ob = torch.tensor(obs, device=DEV)
        it = torch.tensor(ins, device=DEV); St = torch.tensor(S, device=DEV); fr = torch.tensor(frac, dtype=torch.float32, device=DEV)
        for m in range(M):
            for b in range(0, N, bs):
                sl = slice(b, b + bs); B = len(range(N)[sl])
                x = torch.randn(B, K, C, device=DEV)
                for i, t in enumerate(ts):
                    tt = torch.full((B,), int(t), device=DEV)
                    e = self.net(x, tt, self.T, cobs_t[sl] * cm[sl], ob[sl], it[sl], St[sl], fr[sl])
                    a_ = self.ab[t]
                    x0 = ((x - (1 - a_).sqrt() * e) / a_.sqrt()).clamp(-6, 6)
                    if i == len(ts) - 1:
                        x = x0; break
                    ap = self.ab[ts[i + 1]]
                    sig = eta * ((1 - ap) / (1 - a_) * (1 - a_ / ap)).clamp(min=0).sqrt()
                    x = ap.sqrt() * x0 + (1 - ap - sig ** 2).clamp(min=0).sqrt() * e + sig * torch.randn_like(x)
                    if REPAINT:
                        xk = ap.sqrt() * cobs_t[sl] + (1 - ap).sqrt() * torch.randn_like(x)
                        x = cm[sl] * xk + (1 - cm[sl]) * x
                x = cm[sl] * cobs_t[sl] + (1 - cm[sl]) * x
                out[m, sl] = x.cpu().numpy()
        return out
os.makedirs(a.out, exist_ok=True)
tag = f"{a.ds}_temporal_{a.year}_s{a.seed}{a.tag_suffix}"
t_start = time.time()

cx = fold_context(a.ds, a.year)
D, tr, va, te, prep = cx["D"], cx["tr"], cx["va"], cx["te"], cx["prep"]
Sst, trend, rsd, R, ins_all, K = cx["Sst"], cx["trend"], cx["rsd"], cx["R"], cx["ins"], cx["K"]
nd, adm = D["nd"], D["adm"]
T_ = lambda x: torch.tensor(np.asarray(x, np.float32))
St_all, ins_t, Rt = T_(Sst), T_(ins_all), T_(R)
FR_PART = [f for f in FRACS if f < 1.0]
timing = {}

def to_y(mu_r, sd_r, idx):
    return trend[idx] + mu_r * rsd, sd_r * rsd

P = {}
def put(m, f, part, mu, sd):
    P.setdefault(m, {}).setdefault(f, {})[part] = (np.asarray(mu, np.float64), np.maximum(np.asarray(sd, np.float64), 1e-3))

# ---------------------------------------------------------------- observed / filled tensors
def observed(idx, f):
    Xc, miss, cl = prep.causal_fill(D["X"][idx], adm[idx])
    Zc = prep.norm(Xc).astype(np.float32)
    o = n_obs(nd[idx], f)
    obs = (np.arange(K)[None] < o[:, None]).astype(np.float32)
    cmask = (obs[..., None] * (~miss)).astype(np.float32)
    Zclim = prep.norm(np.where(obs[..., None] > 0, Xc, cl)).astype(np.float32)
    return Zc, miss, obs, cmask, Zclim

# ---------------------------------------------------------------- diffusion generator (train or fine-tune)
Xtr_f, miss_tr, _ = prep.causal_fill(D["X"][tr], adm[tr])
Ztr = prep.norm(Xtr_f).astype(np.float32)
REPAINT = bool(a.repaint)
dif = DiffusionDev(Sst.shape[1], T=a.T, K=K, seed=a.seed)
t0 = time.time()
if a.gen_ckpt and os.path.exists(a.gen_ckpt):
    dif.net.load_state_dict(torch.load(a.gen_ckpt, map_location=DEV)); hist = []
    timing["generator_loaded_from"] = a.gen_ckpt
elif a.pretrained:
    ck = torch.load(a.pretrained, map_location="cpu")
    dif.net.load_state_dict(ck["dif"])
    hist = dif.fit(Ztr, (~miss_tr).astype(np.float32), Sst[tr], ins_all[tr], nd[tr], epochs=a.ft_epochs, lr=3e-4, seed=a.seed)
else:
    ep_d = a.dif_epochs or int(np.clip(round(6000 / (len(tr) / 256)), 60, 150))
    if a.dif_iters:
        ep_d = int(math.ceil(a.dif_iters / math.ceil(len(tr) / 256)))
    timing['diffusion_epochs'] = ep_d
    hist = dif.fit(Ztr, (~miss_tr).astype(np.float32), Sst[tr], ins_all[tr], nd[tr], epochs=ep_d, seed=a.seed)
timing["diffusion_train_s"] = time.time() - t0
if a.gen_ckpt and not os.path.exists(a.gen_ckpt):
    torch.save(dif.net.state_dict(), a.gen_ckpt)
GEN = {"s": 0.0, "n": 0}

def generate(idx, f, M, seed):
    Zc, miss, obs, cmask, _ = observed(idx, f)
    if DEV.type == "cuda": torch.cuda.synchronize()
    tg = time.time()
    sc = dif.sample(Zc, cmask, obs, ins_all[idx], Sst[idx], np.full(len(idx), f, np.float32), M=M, steps=a.steps, eta=a.eta, seed=seed)
    if DEV.type == "cuda": torch.cuda.synchronize()
    GEN["s"] += time.time() - tg; GEN["n"] += len(idx) * M
    proj, _ = project_plausible(prep.denorm(sc), prep)
    return prep.norm(proj).astype(np.float32)  # [M, n, K, C]

# ---------------------------------------------------------------- E1: scenario skill
def crps_ens(ens, y):
    """ens [M, ...], y [...] -> elementwise sample CRPS (fair version not needed: M equal for all)."""
    t1 = np.abs(ens - y[None]).mean(0)
    M = ens.shape[0]
    t2 = np.zeros_like(t1)
    for i in range(M):
        t2 += np.abs(ens[i][None] - ens).mean(0)
    return t1 - 0.5 * t2 / M

t0 = time.time()
Xtr_c, _, _ = prep.causal_fill(D["X"][tr], adm[tr]); Ztr_c = prep.norm(Xtr_c).astype(np.float32)
rng = np.random.default_rng(a.seed)
by_region = {}
for j, r in enumerate(adm[tr]):
    by_region.setdefault(r, []).append(j)
site_tr = Sst[tr][:, :8]  # normalised static site descriptors (soil, location, crop share, calendar)
def e1_eval(idx_e, seed_off):
    e1 = {}
    for f in FR_PART:
        Zc, miss, obs, cmask, Zclim = observed(idx_e, f)
        ev = (1 - obs)[..., None] * ins_all[idx_e][..., None] * (~miss)  # genuine, unobserved, in-season
        ev = np.broadcast_to(ev, Zc.shape).astype(bool)
        Me = a.M_e1
        dif_ens = generate(idx_e, f, Me, a.seed + 101 + seed_off)
        # analog random: same region, training years
        ar = np.empty((Me,) + Zc.shape, np.float32)
        for i, g in enumerate(idx_e):
            pool = by_region.get(adm[g], None)
            pick = rng.choice(pool, Me, replace=len(pool) < Me) if pool else rng.choice(len(tr), Me)
            ar[:, i] = Ztr_c[pick]
        # analog kNN: nearest training seasons on observed prefix (all variables) + site descriptors
        # prefix length differs per season, so distances are computed per prefix length o (vectorised on DEV)
        kn = np.empty_like(ar)
        Ztr_d = torch.tensor(Ztr_c, device=DEV); site_d = torch.tensor(site_tr, device=DEV)
        o_te = obs.sum(1).astype(int)
        for o in np.unique(o_te):
            rows = np.where(o_te == o)[0]
            for b in range(0, len(rows), 512):
                rr = rows[b:b + 512]
                q = torch.tensor(Zc[rr, :o], device=DEV)  # [b, o, C]
                if o > 0:
                    d_pref = torch.cdist(q.reshape(len(rr), -1), Ztr_d[:, :o].reshape(len(tr), -1)) ** 2 / (o * C)
                else:
                    d_pref = 0.0
                d_site = torch.cdist(torch.tensor(Sst[idx_e][rr, :8], device=DEV), site_d) ** 2
                nn_ = torch.topk(d_pref + 0.1 * d_site, Me, dim=1, largest=False).indices.cpu().numpy()
                kn[:, rr] = np.transpose(Ztr_c[nn_], (1, 0, 2, 3))
        res = {}
        for name, ens in (("Climatology", Zclim[None]), ("Analog (random)", ar), ("Analog (kNN)", kn), ("Diffusion", dif_ens)):
            cr = crps_ens(ens, Zc)
            res[name] = {"all": float(cr[ev].mean())}
            for c, v in enumerate(DYN):
                e = ev[..., c]
                res[name][v] = float(cr[..., c][e].mean()) if e.any() else None
            # per-season mean (for paired tests later)
            w = ev.sum((1, 2)); s = (cr * ev).sum((1, 2))
            res[name]["_per_season"] = np.where(w > 0, s / np.maximum(w, 1), np.nan).tolist()
        e1[str(f)] = res
    return e1
e1 = e1_eval(te, 0)
e1_val = e1_eval(va, 50)
timing["E1_s"] = time.time() - t0
timing["E1_sampling_s"] = GEN["s"]; timing["E1_scenarios"] = GEN["n"]
timing["ms_per_scenario"] = 1000 * GEN["s"] / max(GEN["n"], 1)
timing["device"] = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"
if a.e1_only:
    json.dump({"tag": tag, "eta": a.eta, "steps": a.steps, "E1": e1, "E1_val": e1_val, "timing": timing}, open(f"{a.out}/{tag}_e1_eta{a.eta}_it{a.dif_iters}_rp{a.repaint}.json", "w"))
    print("E1 val:", {f: round(v["Diffusion"]["all"], 3) for f, v in e1_val.items()}, "analog kNN val:", {f: round(v["Analog (kNN)"]["all"], 3) for f, v in e1_val.items()})
    raise SystemExit

# ---------------------------------------------------------------- E2: twins
class TwinObs(TwinGRU):
    """Same digital twin, plus an observation flag per dekad so the state update knows which
    inputs are measured and which are generated/climatological."""
    def __init__(self, c_in, s_in, h=64):
        super().__init__(c_in, s_in, h)
        self.proj = nn.ModuleDict({k: nn.Linear(len(v) + 2, h) for k, v in GROUPS.items()})

    def forward(self, x, s, ins, ob):
        z = self.sz(s); B, K_, _ = x.shape
        toks = torch.stack([self.proj[k](torch.cat([x[..., v], ins[..., None], ob[..., None]], -1)) for k, v in GROUPS.items()], 2)
        st = z
        for t in range(K_):
            q = self.q(st)[:, None]
            att = torch.softmax((q * toks[:, t]).sum(-1) / math.sqrt(self.h), -1)
            ht = (att[..., None] * toks[:, t]).sum(1)
            st_new = self.cell(ht + z, st)
            m = ins[:, t:t + 1]
            st = m * st_new + (1 - m) * st
        return self.head(self.drop(st))

t0 = time.time()
# training / validation inputs per issue time
def pack(idx, part, scen, Mgen, seed):
    """returns dict f -> (Z [Mgen, n, K, C], obs [n, K]); f = 1.0 uses the complete observed season."""
    out = {}
    for k, f in enumerate(FRACS):
        Zc, miss, obs, cmask, Zclim = observed(idx, f)
        if f == 1.0:
            out[f] = (Zc[None], obs)
        elif scen:
            out[f] = (generate(idx, f, Mgen, seed + 1000 * k), obs)
        else:
            out[f] = (Zclim[None], obs)
    return out

data = {}
for scen in (True, False):
    data[(scen, "tr")] = pack(tr, "tr", scen, a.M_train, a.seed + 7)
    data[(scen, "val")] = pack(va, "val", scen, 1, a.seed + 8)
timing["E2_generate_train_s"] = time.time() - t0

def train_twin(scen, seed):
    torch.manual_seed(seed)
    net = TwinObs(C, Sst.shape[1])
    Dtr = {f: (T_(z), T_(o)) for f, (z, o) in data[(scen, "tr")].items()}
    Dva = {f: (T_(z[0]), T_(o)) for f, (z, o) in data[(scen, "val")].items()}
    g = np.random.default_rng(seed)
    def fn(ib, ep):
        fs = g.integers(0, len(FRACS), len(ib))
        xs, os_ = [], []
        for k, i in zip(fs, ib):
            z, o = Dtr[FRACS[k]]
            xs.append(z[g.integers(0, z.shape[0])][i]); os_.append(o[i])
        gi = tr[ib]
        mu, lv = net(torch.stack(xs), St_all[gi], ins_t[gi], torch.stack(os_))
        return gauss_nll(mu, lv, Rt[gi])
    def vfn():
        return float(np.mean([gauss_nll(*net(Dva[f][0], St_all[va], ins_t[va], Dva[f][1]), Rt[va]).item() for f in FRACS]))
    train_net(net, fn, len(tr), a.twin_epochs, seed=seed, val_fn=vfn)
    return net

t0 = time.time()
twin_s = train_twin(True, a.seed)
twin_c = train_twin(False, a.seed)
timing["E2_twins_train_s"] = time.time() - t0

@torch.no_grad()
def pred_mix(net, Zs, idx, obs):
    mus, sds = [], []
    for m in range(Zs.shape[0]):
        mu, lv = net(T_(Zs[m]), St_all[idx], ins_t[idx], T_(obs))
        mus.append(mu.numpy()); sds.append(np.exp(0.5 * lv.numpy()))
    mus, sds = np.array(mus), np.array(sds)
    return mus.mean(0), np.sqrt((sds ** 2).mean(0) + mus.var(0)), np.sqrt((sds ** 2).mean(0))

t0 = time.time()
spread = {}
for f in FRACS:
    for part, idx in (("val", va), ("test", te)):
        Zc, miss, obs, cmask, Zclim = observed(idx, f)
        mu, sd, _ = pred_mix(twin_c, Zclim[None] if f < 1 else Zc[None], idx, obs)
        put("Twin, partial-season training (clim future)", f, part, *to_y(mu, sd, idx))
        Zs = generate(idx, f, a.M, a.seed + 500) if f < 1 else Zc[None]
        mu, sd, sd_in = pred_mix(twin_s, Zs, idx, obs)
        put("DST-CT v2 (scenario-consistent twin)", f, part, *to_y(mu, sd, idx))
        put("DST-CT v2, no scenario spread", f, part, *to_y(mu, sd_in, idx))
        if part == "test":
            spread[str(f)] = float(np.mean(np.sqrt(np.maximum(sd ** 2 - sd_in ** 2, 0)) / sd))
timing["E2_predict_s"] = time.time() - t0
timing["total_s"] = time.time() - t_start

# ---------------------------------------------------------------- save
json.dump({"tag": tag, "ds": a.ds, "year": a.year, "seed": a.seed, "steps": a.steps, "M": a.M, "pretrained": a.pretrained,
           "E1": e1, "E1_val": e1_val, "eta": a.eta, "dif_iters": a.dif_iters, "repaint": a.repaint, "scenario_share_of_sd": spread, "timing": timing, "diff_loss_hist": hist},
          open(f"{a.out}/{tag}.json", "w"))
arr = {"te": te, "va": va, "y_te": D["y"][te], "y_va": D["y"][va], "adm_te": adm[te]}
for m in P:
    for f in P[m]:
        for part in P[m][f]:
            mu, sd = P[m][f][part]
            arr[f"{m}|{f}|{part}|mu"] = mu; arr[f"{m}|{f}|{part}|sd"] = sd
np.savez_compressed(f"{a.out}/{tag}.npz", **arr)
print(tag, json.dumps({k: round(v, 1) for k, v in timing.items()}))
for f in FR_PART:
    print(f"E1 frac={f}: " + ", ".join(f"{k} {v['all']:.3f}" for k, v in e1[str(f)].items()))
