"""DST-CT re-implementation on CY-Bench + leakage-free zero-day protocol + baselines."""
import numpy as np, math, time, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression

import os
torch.set_num_threads(int(os.environ.get("NTH", "2")))
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]  # forecast issue time as fraction of the season
DYN = ["ndvi", "fpar", "tmin", "tmax", "tavg", "prec", "rad", "et0", "vpd", "cwb", "ssm", "rsm"]
C = len(DYN)
VEG = [0, 1]


def load(ds):
    z = np.load(f"data/{ds}.npz", allow_pickle=True)
    D = {k: z[k] for k in z.files}
    D["ds"] = ds
    return D


# ---------------------------------------------------------------- history features (past yields only)
def history_features(D, allowed_years, use_own_history=True, train_mask=None):
    """For each sample (region r, year t): climatology, last-year yield and linear-trend
    prediction computed ONLY from labels of years < t that are in allowed_years.
    If use_own_history=False (unseen-region test), the region's own labels are never used
    and the features fall back to the mean over training regions for that year."""
    adm, yr, y = D["adm"], D["year"], D["y"]
    N = len(y)
    clim = np.zeros(N, np.float32); last = np.zeros(N, np.float32); trend = np.zeros(N, np.float32); mean3 = np.zeros(N, np.float32)
    ok = np.isin(yr, list(allowed_years))
    if train_mask is not None:
        ok = ok & train_mask
    lab = {}
    for i in np.where(ok)[0]:
        lab.setdefault(adm[i], []).append((yr[i], y[i]))
    # global per-year means over labelled training rows (for fallback)
    gyear = {}
    for i in np.where(ok)[0]:
        gyear.setdefault(yr[i], []).append(y[i])
    # older official statistics (before the first modelled season) are also past information
    import pandas as pd, os
    hp = f"data/{D.get('ds', '')}_yield_hist.csv"
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        h = h[h.harvest_year < yr.min()]
        allowed_regions = set(adm[train_mask]) if train_mask is not None else None
        for a_, t_, v_ in zip(h.adm_id.values, h.harvest_year.values, h["yield"].values):
            if allowed_regions is not None and a_ not in allowed_regions:
                continue
            lab.setdefault(a_, []).append((int(t_), float(v_)))
            gyear.setdefault(int(t_), []).append(float(v_))
    gy = {k: np.mean(v) for k, v in gyear.items()}
    gyrs = np.array(sorted(gy)); gvals = np.array([gy[k] for k in gyrs])
    for i in range(N):
        t = yr[i]
        past = [(a, b) for a, b in lab.get(adm[i], []) if a < t] if use_own_history else []
        if len(past) == 0:
            m = gyrs < t
            if m.sum() == 0:
                clim[i] = last[i] = trend[i] = mean3[i] = np.nan
                continue
            gx, gv = gyrs[m], gvals[m]
            clim[i] = gv.mean(); last[i] = gv[-1]; mean3[i] = gv[-3:].mean()
            trend[i] = np.polyval(np.polyfit(gx, gv, 1), t) if len(gx) >= 5 else gv.mean()
            continue
        px = np.array([a for a, _ in past], float); pv = np.array([b for _, b in past], float)
        o = np.argsort(px); px, pv = px[o], pv[o]
        clim[i] = pv.mean(); mean3[i] = pv[-3:].mean()
        last[i] = pv[np.argmax(px)] if (t - 1) in set(px.astype(int)) else pv.mean()
        if len(px) >= 5:
            c = np.polyfit(px, pv, 1)
            trend[i] = max(np.polyval(c, t), 0.0)
        else:
            trend[i] = pv.mean()
    return clim, last, trend, mean3


# anchor (baseline expectation) for the learned residual, chosen per dataset on pre-test seasons only
# (India: validation seasons 2008-2012; USA: 2014-2020): mean of the three previous seasons for India,
# linear trend over all previous seasons for the USA.
ANCHOR = {"wheat_IN": "mean3", "maize_IN": "mean3", "maize_US": "trend"}


# ---------------------------------------------------------------- fold preprocessing (fit on train only)
class Prep:
    def __init__(self, D, tr):
        X = D["X"][tr]
        self.mu = np.nanmean(X, axis=(0, 1)); self.sd = np.nanstd(X, axis=(0, 1)) + 1e-6
        # climatology per region x dekad-position (train years only) with global fallback
        self.gclim = np.nanmean(X, axis=0)  # [K,C]
        self.gclim = np.where(np.isnan(self.gclim), self.mu[None], self.gclim)
        self.rclim = {}
        for a in np.unique(D["adm"][tr]):
            m = D["adm"][tr] == a
            with np.errstate(all="ignore"):
                r = np.nanmean(X[m], axis=0)
            self.rclim[a] = np.where(np.isnan(r), self.gclim, r)
        S = D["S"][tr]
        self.smu = np.nanmean(S, 0); self.ssd = np.nanstd(S, 0) + 1e-6

    def clim_for(self, adm):
        return np.stack([self.rclim.get(a, self.gclim) for a in adm])

    def causal_fill(self, X, adm):
        """Fill gaps using only earlier dekads (last observation carried forward within
        the season); leading gaps use train-year climatology. Never looks ahead."""
        X = X.copy(); cl = self.clim_for(adm)
        miss = np.isnan(X)
        for j in range(X.shape[1]):
            if j == 0:
                X[:, 0] = np.where(miss[:, 0], cl[:, 0], X[:, 0])
            else:
                X[:, j] = np.where(miss[:, j], X[:, j - 1], X[:, j])
        return X, miss, cl

    def norm(self, X):
        return (X - self.mu) / self.sd

    def denorm(self, Z):
        return Z * self.sd + self.mu

    def nstat(self, S):
        return np.nan_to_num((S - self.smu) / self.ssd)


def n_obs(nd, frac):
    return np.round(nd * frac).astype(int)


def build_inputs(D, idx, prep, frac, future="clim"):
    """Observed part = dekads [0, o); future part filled with climatology (baselines)."""
    X, miss, cl = prep.causal_fill(D["X"][idx], D["adm"][idx])
    o = n_obs(D["nd"][idx], frac)
    K = X.shape[1]
    obs = (np.arange(K)[None, :] < o[:, None])
    Xf = np.where(obs[:, :, None], X, cl)
    return prep.norm(Xf).astype(np.float32), obs.astype(np.float32), miss


# ---------------------------------------------------------------- summary features for linear / tree models
def summary_features(Z, obs, inseason):
    K = Z.shape[1]
    w = [range(0, 6), range(6, 12), range(12, K)]
    f = [Z[:, list(r)].mean(1) for r in w]
    f.append(Z[:, :, VEG].max(1))
    f.append(obs.mean(1, keepdims=True))
    return np.concatenate(f, 1)


# ---------------------------------------------------------------- neural building blocks
class GaussHead(nn.Module):
    def __init__(self, h, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(p_drop), nn.Linear(h, 2))

    def forward(self, z):
        o = self.net(z)
        return o[:, 0], o[:, 1].clamp(-7, 5)


GROUPS = {"veg": [0, 1], "met": [2, 3, 4, 5, 6, 7, 8, 9], "soil": [10, 11]}


class TwinGRU(nn.Module):
    """Digital-twin state-space model.
    Modality fusion (eq. 6):  h_t = sum_k alpha_{t,k} W_k u_t^k,  alpha_t = softmax_k(q(s_{t-1}) . W_k u_t^k / sqrt(d))
    State update  (eq. 5):    s_t = GRU(s_{t-1}, h_t + e(z)),  s_0 = e(z) (static soil/site/yield-history embedding)
    Yield head    (eq. 7):    y ~ N(mu(s_K), sigma^2(s_K)) with s_K the state at the last in-season dekad.
    alpha_t are returned for interpretation (which modality drives the state at each growth stage)."""
    def __init__(self, c_in, s_in, h=64, extra_in=0, p_drop=0.1):
        super().__init__()
        self.sz = nn.Sequential(nn.Linear(s_in, h), nn.GELU())
        self.proj = nn.ModuleDict({k: nn.Linear(len(v) + 1, h) for k, v in GROUPS.items()})
        self.q = nn.Linear(h, h)
        self.cell = nn.GRUCell(h, h)
        self.drop = nn.Dropout(p_drop)
        self.head = GaussHead(h, p_drop)
        self.h = h
        self.last_alpha = None

    def forward(self, x, s, ins, extra=None):
        z = self.sz(s)
        B, K, _ = x.shape
        toks = torch.stack([self.proj[k](torch.cat([x[..., v], ins[..., None]], -1)) for k, v in GROUPS.items()], 2)
        st = z; alphas = []
        for t in range(K):
            q = self.q(st)[:, None]
            att = torch.softmax((q * toks[:, t]).sum(-1) / math.sqrt(self.h), -1)
            ht = (att[..., None] * toks[:, t]).sum(1)
            st_new = self.cell(ht + z, st)
            m = ins[:, t:t + 1]
            st = m * st_new + (1 - m) * st  # state frozen after the season ends
            alphas.append(att)
        self.last_alpha = torch.stack(alphas, 1)
        return self.head(self.drop(st))


class LSTMNet(nn.Module):
    def __init__(self, c_in, s_in, h=64):
        super().__init__()
        self.sz = nn.Sequential(nn.Linear(s_in, h), nn.GELU())
        self.inp = nn.Linear(c_in + 2, h)
        self.rnn = nn.LSTM(h, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = GaussHead(h)

    def forward(self, x, s, ins, extra):
        z = self.sz(s)
        u = self.inp(torch.cat([x, ins[..., None], extra], -1)) + z[:, None]
        out, _ = self.rnn(u)
        last = (ins.sum(1).long() - 1).clamp(min=0)
        return self.head(out[torch.arange(len(x)), last])


class TransNet(nn.Module):
    def __init__(self, c_in, s_in, h=64, K=18):
        super().__init__()
        self.sz = nn.Linear(s_in, h)
        self.inp = nn.Linear(c_in + 2, h)
        self.pos = nn.Parameter(torch.randn(1, K, h) * 0.02)
        el = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, 2)
        self.head = GaussHead(h)

    def forward(self, x, s, ins, extra):
        u = self.inp(torch.cat([x, ins[..., None], extra], -1)) + self.pos + self.sz(s)[:, None]
        out = self.enc(u, src_key_padding_mask=(ins == 0))
        w = ins[..., None]
        return self.head((out * w).sum(1) / w.sum(1).clamp(min=1))


def gauss_nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / lv.exp()).mean()


def train_net(net, batches_fn, n, epochs, lr=2e-3, bs=256, seed=0, val_fn=None, patience=6):
    """Generic trainer with early stopping on a validation NLL (val year only)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, bs):
            ib = perm[b:b + bs].numpy()
            loss = batches_fn(ib, ep)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if val_fn is not None:
            net.eval()
            with torch.no_grad():
                v = val_fn()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return ep + 1


# ---------------------------------------------------------------- conditional diffusion scenario generator
def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1)
    beta = (1 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999)
    return torch.cumprod(1 - beta, 0)


class Denoiser(nn.Module):
    """eps_theta(x_t, t | context). Context c = observed prefix of the dynamic
    predictors (masked), per-dekad observation mask, static vector z, and issue-time fraction.
    Conditioning is time-varying (per-dekad tokens) + static (added to every token)."""
    def __init__(self, s_in, h=64, K=18, layers=3):
        super().__init__()
        self.inp = nn.Linear(C + C + 2, h)
        self.pos = nn.Parameter(torch.randn(1, K, h) * 0.02)
        self.temb = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.sz = nn.Linear(s_in + 1, h)
        el = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Linear(h, C)
        self.h = h

    def tfeat(self, t, T):
        half = self.h // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half) / half)
        a = (t.float() / T)[:, None] * 1000 * fr[None]
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, xt, t, T, cobs, m, ins, s, frac):
        u = torch.cat([xt, cobs, m[..., None], ins[..., None]], -1)
        e = self.temb(self.tfeat(t, T)) + self.sz(torch.cat([s, frac[:, None]], 1))
        hdn = self.inp(u) + self.pos + e[:, None]
        return self.out(self.enc(hdn))


class Diffusion:
    def __init__(self, s_in, T=1000, K=18, h=64, seed=0):
        torch.manual_seed(seed)
        self.T = T; self.net = Denoiser(s_in, h, K); self.ab = cosine_schedule(T)

    def fit(self, Z, zmask, S, ins, nd, epochs=40, bs=256, lr=1e-3, seed=0, drop_ndvi=0.15, Zv=None, log=None):
        """Z: normalised complete trajectories (gaps filled), zmask: 1 where the source value
        existed (loss only on genuine observations). Random issue times and random NDVI
        drop-outs in the observed prefix teach both forecasting and cloud-gap repair."""
        g = np.random.default_rng(seed); torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        n = len(Z); Zt = torch.tensor(Z); Mt = torch.tensor(zmask, dtype=torch.float32)
        St = torch.tensor(S); It = torch.tensor(ins); K = Z.shape[1]
        steps = epochs * math.ceil(n / bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
        self.net.train(); hist = []
        for ep in range(epochs):
            perm = g.permutation(n); tot = 0
            for b in range(0, n, bs):
                ib = perm[b:b + bs]; B = len(ib)
                x0 = Zt[ib]
                frac = torch.tensor(g.choice(FRACS + [0.1, 0.4, 0.6, 0.9], B), dtype=torch.float32)
                o = torch.round(torch.tensor(nd[ib]) * frac).long()
                obs = (torch.arange(K)[None] < o[:, None]).float()
                cmask = obs[..., None].repeat(1, 1, C)
                drop = torch.tensor(g.random((B, K)) < drop_ndvi) & (obs > 0)
                cmask[..., 0] = cmask[..., 0] * (~drop).float()
                cmask = cmask * Mt[ib]
                t = torch.randint(0, self.T, (B,))
                eps = torch.randn_like(x0)
                a = self.ab[t][:, None, None]
                xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
                pred = self.net(xt, t, self.T, x0 * cmask, obs, It[ib], St[ib], frac)
                w = (1 - cmask) * Mt[ib] * It[ib][..., None]
                loss = ((pred - eps) ** 2 * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step(); sched.step()
                tot += loss.item() * B
            hist.append(tot / n)
        self.net.eval()
        return hist

    @torch.no_grad()
    def sample(self, cobs, cmask, obs, ins, S, frac, M=10, steps=50, eta=1.0, seed=0, bs=4096):
        """DDIM sampling with `steps` denoising steps (subsequence of the T training steps).
        Returns [M,N,K,C]; observed entries are re-imposed (assimilation of observations)."""
        torch.manual_seed(seed)
        N, K, _ = cobs.shape
        ts = torch.linspace(self.T - 1, 0, steps).long()
        out = np.zeros((M, N, K, C), np.float32)
        cobs_t = torch.tensor(cobs); cm = torch.tensor(cmask); ob = torch.tensor(obs); it = torch.tensor(ins)
        St = torch.tensor(S); fr = torch.tensor(frac, dtype=torch.float32)
        for m in range(M):
            for b in range(0, N, bs):
                sl = slice(b, b + bs); B = len(range(N)[sl])
                x = torch.randn(B, K, C)
                for i, t in enumerate(ts):
                    tt = torch.full((B,), int(t))
                    e = self.net(x, tt, self.T, cobs_t[sl] * cm[sl], ob[sl], it[sl], St[sl], fr[sl])
                    a = self.ab[t]
                    x0 = (x - (1 - a).sqrt() * e) / a.sqrt()
                    x0 = x0.clamp(-6, 6)
                    if i == len(ts) - 1:
                        x = x0; break
                    ap = self.ab[ts[i + 1]]
                    sig = eta * ((1 - ap) / (1 - a) * (1 - a / ap)).clamp(min=0).sqrt()
                    x = ap.sqrt() * x0 + (1 - ap - sig ** 2).clamp(min=0).sqrt() * e + sig * torch.randn_like(x)
                x = cm[sl] * cobs_t[sl] + (1 - cm[sl]) * x
                out[m, sl] = x.numpy()
        return out


def rule_violations(X):
    """Fraction of trajectories violating each agronomic/physical rule (with tolerances)."""
    i = {v: j for j, v in enumerate(DYN)}
    r = {}
    r["ndvi_range"] = ((X[..., i["ndvi"]] < -0.1) | (X[..., i["ndvi"]] > 1.0)).any(-1)
    r["fpar_range"] = ((X[..., i["fpar"]] < 0) | (X[..., i["fpar"]] > 100)).any(-1)
    r["non_negative"] = (X[..., [i[v] for v in ["prec", "rad", "et0", "vpd", "ssm", "rsm"]]] < 0).any((-1, -2))
    r["temp_order"] = ((X[..., i["tmin"]] > X[..., i["tavg"]] + 0.1) | (X[..., i["tavg"]] > X[..., i["tmax"]] + 0.1)).any(-1)
    r["water_balance"] = (np.abs(X[..., i["cwb"]] - (X[..., i["prec"]] - X[..., i["et0"]])) > 2.0).any(-1)
    anyv = np.zeros(r["ndvi_range"].shape, bool)
    for v in r.values():
        anyv |= v
    out = {k: float(v.mean()) for k, v in r.items()}
    out["any"] = float(anyv.mean())
    return out


def project_plausible(Xp, prep):
    """Semantic-conditioning constraints applied in physical units (agronomic/physical rules):
    NDVI in [-0.1,1]; FPAR in [0,100] %; non-negative prec, rad, ET0, VPD, soil moisture;
    tmin <= tavg <= tmax; climatic water balance identity cwb = prec - et0 (dekadal sums, mm).
    Returns the projected array and per-rule violation rates before projection."""
    X = Xp.copy()
    i = {v: j for j, v in enumerate(DYN)}
    before = rule_violations(X)
    X[..., i["ndvi"]] = X[..., i["ndvi"]].clip(-0.1, 1.0)
    X[..., i["fpar"]] = X[..., i["fpar"]].clip(0, 100)
    for v in ["prec", "rad", "et0", "vpd", "ssm", "rsm"]:
        X[..., i[v]] = X[..., i[v]].clip(min=0)
    tt = np.sort(X[..., [i["tmin"], i["tavg"], i["tmax"]]], -1)
    X[..., i["tmin"]], X[..., i["tavg"]], X[..., i["tmax"]] = tt[..., 0], tt[..., 1], tt[..., 2]
    X[..., i["cwb"]] = X[..., i["prec"]] - X[..., i["et0"]]
    return X, before


# ---------------------------------------------------------------- probabilistic metrics & calibration
def crps_gauss(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


class IsoCal:
    """Kuleshov et al. (2018) isotonic recalibration of the predictive CDF, fitted on the
    validation year only."""
    def fit(self, mu, sd, y):
        p = norm.cdf((y - mu) / sd)
        emp = np.array([(p <= q).mean() for q in p])
        self.ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, emp)
        grid = np.linspace(1e-4, 1 - 1e-4, 2001)
        self.grid, self.cal = grid, self.ir.predict(grid)
        return self

    def interval(self, mu, sd, level):
        lo_t, hi_t = (1 - level) / 2, (1 + level) / 2
        plo = self.grid[np.searchsorted(self.cal, lo_t, side="left").clip(0, len(self.grid) - 1)]
        phi = self.grid[np.searchsorted(self.cal, hi_t, side="left").clip(0, len(self.grid) - 1)]
        return mu + sd * norm.ppf(plo), mu + sd * norm.ppf(phi)
