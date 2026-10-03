"""Domain transfer maize India (CY-Bench maize/IN) -> maize USA (CY-Bench maize/US).
Settings (test = US harvest years 2019-2023, all counties):
  zero-shot       : models trained on India only; no US labels and no parameter updates.
                    US predictor climatology (unlabelled) and US yield history (past years) are used as inputs.
  fine-tune k     : India models further trained on k most recent US years before 2019 (k=1: 2018; k=3: 2016-2018).
  scratch k       : same k US years, trained from random initialisation (isolates the benefit of pre-training).
  fully retrained : in-domain expanding-window US models (taken from the main temporal experiment)."""
import argparse, json, time, numpy as np, torch
from sklearn.linear_model import Ridge
from ctx import *

p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--quick", action="store_true"); a = p.parse_args()
Q = a.quick
torch.manual_seed(a.seed); np.random.seed(a.seed)
FR = [0.0, 0.5, 1.0]
src = fold_context("maize_IN", 2018)  # train 2003-2016, val 2017
Ds, prep, Sst_s, K = src["D"], src["prep"], src["Sst"], src["K"]
ymu, ysd, rsd = Ds["y"][src["tr"]].mean(), Ds["y"][src["tr"]].std(), src["rsd"]
Du = load("maize_US"); yu = Du["year"]
clim_u, last_u, lintrend_u, mean3_u = history_features(Du, set(yu[yu < 2024]))
trend_u = mean3_u  # same anchor as the Indian source models (mean of the three previous seasons)
Hu = np.c_[(clim_u - ymu) / ysd, (last_u - ymu) / ysd, (lintrend_u - ymu) / ysd, (mean3_u - ymu) / ysd, (yu - 2010) / 10.0].astype(np.float32)
Sst_u = np.c_[prep.nstat(Du["S"]), Hu].astype(np.float32)
Ru = ((Du["y"] - trend_u) / rsd).astype(np.float32)
ins_u = (np.arange(K)[None] < Du["nd"][:, None]).astype(np.float32)
fill_u = Prep(Du, np.where(yu < 2019)[0])  # unlabelled US predictor climatology (for gap/future filling only)
te_u = np.where(yu >= 2019)[0]
if Q: te_u = te_u[::40]
Tt = lambda x: torch.tensor(np.asarray(x, np.float32))


def inputs_u(idx, f):
    X, miss, cl = fill_u.causal_fill(Du["X"][idx], Du["adm"][idx])
    o = n_obs(Du["nd"][idx], f); obs = (np.arange(K)[None] < o[:, None])
    Xf = np.where(obs[:, :, None], X, cl)
    return prep.norm(Xf).astype(np.float32), obs.astype(np.float32), prep.norm(X).astype(np.float32), miss


def to_y(mu, sd, idx):
    return trend_u[idx] + mu * rsd, sd * rsd


def train_twin(twin, Z, S, ins, R, epochs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(twin.parameters(), lr=lr, weight_decay=1e-4)
    Z, S, ins, R = Tt(Z), Tt(S), Tt(ins), Tt(R)
    for ep in range(epochs):
        twin.train(); perm = torch.randperm(len(Z), generator=g)
        for b in range(0, len(Z), 256):
            ib = perm[b:b + 256]; mu, lv = twin(Z[ib], S[ib], ins[ib]); loss = gauss_nll(mu, lv, R[ib])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(twin.parameters(), 1.0); opt.step()
    twin.eval(); return twin


def train_lstm(net, Zs, Os, S, ins, R, epochs, lr, seed):
    g = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(S)
    for ep in range(epochs):
        net.train(); perm = g.permutation(n)
        for b in range(0, n, 256):
            ib = perm[b:b + 256]; k = g.integers(0, len(FRACS), len(ib))
            x = torch.stack([Zs[FRACS[j]][i] for j, i in zip(k, ib)]); ob = torch.stack([Os[FRACS[j]][i] for j, i in zip(k, ib)])
            mu, lv = net(x, S[ib], ins[ib], ob[..., None]); loss = gauss_nll(mu, lv, R[ib])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval(); return net


# ---------------- source training (India only)
tr, va = src["tr"], src["va"]
Zs_tr = {f: build_inputs(Ds, tr, prep, f)[:2] for f in FRACS}
Zs_va = {f: build_inputs(Ds, va, prep, f)[:2] for f in FRACS}
torch.manual_seed(a.seed)
twin = TwinGRU(C, Sst_s.shape[1])
def tfn(ib, ep):
    gi = tr[ib]; mu, lv = twin(Tt(Zs_tr[1.0][0][ib]), Tt(Sst_s[gi]), Tt(src["ins"][gi])); return gauss_nll(mu, lv, Tt(src["R"][gi]))
def tval():
    mu, lv = twin(Tt(Zs_va[1.0][0]), Tt(Sst_s[va]), Tt(src["ins"][va])); return gauss_nll(mu, lv, Tt(src["R"][va])).item()
train_net(twin, tfn, len(tr), 2 if Q else 60, seed=a.seed, val_fn=tval)
lstm = LSTMNet(C, Sst_s.shape[1])
rng = np.random.default_rng(a.seed)
def lfn(ib, ep):
    k = rng.integers(0, len(FRACS), len(ib)); gi = tr[ib]
    x = torch.stack([Tt(Zs_tr[FRACS[j]][0][i]) for j, i in zip(k, ib)]); ob = torch.stack([Tt(Zs_tr[FRACS[j]][1][i]) for j, i in zip(k, ib)])
    mu, lv = lstm(x, Tt(Sst_s[gi]), Tt(src["ins"][gi]), ob[..., None]); return gauss_nll(mu, lv, Tt(src["R"][gi]))
def lval():
    t = 0
    for f in FRACS:
        mu, lv = lstm(Tt(Zs_va[f][0]), Tt(Sst_s[va]), Tt(src["ins"][va]), Tt(Zs_va[f][1])[..., None]); t += gauss_nll(mu, lv, Tt(src["R"][va])).item()
    return t / len(FRACS)
train_net(lstm, lfn, len(tr), 2 if Q else 60, seed=a.seed, val_fn=lval)
Xs, miss_s, _ = prep.causal_fill(Ds["X"][tr], Ds["adm"][tr])
dif = Diffusion(Sst_s.shape[1], T=1000, K=K, seed=a.seed)
dif.fit(prep.norm(Xs).astype(np.float32), (~miss_s).astype(np.float32), Sst_s[tr], src["ins"][tr], Ds["nd"][tr],
        epochs=2 if Q else int(np.clip(round(6000 / (len(tr) / 256)), 60, 150)), seed=a.seed)
def ridge_feats(Z, ob, idx, S):
    return np.c_[summary_features(Z, ob, None), S[idx]]
rdg = Ridge(1.0).fit(np.concatenate([ridge_feats(*Zs_tr[f], tr, Sst_s) for f in FRACS]), np.concatenate([src["R"][tr]] * len(FRACS)))

import copy
state0 = dict(twin=copy.deepcopy(twin.state_dict()), lstm=copy.deepcopy(lstm.state_dict()), dif=copy.deepcopy(dif.net.state_dict()))

res = []
def evaluate(setting, twin, lstm, dif, rdg):
    for f in FR:
        Z, ob, Zc, miss = inputs_u(te_u, f)
        y = Du["y"][te_u]
        with torch.no_grad():
            mu, lv = twin(Tt(Z), Tt(Sst_u[te_u]), Tt(ins_u[te_u])); m_det, s_det = to_y(mu.numpy(), np.exp(0.5 * lv.numpy()), te_u)
            mu, lv = lstm(Tt(Z), Tt(Sst_u[te_u]), Tt(ins_u[te_u]), Tt(ob)[..., None]); m_l, s_l = to_y(mu.numpy(), np.exp(0.5 * lv.numpy()), te_u)
        m_r = trend_u[te_u] + rdg.predict(ridge_feats(Z, ob, te_u, Sst_u)) * rsd
        cmask = (ob[..., None] * (~miss)).astype(np.float32)
        sc = dif.sample(Zc, cmask, ob, ins_u[te_u], Sst_u[te_u], np.full(len(te_u), f, np.float32), M=2 if Q else 10, steps=5 if Q else 50, seed=a.seed)
        proj, _ = project_plausible(prep.denorm(sc), prep); Zs_ = prep.norm(proj)
        mus, sds = [], []
        with torch.no_grad():
            for m in range(sc.shape[0]):
                mu, lv = twin(Tt(Zs_[m]), Tt(Sst_u[te_u]), Tt(ins_u[te_u])); mus.append(mu.numpy()); sds.append(np.exp(0.5 * lv.numpy()))
        mus, sds = np.array(mus), np.array(sds)
        m_d, s_d = to_y(mus.mean(0), np.sqrt((sds ** 2).mean(0) + mus.var(0)), te_u)
        for name, mu_, sd_ in (("DST-CT", m_d, s_d), ("DST-CT (deterministic)", m_det, s_det), ("LSTM", m_l, s_l), ("Ridge", m_r, None)):
            r = dict(setting=setting, model=name, frac=f, seed=a.seed, rmse=float(np.sqrt(np.mean((mu_ - y) ** 2))),
                     mae=float(np.mean(np.abs(mu_ - y))), r2=float(1 - np.sum((mu_ - y) ** 2) / np.sum((y - y.mean()) ** 2)))
            sub = yu[te_u] >= 2021
            r["rmse_2021_2023"] = float(np.sqrt(np.mean((mu_[sub] - y[sub]) ** 2)))
            if sd_ is not None:
                r["crps"] = float(crps_gauss(mu_, sd_, y).mean()); r["cov90_raw"] = float(np.mean(np.abs(y - mu_) <= 1.645 * sd_))
            res.append(r)
        for nm, yt in (("Linear trend (no training)", lintrend_u[te_u]), ("3-season mean (no training)", mean3_u[te_u])):
            sub = yu[te_u] >= 2021
            res.append(dict(setting=setting, model=nm, frac=f, seed=a.seed, rmse=float(np.sqrt(np.mean((yt - y) ** 2))),
                            rmse_2021_2023=float(np.sqrt(np.mean((yt[sub] - y[sub]) ** 2))),
                            mae=float(np.mean(np.abs(yt - y))), r2=float(1 - np.sum((yt - y) ** 2) / np.sum((y - y.mean()) ** 2))))
        print(setting, f, [(r["model"], round(r["rmse"], 3)) for r in res[-6:]], flush=True)

t0 = time.time(); evaluate("zero-shot", twin, lstm, dif, rdg)
for k, yrs in ((1, [2018]), (3, [2016, 2017, 2018])):
    idx = np.where(np.isin(yu, yrs))[0]
    Zk = {f: inputs_u(idx, f) for f in FRACS}
    Xk, missk, _ = fill_u.causal_fill(Du["X"][idx], Du["adm"][idx])
    for mode in ("fine-tune", "scratch"):
        torch.manual_seed(a.seed)
        tw = TwinGRU(C, Sst_s.shape[1]); ls = LSTMNet(C, Sst_s.shape[1]); df = Diffusion(Sst_s.shape[1], T=1000, K=K, seed=a.seed)
        if mode == "fine-tune":
            tw.load_state_dict(state0["twin"]); ls.load_state_dict(state0["lstm"]); df.net.load_state_dict(state0["dif"]); lr_n, lr_d, ep_n, ep_d = 5e-4, 3e-4, (1 if Q else 15), (1 if Q else 20)
        else:
            lr_n, lr_d, ep_n, ep_d = 2e-3, 1e-3, (1 if Q else 30), (1 if Q else int(np.clip(round(6000 / (len(idx) / 256)), 60, 150)))
        train_twin(tw, Zk[1.0][0], Sst_u[idx], ins_u[idx], Ru[idx], ep_n, lr_n, a.seed)
        train_lstm(ls, {f: Tt(Zk[f][0]) for f in FRACS}, {f: Tt(Zk[f][1]) for f in FRACS}, Tt(Sst_u[idx]), Tt(ins_u[idx]), Tt(Ru[idx]), ep_n, lr_n, a.seed)
        df.fit(prep.norm(Xk).astype(np.float32), (~missk).astype(np.float32), Sst_u[idx], ins_u[idx], Du["nd"][idx], epochs=ep_d, lr=lr_d, seed=a.seed)
        rg = Ridge(1.0).fit(np.concatenate([ridge_feats(Zk[f][0], Zk[f][1], idx, Sst_u) for f in FRACS]), np.concatenate([Ru[idx]] * len(FRACS)))
        evaluate(f"{mode} ({k} US year{'s' if k > 1 else ''})", tw, ls, df, rg)
json.dump(res, open(f"results/transfer_s{a.seed}{'_quick' if Q else ''}.json", "w"))
print("done", time.time() - t0)
