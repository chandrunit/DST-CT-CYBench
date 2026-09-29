"""Rebuild the exact fold context used by run_fold.py (same splits, preprocessing, features)."""
import numpy as np, torch
from sklearn.cluster import KMeans
from lib import *


def fold_context(ds, year, mode="temporal", cluster=-1):
    D = load(ds); yr = D["year"]; N = len(yr)
    reg = np.ones(N, bool)
    if mode == "spatial":
        lat, lon = D["S"][:, 3], D["S"][:, 4]
        u, inv = np.unique(D["adm"], return_index=True)
        km = KMeans(5, n_init=10, random_state=0).fit(np.c_[lat[inv], lon[inv]])
        lab = dict(zip(u, km.labels_)); cl = np.array([lab[x] for x in D["adm"]])
        reg = cl != cluster
    tr = np.where((yr < year - 1) & reg)[0]
    va = np.where((yr == year - 1) & reg)[0]
    te = np.where((yr == year) & (reg if mode == "temporal" else ~reg))[0]
    if mode == "temporal":
        clim, last, trend, mean3 = history_features(D, set(yr[yr < year]))
    else:
        clim, last, trend, mean3 = history_features(D, set(yr[yr < year]), True, train_mask=reg)
        c2, l2, t2, m2 = history_features(D, set(yr[yr < year]), False, train_mask=reg)
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
    R = (res / rsd).astype(np.float32)
    K = D["X"].shape[1]
    ins_all = (np.arange(K)[None] < D["nd"][:, None]).astype(np.float32)
    return dict(D=D, tr=tr, va=va, te=te, prep=prep, Sst=Sst, trend=trend, rsd=rsd, R=R, ins=ins_all, K=K,
                clim=clim, last=last, H=H, lin_trend=lin_trend, mean3=mean3)


def load_models(path, s_in, K):
    ck = torch.load(path)
    twin = TwinGRU(C, s_in); twin.load_state_dict(ck["twin"]); twin.eval()
    dif = Diffusion(s_in, T=ck["T"], K=K); dif.net.load_state_dict(ck["dif"]); dif.net.eval()
    return twin, dif
