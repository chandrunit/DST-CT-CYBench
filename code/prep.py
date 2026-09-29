"""Build season-aligned dekadal tensors from CY-Bench (v1.10) subsets.
Output: data/<ds>.npz with
  X   [N,K,C] physical units, NaN where the source has no observation
  S   [N,P]   static predictors
  y   [N]     yield (t ha-1)
  adm [N], year [N], and season meta (sos doy, season length days)
No imputation and no scaling happens here: both are fitted inside each
training fold later, so nothing from test years leaks into preprocessing.
"""
import numpy as np, pandas as pd, sys, json, os

K = 18  # dekads per season window (180 days from the start-of-season dekad)
DYN = ["ndvi", "fpar", "tmin", "tmax", "tavg", "prec", "rad", "et0", "vpd", "cwb", "ssm", "rsm"]
STAT = ["awc", "bulk_density", "drainage_class", "latitude", "longitude", "crop_area_percentage", "sos", "season_len"]


def build(ds, y0):
    d = f"data/{ds}"
    yl = pd.read_csv(f"{d}/yield.csv")[["adm_id", "harvest_year", "yield"]].dropna()
    yl = yl[(yl.harvest_year >= y0) & (yl["yield"] > 0)]
    cal = pd.read_csv(f"{d}/crop_calendar.csv")[["adm_id", "sos", "eos"]]
    soil = pd.read_csv(f"{d}/soil.csv").drop(columns=["crop_name"])
    loc = pd.read_csv(f"{d}/location.csv")[["adm_id", "latitude", "longitude"]]
    msk = pd.read_csv(f"{d}/crop_mask.csv")[["adm_id", "crop_area_percentage"]]
    st = cal.merge(soil, on="adm_id").merge(loc, on="adm_id").merge(msk, on="adm_id", how="left")
    st["season_len"] = (st.eos - st.sos) % 365
    st = st[(st.season_len >= 60)]
    # dynamic predictors indexed by absolute dekad
    dyn = None
    for v in ["ndvi", "fpar", "soil_moisture", "meteo"]:
        x = pd.read_csv(f"{d}/{v}_dekad.csv.gz")
        x["t"] = x.year.astype(int) * 36 + x.dek.astype(int)
        x = x.drop(columns=["year", "dek"])
        dyn = x if dyn is None else dyn.merge(x, on=["adm_id", "t"], how="outer")
    dyn["rad"] = dyn["rad"] / 1e6  # J m-2 day-1 -> MJ m-2 day-1
    dyn = dyn.set_index(["adm_id", "t"])[DYN].sort_index()
    samp = yl.merge(st, on="adm_id")
    sos_dek = np.floor(((samp.sos.values % 365)) / 365 * 36).astype(int)
    start_year = np.where(samp.sos.values <= samp.eos.values, samp.harvest_year.values, samp.harvest_year.values - 1)
    t0 = start_year * 36 + sos_dek
    N = len(samp)
    X = np.full((N, K, len(DYN)), np.nan, dtype=np.float32)
    groups = {a: g for a, g in dyn.groupby(level=0)}
    for i, (a, tt) in enumerate(zip(samp.adm_id.values, t0)):
        g = groups.get(a)
        if g is None:
            continue
        g = g.droplevel(0)
        idx = np.arange(tt, tt + K)
        sub = g.reindex(idx)
        X[i] = sub.values.astype(np.float32)
    S = samp[STAT].values.astype(np.float32)
    keep = ~np.isnan(X[:, :, 2]).all(1)  # need weather at least
    out = dict(X=X[keep], S=S[keep], y=samp["yield"].values[keep].astype(np.float32),
               adm=samp.adm_id.values[keep].astype(str), year=samp.harvest_year.values[keep].astype(int),
               nd=np.ceil(samp.season_len.values[keep] / 10).clip(1, K).astype(int))
    np.savez_compressed(f"data/{ds}.npz", **out, dyn=np.array(DYN), stat=np.array(STAT))
    miss = {v: float(np.isnan(out["X"][:, :, j]).mean()) for j, v in enumerate(DYN)}
    info = dict(ds=ds, N=int(keep.sum()), regions=int(len(set(out["adm"]))), years=[int(out["year"].min()), int(out["year"].max())],
                yield_mean=float(out["y"].mean()), yield_sd=float(out["y"].std()), season_dekads_median=float(np.median(out["nd"])),
                missing_fraction=miss)
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    infos = [build("wheat_IN", 2003), build("maize_IN", 2003), build("maize_US", 2003)]
    json.dump(infos, open("results/data_summary.json", "w"), indent=1)
