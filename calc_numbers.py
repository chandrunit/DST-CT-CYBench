"""Compute every number quoted in the text -> results/numbers.json (keys used as <<KEY>> placeholders)."""
import json, glob, numpy as np, pandas as pd
from tables_results import S, DSN

N = {}
dt = pd.read_csv("results/dataset_table.csv")
N["N_TOTAL"] = f"{int(dt.seasons.sum()):,}"
N["N_REGIONS"] = f"{int(dt.regions.sum()):,}"
N["MISS_NDVI_W"], N["MISS_NDVI_M"], N["MISS_NDVI_U"] = [f"{v:.1f}" for v in dt.ndvi_missing_pct]
N["N_FOLDS_TEMPORAL"] = "13 (five for each Indian dataset and three for the United States)"
yu = np.load("data/maize_US.npz", allow_pickle=True)["year"]
N["N_TRANSFER"] = f"{int((yu >= 2019).sum()):,}"
TAG = {"wheat_IN": "W", "maize_IN": "M", "maize_US": "U"}


def g(ds, m, f, col="RMSE", mode="temporal"):
    s = S(ds, mode)
    if m == "DEEP":
        m = [x for x in s.model.unique() if x.startswith("Deep ensemble")][0]
    r = s[(s.model == m) & np.isclose(s.frac, f)]
    return float(r.iloc[0][col]) if not r.empty else np.nan


for ds, t in TAG.items():
    for f, ft in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
        for m, mt in (("DST-CT (full)", "DST"), ("DST-CT (deterministic, no diffusion)", "DET"), ("LSTM", "LSTM"), ("Transformer", "TRF"),
                      ("Ridge", "RDG"), ("LightGBM", "LGB"), ("VI regression", "VI"), ("Linear trend", "TRD"), ("3-season mean", "M3"),
                      ("Persistence", "PER"), ("Climatology", "CLI"), ("DEEP", "ENS"), ("MC-Dropout twin", "MCD")):
            v = g(ds, m, f)
            N[f"{t}_{mt}_{ft}"] = f"{v:.3f}"
        for m, mt in (("DST-CT (full)", "DST"), ("DST-CT (deterministic, no diffusion)", "DET"), ("DEEP", "ENS"), ("LSTM", "LSTM"), ("MC-Dropout twin", "MCD"), ("Transformer", "TRF")):
            for col in ("CRPS", "PICP90", "PICP90_raw", "NLL", "PIW90"):
                v = g(ds, m, f, col)
                N[f"{t}_{mt}_{ft}_{col}"] = f"{v:.3f}" if col in ("CRPS", "NLL") else (f"{v:.2f}" if col == "PIW90" else f"{v:.1f}")
    N[f"{t}_DST_100_R2"] = f"{g(ds, 'DST-CT (full)', 1.0, 'R2'):.2f}"
    N[f"{t}_DST_100_NRMSE"] = f"{g(ds, 'DST-CT (full)', 1.0, 'nRMSE'):.1f}"
    s = S(ds)
    best = s[np.isclose(s.frac, 1.0)].sort_values("RMSE").iloc[0]
    N[f"{t}_BEST_100"] = best.model

# paired
for ds, t in TAG.items():
    p = pd.read_csv(f"results/paired_{ds}_temporal.csv")
    for f, ft in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
        for b, bt in (("Linear trend", "TRD"), ("3-season mean", "M3"), ("Ridge", "RDG"), ("LSTM", "LSTM"), ("LightGBM", "LGB"),
                      ("Transformer", "TRF"), ("DST-CT (deterministic, no diffusion)", "DET")):
            r = p[(p.baseline == b) & np.isclose(p.frac, f)]
            if r.empty:
                continue
            r = r.iloc[0]
            N[f"P_{t}_{bt}_{ft}"] = f"{r.d_rmse:+.3f} t ha^−1^ (95% CI {r.d_lo:+.3f} to {r.d_hi:+.3f}; Wilcoxon *p* {'< 0.001' if r.wilcoxon_p < 0.001 else '= ' + format(r.wilcoxon_p, '.3f')})"

# relative gains of the twin at end of season (USA)
for ds, t in TAG.items():
    d = g(ds, "DST-CT (full)", 1.0)
    for m, mt in (("LSTM", "LSTM"), ("Ridge", "RDG"), ("LightGBM", "LGB"), ("Transformer", "TRF"), ("Linear trend", "TRD")):
        N[f"{t}_GAIN_{mt}_100"] = f"{100 * (g(ds, m, 1.0) - d) / g(ds, m, 1.0):.0f}"

vals = [g("maize_US", m, 1.0) for m in ("LSTM", "Transformer", "LightGBM", "Ridge")]
N["U_BASE_MIN"], N["U_BASE_MAX"] = f"{min(vals):.2f}", f"{max(vals):.2f}"
for ds, t in (("wheat_IN", "W"), ("maize_US", "U")):
    try:
        for f, ft in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
            for m, mt in (("DST-CT (full)", "DST"), ("DST-CT (deterministic, no diffusion)", "DET"), ("Ridge", "RDG"), ("LightGBM", "LGB"), ("LSTM", "LSTM"),
                          ("Transformer", "TRF"), ("Climatology", "CLI"), ("Linear trend", "TRD")):
                N[f"SP_{t}_{mt}_{ft}"] = f"{g(ds, m, f, 'RMSE', 'spatial'):.3f}"
            N[f"SP_{t}_DST_{ft}_PICP90"] = f"{g(ds, 'DST-CT (full)', f, 'PICP90', 'spatial'):.0f}"
            N[f"SP_{t}_DST_{ft}_R2"] = f"{g(ds, 'DST-CT (full)', f, 'R2', 'spatial'):.2f}"
    except FileNotFoundError:
        pass
# timing from fold JSONs (CPU, one thread)
tim = []
for fn in glob.glob("preds/*_temporal_*_s*.json"):
    if "_T" in fn.split("/")[-1]:
        continue
    j = json.load(open(fn)); t = j["timing"]; t["ds"] = j["ds"]; t["n_test"] = j["n_test"]; t["n_train"] = j["n_train"]; tim.append(t)
T = pd.DataFrame(tim)
for ds, t in TAG.items():
    d = T[T.ds == ds]
    for k, kt in (("Diffusion_train_s", "DIFTR"), ("Twin_train_s", "TWTR"), ("LSTM_train_s", "LSTMTR"), ("Transformer_train_s", "TRFTR"), ("LightGBM_train_s", "LGBTR")):
        N[f"{t}_{kt}"] = f"{d[k].mean() / 60:.1f}"
    per = (d.DSTCT_infer_s / 5 / d.n_test * 1000)
    N[f"{t}_DST_LAT"] = f"{per.mean():.1f}"
    N[f"{t}_DET_LAT"] = f"{(d.Det_infer_s / 5 / d.n_test * 1000).mean():.2f}"
    N[f"{t}_LSTM_LAT"] = f"{(d.LSTM_infer_s / 5 / d.n_test * 1000).mean():.2f}"
    N[f"{t}_TRF_LAT"] = f"{(d.Transformer_infer_s / 5 / d.n_test * 1000).mean():.2f}"
    N[f"{t}_NTRAIN"] = f"{int(d.n_train.mean()):,}"
    # constraint violations before projection
vi = []
for fn in glob.glob("preds/*_temporal_*_s*.json"):
    if "_T" in fn.split("/")[-1]:
        continue
    j = json.load(open(fn))
    for k, v in j["viol"].items():
        if isinstance(v, dict):
            vi.append(dict(ds=j["ds"], key=k, **v))
V = pd.DataFrame(vi)
if not V.empty:
    gen = V[V.key != "real_test"].groupby("ds").mean(numeric_only=True)
    real = V[V.key == "real_test"].groupby("ds").mean(numeric_only=True)
    for ds, t in TAG.items():
        for c in ("any", "ndvi_range", "fpar_range", "non_negative", "temp_order", "water_balance"):
            N[f"VIOL_{t}_{c}"] = f"{100 * gen.loc[ds, c]:.1f}" if ds in gen.index else "–"
            N[f"VIOLREAL_{t}_{c}"] = f"{100 * real.loc[ds, c]:.1f}" if ds in real.index else "–"
    N["VIOL_TABLE"] = V.to_json()

# deep analyses
for tag, t in (("wheat_IN_temporal_2017_s0", "W"), ("maize_IN_temporal_2017_s0", "M"), ("maize_US_temporal_2023_s0", "U")):
    try:
        d = json.load(open(f"results/deep_{tag}.json"))
    except FileNotFoundError:
        continue
    if "gap" in d:
        for k, v in d["gap"].items():
            N[f"GAP_{t}_{k}"] = f"{v:.3f}" if isinstance(v, float) else str(v)
    if "attn_by_phase" in d:
        for ph, v in d["attn_by_phase"].items():
            for i, mo in enumerate(("veg", "met", "soil")):
                N[f"ATT_{t}_{ph}_{mo}"] = f"{v[i]:.2f}"
    if "perm_importance" in d:
        for k, v in d["perm_importance"].items():
            N[f"PERM_{t}_{k.replace('|', '_').replace('+', '_')}"] = f"{1000 * v:.0f}"
        N[f"PERM_{t}_base"] = f"{d['perm_base_rmse']:.3f}"
    if "n_params" in d:
        N["PARAMS_TWIN"] = f"{d['n_params']['twin']:,}"; N["PARAMS_DEN"] = f"{d['n_params']['denoiser']:,}"
    if "cases" in d:
        for i, cs in enumerate(d["cases"]):
            N[f"CASE_{t}{i}_adm"] = cs["adm"]; N[f"CASE_{t}{i}_y"] = f"{cs['y']:.2f}"; N[f"CASE_{t}{i}_anchor"] = f"{cs['trend']:.2f}"
            for fr in ("0.0", "0.5", "1.0"):
                N[f"CASE_{t}{i}_mu_{fr}"] = f"{cs['fc'][fr]['mu']:.2f}"; N[f"CASE_{t}{i}_sd_{fr}"] = f"{cs['fc'][fr]['sd']:.2f}"
                N[f"CASE_{t}{i}_det_{fr}"] = f"{cs['fc'][fr]['det']:.2f}"
# steps sensitivity
rows = []
for tag in ("wheat_IN_temporal_2017_s0", "wheat_IN_temporal_2017_s1", "wheat_IN_temporal_2017_s2", "maize_US_temporal_2023_s0"):
    try:
        d = json.load(open(f"results/deep_{tag}.json"))
    except FileNotFoundError:
        continue
    for r in d.get("steps", []):
        r = dict(r); r["ds"] = tag[:8]; rows.append(r)
if rows:
    St = pd.DataFrame(rows).groupby(["ds", "frac", "steps"]).mean(numeric_only=True).reset_index()
    St.to_csv("results/steps_sensitivity.csv", index=False)

# T sensitivity (training steps), wheat India 2017
from analyze import metrics
from lib import IsoCal
tr = []
for T_ in (100, 250, 1000):
    for s in (0, 1, 2):
        fn = f"preds/wheat_IN_temporal_2017_s{s}" + ("" if T_ == 1000 else f"_T{T_}") + ".npz"
        try:
            z = np.load(fn)
        except FileNotFoundError:
            continue
        for f in ("0.0", "0.5"):
            mu, sd = z[f"DST-CT (full)|{f}|test|mu"], z[f"DST-CT (full)|{f}|test|sd"]
            y = z["y_te"]
            from lib import crps_gauss
            tr.append(dict(T=T_, seed=s, frac=float(f), rmse=float(np.sqrt(np.mean((mu - y) ** 2))), crps=float(crps_gauss(mu, sd, y).mean())))
if tr:
    Tt = pd.DataFrame(tr).groupby(["T", "frac"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), crps=("crps", "mean"), n=("seed", "count")).reset_index()
    Tt.to_csv("results/T_sensitivity.csv", index=False)

fs = sorted(glob.glob("results/transfer_s*.json"))
if fs:
    d = pd.DataFrame(sum([json.load(open(f)) for f in fs], []))
    code = {"zero-shot": "ZS", "fine-tune (1 US year)": "FT1", "scratch (1 US year)": "SC1", "fine-tune (3 US years)": "FT3", "scratch (3 US years)": "SC3"}
    mc = {"DST-CT": "DST", "DST-CT (deterministic)": "DET", "LSTM": "LSTM", "Ridge": "RDG", "Linear trend (no training)": "TRD", "3-season mean (no training)": "M3"}
    g_ = d.groupby(["setting", "model", "frac"]).agg(rmse=("rmse", "mean"), r2123=("rmse_2021_2023", "mean")).reset_index()
    for _, r in g_.iterrows():
        if r.setting in code and r.model in mc:
            ft = {0.0: "0", 0.5: "50", 1.0: "100"}[r.frac]
            N[f"TR_{code[r.setting]}_{mc[r.model]}_{ft}"] = f"{r.rmse:.3f}"
            N[f"TR2123_{code[r.setting]}_{mc[r.model]}_{ft}"] = f"{r.r2123:.3f}"
json.dump(N, open("results/numbers.json", "w"), indent=0)
print(len(N), "numbers")
