"""Seed robustness (seeds 0,1,2) of DST-CT (full) vs Linear trend on CY-Bench wheat India, test years 2006-2017.
Outputs results/seeds_state.csv (state x issue x seed), results/seeds_state_year.csv, results/seeds_summary.xlsx."""
import numpy as np, pandas as pd, glob
from states import STATES

D = np.load("data/wheat_IN.npz", allow_pickle=True); adm = D["adm"]
FR = [0.0, 0.5, 1.0]
rows = []
for f in sorted(glob.glob("preds/wheat_IN_temporal_*_s[0-9].npz")):
    z = np.load(f); parts = f.split("_"); yr = int(parts[-2]); sd = int(parts[-1][1:-4]); te = z["te"]
    base = pd.DataFrame(dict(seed=sd, year=yr, adm=adm[te], state=[STATES[a[3:5]] for a in adm[te]], y=z["y_te"],
                             trend=z["Linear trend|1.0|test|mu"]))
    for fr in FR:
        base[f"dst_{fr}"] = z[f"DST-CT (full)|{fr}|test|mu"]
    rows.append(base)
P = pd.concat(rows, ignore_index=True)
P.to_csv("results/district_predictions_3seeds.csv", index=False)


def met(d, c):
    e = d[c] - d.y
    return pd.Series(dict(n=len(d), RMSE=np.sqrt((e ** 2).mean()), MAE=e.abs().mean(),
                          R2=1 - (e ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum(),
                          RMSE_trend=np.sqrt(((d.trend - d.y) ** 2).mean()), yield_mean=d.y.mean()))


out = []
for fr in FR:
    g = P.groupby(["seed", "state"]).apply(met, c=f"dst_{fr}", include_groups=False).reset_index(); g["issue"] = fr
    a = P.groupby("seed").apply(met, c=f"dst_{fr}", include_groups=False).reset_index(); a["state"] = "All India"; a["issue"] = fr
    out += [g, a]
S = pd.concat(out); S["better_vs_trend_pct"] = 100 * (1 - S.RMSE / S.RMSE_trend)
S.to_csv("results/seeds_state.csv", index=False)
agg = S.groupby(["issue", "state"]).agg(seeds=("seed", "nunique"), n=("n", "mean"), RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                                        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), R2=("R2", "mean"), R2_sd=("R2", "std"),
                                        RMSE_trend=("RMSE_trend", "mean"), better_pct=("better_vs_trend_pct", "mean"),
                                        better_pct_sd=("better_vs_trend_pct", "std"), better_pct_min=("better_vs_trend_pct", "min"),
                                        yield_mean=("yield_mean", "mean")).reset_index()
agg["nRMSE_pct"] = 100 * agg.RMSE / agg.yield_mean
# per year for key states
SY = []
for fr in FR:
    g = P.groupby(["seed", "state", "year"]).apply(met, c=f"dst_{fr}", include_groups=False).reset_index(); g["issue"] = fr; SY.append(g)
SY = pd.concat(SY); SY["dst_better"] = SY.RMSE < SY.RMSE_trend
SY.to_csv("results/seeds_state_year.csv", index=False)
wins = SY.groupby(["issue", "state", "seed"]).dst_better.agg(["sum", "size"]).reset_index()
with pd.ExcelWriter("results/seeds_summary.xlsx") as w:
    agg.round(4).to_excel(w, sheet_name="state_mean_over_seeds", index=False)
    S.round(4).to_excel(w, sheet_name="state_by_seed", index=False)
    wins.to_excel(w, sheet_name="years_dst_beats_trend", index=False)
    SY[SY.state.isin(["Odisha", "Uttarakhand", "Punjab"])].round(4).to_excel(w, sheet_name="key_states_by_year", index=False)
pd.set_option("display.width", 250)
print(P.groupby("seed").year.nunique())
for st in ["Odisha", "Uttarakhand", "Punjab", "All India"]:
    print(agg[agg.state == st].round(3).to_string())
    print(S[(S.state == st) & (S.issue.isin([0.0, 1.0]))][["issue", "seed", "RMSE", "MAE", "R2", "RMSE_trend", "better_vs_trend_pct"]].round(3).to_string())
print(wins[wins.state.isin(["Odisha", "Uttarakhand", "Punjab"])].to_string())
print(agg[agg.issue == 1.0].sort_values("better_pct", ascending=False)[["state", "n", "RMSE", "RMSE_trend", "better_pct", "better_pct_sd", "better_pct_min", "R2"]].round(3).to_string())
