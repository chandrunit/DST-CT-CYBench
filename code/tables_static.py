"""Static (non-result) tables for the manuscript, as pandoc pipe tables."""
import pandas as pd


def pipe(df, caption, hl=True):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:" if pd.api.types.is_numeric_dtype(df[c]) else ":--" for c in cols[1:]]) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    cap = f"[{caption}]{{.mark}}" if hl else caption
    return cap + "\n\n" + "\n".join(lines) + "\n"


def t_datasets():
    d = pd.read_csv("results/dataset_table.csv")
    rows = [
        ("Administrative level", ["district", "district", "county"]),
        ("Regions", list(d.regions)),
        ("Region-seasons", [f"{v:,}" for v in d.seasons]),
        ("Harvest years", list(d.years)),
        ("Seasons per region (median)", list(d.seasons_per_region_median)),
        ("Yield mean ± s.d. (t ha^−1^)", [f"{m:.2f} ± {s:.2f}" for m, s in zip(d.yield_mean, d.yield_sd)]),
        ("Yield range (t ha^−1^)", [f"{a:.2f}–{b:.2f}" for a, b in zip(d.yield_min, d.yield_max)]),
        ("Season length (dekads, median)", list(d.season_dekads_median)),
        ("In-season NDVI missing (%)", list(d.ndvi_missing_pct)),
        ("In-season soil moisture missing (%)", list(d.soilmoist_missing_pct)),
        ("Latitude range (°N)", list(d.lat_range)),
        ("Yield statistics source", ["ICRISAT District Level Database", "ICRISAT District Level Database", "USDA-NASS"]),
    ]
    df = pd.DataFrame({"": [r[0] for r in rows], "Wheat, India": [r[1][0] for r in rows], "Maize, India": [r[1][1] for r in rows],
                       "Maize, USA": [r[1][2] for r in rows]})
    return pipe(df, "Table 1. Datasets used (CY-Bench v1.10).")


def t_predictors():
    rows = [
        ("NDVI", "MOD09CMG, MODIS Terra (Vermote 2015)", "0.05°, daily → ~weekly", "dekad mean", "–"),
        ("FPAR", "MODIS/VIIRS FPAR (EC-JRC 2024)", "500 m, dekadal", "dekad value", "%"),
        ("T~min~, T~max~, T~mean~", "AgERA5 (Boogaard *et al.* 2020)", "0.1°, daily", "dekad mean", "°C"),
        ("Precipitation P", "AgERA5", "0.1°, daily", "dekad sum", "mm"),
        ("Solar radiation", "AgERA5", "0.1°, daily", "dekad mean", "MJ m^−2^ d^−1^"),
        ("Reference evapotranspiration ET~0~", "AgERA5", "0.1°, daily", "dekad sum", "mm"),
        ("Vapour-pressure deficit", "AgERA5", "0.1°, daily", "dekad mean", "hPa"),
        ("Climatic water balance CWB = P − ET~0~", "AgERA5", "0.1°, daily", "dekad sum", "mm"),
        ("Surface / root-zone soil moisture", "GLDAS (Rodell *et al.* 2004)", "0.25°, daily", "dekad mean", "kg m^−2^"),
        ("Available water capacity, bulk density, drainage class", "WISE30sec (Batjes 2016)", "static", "–", "cm m^−1^, kg dm^−3^, class"),
        ("Start / end of season", "WorldCereal calendars (Franch *et al.* 2022)", "static", "–", "day of year"),
        ("Crop-area fraction, centroid", "WorldCereal; CY-Bench", "static", "–", "%, °"),
        ("Yield history (mean, previous season, linear trend)", "Derived from past yields only", "per season", "–", "t ha^−1^"),
    ]
    df = pd.DataFrame(rows, columns=["Predictor", "Source", "Native resolution", "Dekadal aggregation", "Unit"])
    return pipe(df, "Table 2. Predictors, sources and aggregation. All gridded variables are cropland-weighted means over each administrative region (CY-Bench).")


def t_notation():
    rows = [
        ("*r*, *y*, *t*", "region, harvest year, dekad since start of season (*t* = 1, …, *K*, *K* = 18)"),
        ("*L*~r~", "number of in-season dekads of region *r*"),
        ("φ, τ", "issue fraction of the season and issue dekad, τ = round(φ*L*~r~)"),
        ("**u**~t~, **u**~t~^(k)^", "dynamic predictor vector (12 variables) and its modality-*k* block (veg, met, soil)"),
        ("**z**", "static context (soil, location, calendar, crop area, yield history)"),
        ("**X**~0~, **U**", "standardized and physical trajectory of a season (*K* × 12)"),
        ("**B**, **O**, **ι**", "masks: observed at issue time; present in source data; in-season dekads"),
        ("*s*, *S*~T~, ᾱ~s~", "diffusion step, number of training steps (1000), cumulative noise schedule"),
        ("*S*, η, *M*", "sampling steps (50), DDIM stochasticity (1), number of scenarios (10)"),
        ("**ε**~θ~", "noise-prediction network (conditional denoiser)"),
        ("Π", "projection onto agronomically and physically plausible trajectories (table 4)"),
        ("**s**~t~", "latent crop state of the digital twin (64-dimensional)"),
        ("α~t,k~", "modality attention weight at dekad *t*"),
        ("μ~ψ~, σ~ψ~", "mean and standard deviation of the forecasting head (standardized residual)"),
        ("*Ŷ*^trend^, σ~R~", "trend prediction from past yields and training residual standard deviation"),
        ("μ̄, σ̄", "scenario-mixture mean and standard deviation of yield (t ha^−1^)"),
        ("𝓡", "isotonic recalibration map fitted on the validation season"),
    ]
    return pipe(pd.DataFrame(rows, columns=["Symbol", "Meaning"]), "Table 3. Notation.")


def t_rules():
    rows = [
        ("NDVI range", "−0.1 ≤ NDVI ≤ 1", "clip"),
        ("FPAR range", "0 ≤ FPAR ≤ 100 %", "clip"),
        ("Non-negativity", "P, radiation, ET~0~, VPD, soil moisture ≥ 0", "clip at 0"),
        ("Temperature order", "T~min~ ≤ T~mean~ ≤ T~max~ (tolerance 0.1 °C)", "sort the three values"),
        ("Water balance", "CWB = P − ET~0~ (tolerance 2 mm per dekad)", "recompute CWB"),
    ]
    return pipe(pd.DataFrame(rows, columns=["Rule", "Constraint", "Correction"]), "Table 4. Semantic (agronomic and physical) constraints enforced on generated trajectories.")


def t_splits():
    rows = []
    for ds, years in (("Wheat, India", range(2013, 2018)), ("Maize, India", range(2013, 2018)), ("Maize, USA", range(2021, 2024))):
        for Y in years:
            rows.append((ds, Y, f"2003–{Y-2}", Y - 1, Y))
    return pipe(pd.DataFrame(rows, columns=["Dataset", "Fold", "Training harvest years", "Validation year", "Test year"]),
                "Table 5. Expanding-window partitions (temporal protocol). Validation is used only for early stopping and recalibration.")


def t_hyper():
    rows = [
        ("Persistence / climatology / linear trend", "previous-season yield / mean of past yields / per-region OLS trend (≥ 5 past seasons)", "none"),
        ("VI regression", "ridge (α = 1) on NDVI and FPAR seasonal-third means and maxima, observed fraction, trend", "–"),
        ("Ridge", "ridge (α = 1) on seasonal-third means of all 12 predictors, NDVI/FPAR maxima, static inputs", "–"),
        ("LightGBM", "400 trees, learning rate 0.03, 31 leaves, min. 20 samples per leaf, row/column subsampling 0.8", "–"),
        ("LSTM", "2 layers, 64 units, dropout 0.1, Gaussian head", "AdamW, lr 2 × 10^−3^, wd 10^−4^, batch 256, ≤ 60 epochs, early stopping (patience 6) on validation NLL"),
        ("Transformer", "2 encoder layers, width 64, 4 heads, feed-forward 128, pre-norm, dropout 0.1, Gaussian head", "as LSTM"),
        ("DST-CT twin (all variants)", "modality attention (3 groups) + GRU cell, 64 units; head 2 × 64, dropout 0.1", "as LSTM; trained on complete seasons"),
        ("MC dropout", "DST-CT twin, 30 stochastic passes, dropout 0.1", "–"),
        ("Deep ensemble", "the three seed-specific twins, Gaussian mixture", "–"),
        ("Diffusion denoiser", "3-layer transformer, width 64, 4 heads; cosine schedule, *S*~T~ = 1000", "AdamW, one-cycle lr (max 10^−3^), batch 256, ≈ 6000 iterations (60–150 epochs); loss eq. (2)"),
        ("DST-CT sampling", "DDIM, *S* = 50, η = 1, *M* = 10, x̂~0~ clipped to ±6 s.d.", "–"),
    ]
    return pipe(pd.DataFrame(rows, columns=["Model", "Architecture / configuration", "Training"]),
                "Table 6. Models and hyper-parameters. Capacities were fixed a priori; the only data-driven choices use the validation season.")
