# DST-CT on CY-Bench: code and results for the revised IJRS manuscript (TRES-PAP-2025-1681)

Chandiraprakash N and A. Chinnasamy. *Diffusion-based spatio-temporal crop twin for zero-day yield forecasting: a
leakage-controlled evaluation on the public CY-Bench benchmark.*

Every table and figure of the revised manuscript can be regenerated from the public CY-Bench v1.10 files
(doi:10.5281/zenodo.17279151, EUPL 1.2) with the scripts in this repository (all scripts are in the top-level folder).

## Seeds
The manuscript reports **three seeds (0, 1, 2)** for every learned model. Running the commands below with
`--seed 0`, `1` and `2` reproduces the manuscript tables (for example, DST-CT for wheat in India, test years
2013-2017: RMSE 0.639 t/ha at sowing and 0.601 t/ha at the end of the season). The summary files and
`numbers.json` in `results.zip` are averages over five seeds (0-4), because two extra seeds were also run;
they differ from the manuscript by at most a few thousandths of a t/ha and lead to the same conclusions.
The spatial hold-out (table 11) and the transfer experiment (table 12) in the manuscript use seed 0.

## Environment used for the reported results
Container with a limit of 16 CPU cores (Intel Xeon Platinum 8480C) and one multi-instance partition (4g.71gb) of an NVIDIA H200 GPU; Python 3.12, PyTorch 2,
scikit-learn, LightGBM, NumPy, pandas, SciPy, matplotlib. Baselines and DST-CT ran on the CPU (two threads per process,
six processes in parallel). Set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NTH=2` before running.

## 1. Data (run in the repository folder, with `cybench-data.zip` in `../data/`)
```
for x in "wheat IN" "maize IN" "maize US"; do python agg.py ../data/cybench-data.zip $x; done
for d in wheat_IN maize_IN maize_US; do cp data/$d/yield.csv data/${d}_yield_hist.csv; done
mkdir -p results preds figs && python prep.py
```

## 2. Main temporal experiment (tables 7-10, figures 3-5): 13 folds x 3 seeds
```
python run_fold.py --ds <wheat_IN|maize_IN|maize_US> --year <Y> --seed <0-2> --out preds
```
Test years: 2013-2017 (India), 2021-2023 (USA). Then `python analyze.py <ds>` for each dataset.

## 3. Regional performance for wheat in India (section 6.10, table 14, `state_results.csv`)
```
for Y in 2006 ... 2017; for S in 0 1 2:  python run_fold.py --ds wheat_IN --year $Y --seed $S --out preds
python seeds_summary.py
```
`seeds_summary.py` computes, for every state, issue time and seed, the RMSE, MAE and R2 of DST-CT and of the
per-region linear trend, and the number of test years in which DST-CT is more accurate. `state_results.csv`
contains the resulting table for all states (state codes of the CY-Bench `adm_id` are listed in `states.py`).

## 4. Spatial hold-out, transfer, sensitivity, interpretability, all remaining tables and figures
`sh run_rest.sh` (spatial folds, transfer, S_T sensitivity, saved models, deep analyses,
`analyze.py <ds> spatial`, `fig_data.py`, `calc_numbers.py`, `fig_results.py`). `python anchor_check.py` reproduces the
anchor comparison of section 4.2.

## Exploratory scripts (not reported in the manuscript)
`revise2.py`, `summary2.py`, `fig_scenario_skill.py`, `run_gpu_trials.sh` and the folder `revised_generator/` in
`results.zip` belong to an exploratory experiment with a retrained scenario generator. Its results are not part of
the manuscript.

## Licence
Code: MIT licence. Data: CY-Bench v1.10 under EUPL 1.2 (not redistributed here).
