# DST-CT on CY-Bench: code and results for the revised IJRS manuscript (TRES-PAP-2025-1681)

Chandiraprakash N and A. Chinnasamy. *Diffusion-based spatio-temporal crop twin for zero-day yield forecasting: a
leakage-controlled evaluation on the public CY-Bench benchmark.*

Every table and figure of the revised manuscript can be regenerated from the public CY-Bench v1.10 files
(doi:10.5281/zenodo.17279151, EUPL 1.2) with the scripts in `code/`. The folder `results/` contains the fold-level
summaries produced by these scripts in the environment described below.

## Environment used for the reported results
Container with a limit of 16 CPU cores (Intel Xeon Platinum 8480C) and one NVIDIA H200 GPU; Python 3.12, PyTorch 2,
scikit-learn, LightGBM, NumPy, pandas, SciPy, matplotlib. Baselines and DST-CT ran on the CPU (two threads per process,
six processes in parallel); `revise2.py` uses the GPU when available (it also runs on a CPU, more slowly).
Set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NTH=2` before running.

## 1. Data (run inside `code/`, with `cybench-data.zip` one level up in `../data/`)
```
for x in "wheat IN" "maize IN" "maize US"; do python agg.py ../data/cybench-data.zip $x; done
for d in wheat_IN maize_IN maize_US; do cp data/$d/yield.csv data/${d}_yield_hist.csv; done
mkdir -p results preds figs && python prep.py
```

## 2. Main temporal experiment (tables 7-10, figures 3-5): 13 folds x 5 seeds
```
python run_fold.py --ds <wheat_IN|maize_IN|maize_US> --year <Y> --seed <0-4> --out preds
```
Test years: 2013-2017 (India), 2021-2023 (USA). Then `python analyze.py <ds>` for each dataset.

## 3. Generated scenarios and scenario-consistent twin (tables 14-16, figure 10): 13 folds x 5 seeds
```
python revise2.py --ds <ds> --year <Y> --seed <0-4> --dif_iters 10000 --repaint 1 --out preds2
python summary2.py
python fig_scenario_skill.py
```
Generator settings were chosen on the 2012 validation season only (table 15):
`python revise2.py --ds wheat_IN --year 2013 --dif_iters {0,10000,30000} --repaint {0,1} --eta {0,0.3,0.6,1} --e1_only --out tune`.

## 4. Spatial hold-out, transfer, sensitivity, interpretability, all remaining tables and figures
`sh run_rest.sh` (spatial folds x 5 seeds, transfer x 5 seeds, S_T sensitivity, saved models, deep analyses,
`analyze.py <ds> spatial`, `fig_data.py`, `calc_numbers.py`, `fig_results.py`). `python anchor_check.py` reproduces the
anchor comparison of section 4.2.

## Licence
Code: MIT licence. Data: CY-Bench v1.10 under EUPL 1.2 (not redistributed here).
