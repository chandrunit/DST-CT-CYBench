#!/bin/sh
# Re-runs every remaining experiment in the same environment as the 5-seed main runs.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NTH=2
mkdir -p figs results preds_models
rm -f jobs_rest.txt
for c in 0 1 2 3 4; do y=$((2013+c)); for s in 0 1 2 3 4; do echo "python run_fold.py --ds wheat_IN --mode spatial --cluster $c --year $y --seed $s --out preds" >> jobs_rest.txt; done; done
for c in 0 1 2; do y=$((2021+c)); for s in 0 1 2 3 4; do echo "python run_fold.py --ds maize_US --mode spatial --cluster $c --year $y --seed $s --out preds" >> jobs_rest.txt; done; done
for s in 0 1 2 3 4; do echo "python transfer.py --seed $s" >> jobs_rest.txt; done
for T in 100 250; do for s in 0 1 2; do echo "python run_fold.py --ds wheat_IN --year 2017 --seed $s --T $T --only_dstct --out preds" >> jobs_rest.txt; done; done
for s in 0 1 2; do echo "python run_fold.py --ds wheat_IN --year 2017 --seed $s --save_models --out preds_models" >> jobs_rest.txt; done
echo "python run_fold.py --ds maize_US --year 2023 --seed 0 --save_models --out preds_models" >> jobs_rest.txt
echo "STAGE1 $(wc -l < jobs_rest.txt) jobs"
cat jobs_rest.txt | xargs -P 6 -I{} sh -c '{} > /dev/null 2>> rest_errors.log || echo FAILED {} >> rest_errors.log'
cp preds_models/*_models.pt preds/
echo "STAGE2 deep analyses"
python deep_analysis.py --ds wheat_IN --year 2017 --seed 0 > /dev/null 2>> rest_errors.log &
python deep_analysis.py --ds wheat_IN --year 2017 --seed 1 --parts steps > /dev/null 2>> rest_errors.log &
python deep_analysis.py --ds wheat_IN --year 2017 --seed 2 --parts steps > /dev/null 2>> rest_errors.log &
python deep_analysis.py --ds maize_US --year 2023 --seed 0 > /dev/null 2>> rest_errors.log &
wait
echo "STAGE3 tables, numbers, figures"
python analyze.py wheat_IN spatial > results/analyze_wheat_IN_spatial.log 2>&1
python analyze.py maize_US spatial > results/analyze_maize_US_spatial.log 2>&1
python fig_data.py > /dev/null 2>> rest_errors.log
python calc_numbers.py > /dev/null 2>> rest_errors.log
python fig_results.py lead,rel,scatter,deep,steps > /dev/null 2>> rest_errors.log
python summary2.py > results2/results2_summary.txt 2>&1
rm -rf pack2 && mkdir -p pack2
cp -r results results2 figs pack2/
python -c "import shutil; shutil.make_archive('../data/final_results2', 'zip', 'pack2')"
grep -c FAILED rest_errors.log
echo ALLDONE
