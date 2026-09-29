#!/bin/sh
# GPU trials while the GPU is available: sampling-step sensitivity (S) of the revised generator + measured GPU latency.
# Tuning fold only (test = first test year, validation = year before), seed 0; generator trained once per dataset and reused.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NTH=2
mkdir -p tune_steps ckpt
run_ds() {
  for S in 25 10 50 100; do
    python revise2.py --ds $1 --year $2 --seed 0 --dif_iters 10000 --repaint 1 --eta 1.0 --steps $S --e1_only --gen_ckpt ckpt/gen_$1_$2_s0.pt --tag_suffix _S$S --out tune_steps >> tune_steps/log_$1.txt 2>&1 || echo FAILED $1 $S >> tune_steps/log_$1.txt
  done
}
run_ds wheat_IN 2013 &
run_ds maize_IN 2013 &
run_ds maize_US 2021 &
wait
echo GPUDONE >> tune_steps/log_all.txt
