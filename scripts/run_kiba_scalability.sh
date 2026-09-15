#!/usr/bin/env bash
# Runs KIBA scales serially from deterministic random initialization. Each
# scale gets a recoverable best-validation checkpoint and a live log file.
set -euo pipefail

mkdir -p results/kiba_scalability
for scale in 25 50 75 100; do
  fraction="0.${scale}"
  if [[ "$scale" == "100" ]]; then fraction="1.0"; fi
  python trim_dta_experiment.py \
    --dataset kiba --mode default --dim 128 --batch-size 100 --lr 1e-4 \
    --epochs 50 --timing-epochs 3 --inference-repeats 3 --seed 0 \
    --fraction "$fraction" \
    --best-checkpoint "results/kiba_scalability/kiba_${scale}_best.pt" \
    --out "results/kiba_scalability/kiba_${scale}.json" \
    > "results/kiba_scalability/kiba_${scale}.log" 2>&1
done
