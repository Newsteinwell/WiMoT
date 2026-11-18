#!/usr/bin/env bash
# Reproduces one paper table: trains every trajectory of configs/jobs/<study>.tsv in SHARDS parallel processes,
# tests them once all training has finished, and prints the comparison with the reference results.
#
#   FEATURES=/data/features RUNS=/data/runs SHARDS=4 experiments/run_table.sh main
#
# FEATURES must contain <encoder>/<dataset>/<domain>_<split>.npz (see experiments/extract_features.sh).
set -euo pipefail
STUDY=${1:?usage: run_table.sh STUDY}
: "${FEATURES:?set FEATURES}" "${RUNS:?set RUNS}"
SHARDS=${SHARDS:-1}; DEVICE=${DEVICE:-cuda}; PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."
mkdir -p "$RUNS/logs"
for phase in train test; do
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    "$PYTHON" -m wimot.study "$STUDY" --features "$FEATURES" --out "$RUNS" --device "$DEVICE" \
      --shard "$i/$SHARDS" --phase "$phase" > "$RUNS/logs/${STUDY}_${phase}_${i}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
"$PYTHON" scripts/build_tables.py --runs "$RUNS" --study "$STUDY" | tee "$RUNS/${STUDY}_table.txt"
