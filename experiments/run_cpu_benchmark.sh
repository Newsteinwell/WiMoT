#!/usr/bin/env bash
# CPU cost of WiMoT (paper Table 10): results/cpu_benchmark/<part>_<dataset>_t<threads>.json.
#
#   WEIGHTS=/data/weights   experiments/run_cpu_benchmark.sh encoder 4 [DATASET ...]
#   FEATURES=/data/features experiments/run_cpu_benchmark.sh stream  1 [DATASET ...]
#
# The thread count is fixed for every BLAS library before Python starts. The paper used 1 and 4 threads.
set -euo pipefail
part=$1; threads=$2; shift 2
[ $# -eq 0 ] && set -- 3do wimans widar
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads VECLIB_MAXIMUM_THREADS=$threads
for ds in "$@"; do
  case $part in
    encoder) "$PYTHON" scripts/cpu_benchmark.py encoder --dataset "$ds" --threads "$threads" --weights "${WEIGHTS:?set WEIGHTS}/mantis" ;;
    stream)  "$PYTHON" scripts/cpu_benchmark.py stream  --dataset "$ds" --threads "$threads" --features "${FEATURES:?set FEATURES}/mantis/$ds" ;;
    *) echo "part must be encoder or stream" >&2; exit 2 ;;
  esac
done
