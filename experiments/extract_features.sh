#!/usr/bin/env bash
# Frozen-encoder features of the prepared streams: FEATURES/<encoder>/<dataset>/<domain>_<split>.npz.
#
#   PREPARED=/data/prepared FEATURES=/data/features WEIGHTS=/data/weights experiments/extract_features.sh [ENCODER ...]
#
# WEIGHTS/<encoder> is a local snapshot of the model revision listed in THIRD_PARTY_NOTICES.md (mantis_random needs none).
# Default encoder: mantis (all tables); the backbone table also uses moment tspulse chronos2 timesfm25 timesfm30 mantis_random.
set -euo pipefail
: "${PREPARED:?set PREPARED}" "${FEATURES:?set FEATURES}"; PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."
ENCODERS=("${@:-mantis}")
for enc in "${ENCODERS[@]}"; do
  for ds in 3do wimans widar widar_r2 widar_r3; do
    [ -d "$PREPARED/$ds" ] || continue
    if [ "$enc" != mantis ] && [[ "$ds" == widar_r* ]]; then continue; fi
    args=(--prepared "$PREPARED/$ds" --dataset "$ds" --model "$enc" --out "$FEATURES/$enc/$ds")
    [ "$enc" != mantis_random ] && args+=(--weights "${WEIGHTS:?set WEIGHTS}/$enc")
    "$PYTHON" -m wimot.features.extract "${args[@]}"
  done
done
