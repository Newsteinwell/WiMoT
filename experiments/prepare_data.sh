#!/usr/bin/env bash
# Prepares the five streams from the official raw data (see docs/REPRODUCE.md, "Data").
#
#   RAW_3DO=... RAW_WIMANS=... WIMANS_ANNOTATION=... RAW_WIDAR=... RAW_WIDAR_ROOMS=... PREPARED=/data/prepared \
#     experiments/prepare_data.sh
#
# RAW_3DO        folder with d1/ d2/ d3/ session folders, each holding csiposreg.csv
# RAW_WIMANS     folder with the WiMANS amplitude files <recording>.npy (wifi_csi/amp)
# WIMANS_ANNOTATION  WiMANS annotation.csv
# RAW_WIDAR      extracted Widar3.0 Room-1 archives (20181130_user*.zip), searched recursively for *.dat
# RAW_WIDAR_ROOMS    extracted Widar3.0 archives 20181204.zip, 20181209.zip (Room 2) and 20181211.zip (Room 3)
set -euo pipefail
: "${PREPARED:?set PREPARED}"; PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."
[ -n "${RAW_3DO:-}" ] && "$PYTHON" -m wimot.data.prepare 3do --raw "$RAW_3DO" --out "$PREPARED/3do"
if [ -n "${RAW_WIMANS:-}" ]; then
  [ -n "${WIMANS_ANNOTATION:-}" ] && "$PYTHON" -m wimot.data.prepare wimans-split --annotation "$WIMANS_ANNOTATION" \
    --out "$PREPARED/wimans_split.json"
  "$PYTHON" -m wimot.data.prepare wimans --raw "$RAW_WIMANS" --out "$PREPARED/wimans"
fi
[ -n "${RAW_WIDAR:-}" ] && "$PYTHON" -m wimot.data.prepare widar --stream widar --raw "$RAW_WIDAR" --out "$PREPARED/widar"
if [ -n "${RAW_WIDAR_ROOMS:-}" ]; then
  for s in widar_r2 widar_r3; do
    "$PYTHON" -m wimot.data.prepare widar --stream "$s" --raw "$RAW_WIDAR_ROOMS" --out "$PREPARED/$s"
  done
fi
