#!/usr/bin/env bash
# Run an Atelier ASR job on a LOCAL copy of the dataset (the hub runner does the same rewrite).
#   atelier/run.sh loz|toi  [DATASET_DIR]  [OUT_DIR]
# DATASET_DIR = the directory holding manifest.csv and the audio shards
# (default build/hub/zambezi-voice, produced by `python3 scripts/zakuro_hub.py build`).
# Needs `sakura-ml` (with sakura.atelier) and `asr-deepspeech` in the active environment.
set -euo pipefail
lang="${1:?usage: run.sh loz|toi [DATASET_DIR] [OUT_DIR]}"
data="$(cd "${2:-build/hub/zambezi-voice}" && pwd)"
out="${3:-runs/$lang}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
python3 - "$here/$lang.yaml" "$data" "$out/spec.local.yaml" <<'PY'
import sys, yaml
src, data, dst = sys.argv[1:]
spec = yaml.safe_load(open(src))
spec["data"]["uri"] = "file://" + data
yaml.safe_dump(spec, open(dst, "w"), sort_keys=False)
PY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec python3 -m sakura.atelier run "$out/spec.local.yaml" --out "$out" ${ATELIER_DEVICE:+--device "$ATELIER_DEVICE"}
