#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONDA_PREFIX:-}" && "${VIRTUAL_ENV:-}" == "" ]]; then
  echo "[BG-PDM] Warning: no active conda/virtual environment detected."
  echo "[BG-PDM] Activate the environment you created manually before continuing."
fi

PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -c 'import torch, nibabel, yaml, sklearn; print("Python dependencies: OK"); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'

echo "[BG-PDM] Step 1/2: auditing MSD Colon data..."
"$PYTHON_BIN" -m bgpdm.audit --config configs/default.yaml

if [[ "${BG_PDM_AUDIT_ONLY:-0}" == "1" ]]; then
  echo "[BG-PDM] Audit-only mode requested; stopping before experiments."
  exit 0
fi

echo "[BG-PDM] Step 2/2: running patient-level experiments..."
if [[ "${BG_PDM_SMOKE:-0}" == "1" ]]; then
  echo "[BG-PDM] Smoke mode: one fold, reduced cases/bank sizes, two methods."
  "$PYTHON_BIN" -m bgpdm.experiment --config configs/default.yaml --smoke
else
  "$PYTHON_BIN" -m bgpdm.experiment --config configs/default.yaml
fi

echo "[BG-PDM] Finished. Metrics: outputs/full/experiment_results.json"
echo "[BG-PDM] Outputs remain local and are excluded from Git."
