#!/bin/bash
# Full pipeline runner: convert JSONs -> CSV, then preprocess all rows
set -euo pipefail

PROJECT=/home/exedev/Data-ML
VENV=/home/exedev/pipeline-env
LOG_DIR=$PROJECT/logs

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/pipeline.log" 2>&1

echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") pipeline start ====="

cd "$PROJECT"

# Step 1: Convert raw JSONs to CSVs
echo "--- Step 1: Converting JSON -> CSV ---"
"$VENV/bin/python" Job_pipeline/convert_raw_to_csv.py \
  --raw-dir Job_pipeline/data/raw \
  --out-dir Job_pipeline/data/raw \
  --processed-dir Job_pipeline/data/processed

# Step 2: Run full preprocessing pipeline (no Gemini fallback for speed)
echo "--- Step 2: Preprocessing pipeline ---"
"$VENV/bin/python" Job_pipeline/run_preprocessing_pipeline.py \
  --raw-dir Job_pipeline/data/raw \
  --processed-dir Job_pipeline/data/processed

echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") pipeline complete ====="
