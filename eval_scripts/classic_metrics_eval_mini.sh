#!/bin/bash

# Model list
MODELS="your_model_name_here"

path="your_project_path_here"
CACHE_DIR="$path/TextEdit/checkpoint"
export PADDLE_OCR_BASE_DIR="$CACHE_DIR/.paddleocr"

# === Path configuration ===
BENCHMARK_DIR="$path/TextEdit/eval_prompts/miniset"
GT_ROOT_DIR="$path/TextEdit/data"
MODEL_OUTPUT_ROOT="$path/TextEdit/output"

# Output directory (the script will auto-generate {MODEL}.json here)
OUTPUT_DIR="$path/TextEdit/result/classic_miniset"

mkdir -p $OUTPUT_DIR

# === Force shell-level environment variable lock ===
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$path/TextEdit"

python eval_pipeline/classic_metrics_eval.py \
    --benchmark_dir "$BENCHMARK_DIR" \
    --gt_root_dir "$GT_ROOT_DIR" \
    --model_output_root "$MODEL_OUTPUT_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --models "$MODELS"

echo "Done."