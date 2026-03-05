#!/bin/bash

API_KEY="your_gemini_api_key_here"
BASE_URL="your_gemini_api_base_url_here"

path="your_project_path_here"
cd "$path/TextEdit"

python eval_pipeline/vlm_metrics_eval_step1.py \
  --input_data_dir "$path/TextEdit/eval_prompts/fullset" \
  --model_output_root "$path/TextEdit/output" \
  --gt_data_root "$path/TextEdit/data" \
  --output_base_dir "$path/TextEdit/result/vlm_gemini_full_answers" \
  --model_name "gemini-3-pro-preview" \
  --models "your_model_name_here" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --num_workers 64