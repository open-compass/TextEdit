#!/bin/bash
path="your_project_path_here"
cd "$path/TextEdit"

python eval_pipeline/vlm_metrics_eval_step2.py \
  --answer_dir "$path/TextEdit/result/vlm_gemini_full_answers" \
  --output_file "$path/TextEdit/result/gemini_report_fullset.json" \
  --weights 0.4 0.3 0.1 0.1 0.1 \
  --enable_cutoff

python eval_pipeline/vlm_metrics_eval_step2.py \
  --answer_dir "$path/TextEdit/result/vlm_gemini_mini_answers" \
  --output_file "$path/TextEdit/result/gemini_report_miniset.json" \
  --weights 0.4 0.3 0.1 0.1 0.1 \
  --enable_cutoff
