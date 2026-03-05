#!/usr/bin/env python3
"""
Step 2: Calculate Gemini Evaluation Scores
Read 1-5 scores directly from the gemini_eval_score field.
"""
import argparse
import os
import json
import glob
import csv
import numpy as np
from tqdm import tqdm

# ================= Configuration =================

# Default weights
DEFAULT_WEIGHTS = [0.40, 0.30, 0.10, 0.10, 0.10]

# Model order
MODEL_ORDER = [
    "bagel",
]

# Metric name mapping
METRIC_NAMES = {
    "Q1": "Text Accuracy",
    "Q2": "Text Preservation",
    "Q3": "Scene Integrity",
    "Q4": "Local Realism",
    "Q5": "Visual Coherence"
}

# Score normalization mapping: 1-5 -> 0.0-1.0
SCORE_NORMALIZATION = {
    5: 1.0,
    4: 0.8,
    3: 0.6,
    2: 0.2,
    1: 0.0
}

# ===========================================

def normalize_score(raw_score):
    """Normalize 1-5 scores to 0-1."""
    # Map to normalized score
    return SCORE_NORMALIZATION.get(raw_score)


def is_virtual_category(path_info):
    """Determine whether the sample is in the Virtual category (1.x)."""
    # Infer from original_image or gt_image path
    img_path = path_info.get("original_image", "") or path_info.get("gt_image", "")
    
    # Check whether path contains /1.x.x/
    if "/1." in img_path:
        return True
    return False


def print_markdown_table(final_report):
    """Print results in Markdown table format."""
    print("\n" + "="*150)
    print("📊 MARKDOWN TABLE FORMAT")
    print("="*150)
    
    # Header: Model | AVG (R) | AVG (V) | Q1-Q5 Real | Q1-Q5 Virtual
    headers = ["Model", "AVG (R)", "AVG (V)"]
    for i in range(1, 6):
        metric_name = METRIC_NAMES[f"Q{i}"]
        headers.append(f"{metric_name} (R)")
    for i in range(1, 6):
        metric_name = METRIC_NAMES[f"Q{i}"]
        headers.append(f"{metric_name} (V)")
    
    # Print Markdown table header
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---" for _ in headers]) + "|")
    
    # Print each model in the specified order
    for model_name in MODEL_ORDER:
        if model_name not in final_report:
            print(f"| {model_name} | N/A | N/A | " + " | ".join(["N/A"] * 10) + " |")
            continue
        
        model_data = final_report[model_name]
        
        # AVG (Real, Virtual)
        avg_real = model_data["Real"]["Weighted Score"]
        avg_virtual = model_data["Virtual"]["Weighted Score"]
        
        row = [model_name, f"{avg_real:.3f}", f"{avg_virtual:.3f}"]
        
        # Q1-Q5 Real
        for i in range(1, 6):
            q_key = f"Q{i}_{METRIC_NAMES[f'Q{i}'].replace(' ', '_')}"
            score = model_data["Real"]["Detail Metrics"].get(q_key, 0.0)
            row.append(f"{score:.3f}")
        
        # Q1-Q5 Virtual
        for i in range(1, 6):
            q_key = f"Q{i}_{METRIC_NAMES[f'Q{i}'].replace(' ', '_')}"
            score = model_data["Virtual"]["Detail Metrics"].get(q_key, 0.0)
            row.append(f"{score:.3f}")
        
        print("| " + " | ".join(row) + " |")
    
    print("="*150 + "\n")


def save_csv_file(final_report, csv_path):
    """Save results as a CSV file."""
    # CSV headers
    headers = ["Model", "AVG (R)", "AVG (V)"]
    for i in range(1, 6):
        metric_name = METRIC_NAMES[f"Q{i}"]
        headers.append(f"{metric_name} (R)")
        headers.append(f"{metric_name} (V)")
    
    rows = []
    
    # Collect data in the specified order
    for model_name in MODEL_ORDER:
        if model_name not in final_report:
            row = [model_name] + ["N/A"] * 12
            rows.append(row)
            continue
        
        model_data = final_report[model_name]
        
        # AVG (Real, Virtual)
        avg_real = model_data["Real"]["Weighted Score"]
        avg_virtual = model_data["Virtual"]["Weighted Score"]
        
        row = [model_name, f"{avg_real:.4f}", f"{avg_virtual:.4f}"]
        
        # Q1-Q5 Real
        for i in range(1, 6):
            q_key = f"Q{i}_{METRIC_NAMES[f'Q{i}'].replace(' ', '_')}"
            score = model_data["Real"]["Detail Metrics"].get(q_key, 0.0)
            row.append(f"{score:.4f}")
        
        # Q1-Q5 Virtual
        for i in range(1, 6):
            q_key = f"Q{i}_{METRIC_NAMES[f'Q{i}'].replace(' ', '_')}"
            score = model_data["Virtual"]["Detail Metrics"].get(q_key, 0.0)
            row.append(f"{score:.4f}")
        
        rows.append(row)
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"📊 CSV file saved to: {csv_path}")


def calculate_metrics(args):
    print(f"Calculating Gemini scores from: {args.answer_dir}")
    print(f"Weights: {args.weights}")
    print(f"Cutoff Mode: {'ON' if args.enable_cutoff else 'OFF'} (Threshold: Q1 >= 4 [0.8])")
    
    model_dirs = [d for d in glob.glob(os.path.join(args.answer_dir, "*")) if os.path.isdir(d)]
    
    if not model_dirs:
        print(f"[Error] No model directories found in {args.answer_dir}")
        return

    weights = np.array(args.weights)
    final_report = {}
    
    # Print table header
    header = f"{'Model':<20} | {'Acc (V/R)':<15} | {'Pre (V/R)':<15} | {'Scn (V/R)':<15} | {'Rea (V/R)':<15} | {'Coh (V/R)':<15} | {'AVG (V/R)':<15}"
    print("\n" + "="*135)
    print(header)
    print("-" * 135)

    for m_dir in sorted(model_dirs):
        model_name = os.path.basename(m_dir)
        jsonl_files = glob.glob(os.path.join(m_dir, "*.jsonl"))
        
        # Storage structure: { "Virtual": {"weighted": [], "q_scores": [[],[],[],[],[]]}, "Real": ... }
        stats = {
            "Virtual": {"weighted": [], "q_scores": [[] for _ in range(5)]},
            "Real":    {"weighted": [], "q_scores": [[] for _ in range(5)]}
        }
        
        for fpath in jsonl_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        
                        # Determine category from path info
                        path_info = item.get("path", {})
                        is_virtual = is_virtual_category(path_info)
                        
                        group = "Virtual" if is_virtual else "Real"
                        
                        # Read gemini_eval_score
                        gemini_scores = item.get("gemini_eval_score", {})
                        
                        if not gemini_scores or not isinstance(gemini_scores, dict):
                            continue
                        
                        # --- Core scoring logic ---
                        current_scores = []
                        
                        # 1. Get and normalize Q1 score
                        q1_raw = gemini_scores.get("Q1", 0)
                        q1_score = normalize_score(q1_raw)
                        current_scores.append(q1_score)
                        
                        # 2. Decide whether to apply cutoff
                        is_cutoff = False
                        if args.enable_cutoff and q1_raw < 4:  # Q1 score < 4 (normalized 0.8)
                            is_cutoff = True
                        
                        # 3. Get and normalize Q2-Q5
                        for i in range(2, 6):
                            if is_cutoff:
                                current_scores.append(0.0)  # Cutoff to zero
                            else:
                                raw_score = gemini_scores.get(f"Q{i}", 0)
                                score = normalize_score(raw_score)
                                current_scores.append(score)
                        
                        # 4. Compute weighted total score
                        weighted_score = np.sum(np.array(current_scores) * weights)
                        
                        # 5. Record statistics
                        stats[group]["weighted"].append(weighted_score)
                        for i in range(5):
                            stats[group]["q_scores"][i].append(current_scores[i])
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"[Warning] Error processing entry in {fpath}: {e}")
                        continue
        
        # --- Summarize and print current model results ---
        display_str = f"{model_name:<20}"
        
        # Compute V/R averages for each column (Q1-Q5)
        for i in range(5):
            v_list = stats["Virtual"]["q_scores"][i]
            r_list = stats["Real"]["q_scores"][i]
            
            v_avg = np.mean(v_list) if v_list else 0.0
            r_avg = np.mean(r_list) if r_list else 0.0
            
            display_str += f" | {v_avg:.2f} / {r_avg:.2f}   "
        
        # Compute overall average (weighted average)
        v_final = np.mean(stats["Virtual"]["weighted"]) if stats["Virtual"]["weighted"] else 0.0
        r_final = np.mean(stats["Real"]["weighted"]) if stats["Real"]["weighted"] else 0.0
        
        display_str += f" | {v_final:.2f} / {r_final:.2f}"
        print(display_str)
        
        # Build detailed JSON report data
        model_summary = {}
        for grp in ["Virtual", "Real"]:
            scores = stats[grp]["weighted"]
            q_avgs = [np.mean(lst) if lst else 0.0 for lst in stats[grp]["q_scores"]]
            
            model_summary[grp] = {
                "Weighted Score": round(float(np.mean(scores)), 4) if scores else 0.0,
                "Detail Metrics": {
                    f"Q1_{METRIC_NAMES['Q1'].replace(' ', '_')}": round(q_avgs[0], 4),
                    f"Q2_{METRIC_NAMES['Q2'].replace(' ', '_')}": round(q_avgs[1], 4),
                    f"Q3_{METRIC_NAMES['Q3'].replace(' ', '_')}": round(q_avgs[2], 4),
                    f"Q4_{METRIC_NAMES['Q4'].replace(' ', '_')}": round(q_avgs[3], 4),
                    f"Q5_{METRIC_NAMES['Q5'].replace(' ', '_')}": round(q_avgs[4], 4)
                },
                "Count": len(scores)
            }
        
        final_report[model_name] = model_summary

    print("="*135)
    
    # Save JSON results
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Detailed report saved to: {args.output_file}")
    
    # Print Markdown table
    print_markdown_table(final_report)
    
    # Save CSV file
    csv_path = args.output_file.replace('.json', '.csv')
    save_csv_file(final_report, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Gemini Evaluation Scores")
    parser.add_argument("--model_order", default="bagel", help="待测模型名字顺序，逗号分隔")
    parser.add_argument("--answer_dir", required=True, help="Step1 Gemini API 评测结果目录")
    parser.add_argument("--output_file", required=True, help="输出 JSON 报告路径")
    # Weight arguments (5 floats)
    parser.add_argument("--weights", type=float, nargs=5, default=DEFAULT_WEIGHTS, 
                        help="Q1-Q5 的权重, 默认: 0.4 0.3 0.1 0.1 0.1")
    # Cutoff switch
    parser.add_argument("--enable_cutoff", action="store_true", 
                        help="开启熔断机制：如果 Q1 < 4 (归一化 0.8)，后续项记为 0 分")
    args = parser.parse_args()
    
    calculate_metrics(args)