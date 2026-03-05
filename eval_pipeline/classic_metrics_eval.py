#!/usr/bin/env python3
import os
import sys
import time

# ==================== 1. Strict environment variable lock (avoid CPU oversubscription deadlocks) ====================
# Must be set before importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==============================================================================

from torch.distributed.elastic.multiprocessing.errors import record
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import collections
import shutil
import tempfile
import copy
import unicodedata
import torch.multiprocessing as mp
from classic_ocr_tools import norm, poly_to_bbox, bbox_iou, point_in_bbox, bbox_center, union_bboxes, group_into_lines

# ==================== tools function ====================

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class UnifiedMetricsEvaluator:
    def __init__(self, device: str = "auto", cache_dir: str = None):
        """Initialize evaluator"""
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        self.models = {}
        self.cache_dir = cache_dir
        self._load_models()
    
    def _load_models(self):
        """Load all required models (Lazy Import Mode)"""
        print(f"[{self.device}] Loading models...")
        
        # 1. PaddleOCR
        try:
            from paddleocr import PaddleOCR
            # use_gpu=False to avoid multi-process GPU memory contention and hangs
            self.models['ocr'] = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
            self.paddleocr_available = True
            print(f"[{self.device}] PaddleOCR initialized")
        except ImportError:
            self.paddleocr_available = False
            logging.warning("PaddleOCR not available")
        
        # 2. CLIP
        try:
            import clip
            clip_model, clip_preprocess = clip.load("ViT-L/14", device=self.device, jit=False, download_root=self.cache_dir)
            clip_model.eval()
            self.models['clip_official'] = clip_model
            self.models['clip_official_preprocess'] = clip_preprocess
            self.clip_available = True
            print(f"[{self.device}] CLIP loaded")
        except Exception as e:
            logging.error(f"Failed to load CLIP: {e}")
            self.clip_available = False
        
        # 3. OpenCLIP & Aesthetic
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai", cache_dir=self.cache_dir)
            model.to(self.device)
            model.eval()
            self.models['openclip'] = model
            self.models['openclip_preprocess'] = preprocess
            
            aesthetic_model = self._load_aesthetic_model()
            if aesthetic_model:
                self.models['aesthetic'] = aesthetic_model
                print(f"[{self.device}] Aesthetic model loaded")
            self.openclip_available = True
        except Exception as e:
            self.openclip_available = False

    def _load_aesthetic_model(self):
        try:
            import torch.nn as nn
            model_path = os.path.join(self.cache_dir, "sa_0_4_vit_l_14_linear.pth")
            if os.path.exists(model_path):
                m = nn.Linear(768, 1)
                s = torch.load(model_path, map_location=self.device)
                m.load_state_dict(s)
                m.eval().to(self.device)
                return m
        except Exception as e:
            logging.warning(f"Could not load aesthetic model: {e}")
        return None

    def get_ld(self, ls1: str, ls2: str) -> float:
        """Calculate normalized version of Levenshtein distance"""
        if not self.paddleocr_available:
            return 0.0
        import Levenshtein
        edit_dist = Levenshtein.distance(ls1, ls2)
        return 1 - edit_dist / (max(len(ls1), len(ls2)) + 1e-5)

    def compute_roi_ned(self, raw_items, gen_items, source_text, target_text) -> float:
        """Compute ROI-based normalized edit distance (ROI-aware NED)."""
        import Levenshtein
        source_norm = norm(source_text)

        best_sim = 0.0
        source_bbox = None

        for item in raw_items:
            sim = self.get_ld(source_norm, norm(item["text"]))
            if sim > 0.6 and sim > best_sim:
                best_sim = sim
                source_bbox = item["bbox"]

        if source_bbox is None:
            return 0.0

        pred_texts_in_roi = [
            item for item in gen_items
            if bbox_iou(item["bbox"], source_bbox) > 0.3
        ]
        if not pred_texts_in_roi:
            return 0.0

        pred_texts_in_roi.sort(key=lambda x: x["bbox"][0])

        pred_norm = norm("".join([it["text"] for it in pred_texts_in_roi]))
        target_norm = norm(target_text)

        ned_score = self.get_ld(target_norm, pred_norm)

        if self.get_ld(source_norm, pred_norm) > 0.9 and self.get_ld(source_norm, target_norm) < 0.5:
            ned_score *= 0.2

        return ned_score

    def compute_ocr_metrics_textedit(
        self,
        raw_img_path: str,
        gen_img_path: str,
        source_text: str,
        target_text: str,
        target_weight: float = 0.5,
        iou_threshold: float = 0.5
    ) -> Dict:
        """OCR evaluation for text editing."""
        default_res = {
            "target_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roi_ned": 0.0,
        }

        if not self.paddleocr_available or "ocr" not in self.models:
            return default_res

        try:
            source_norm = norm(source_text)
            target_norm = norm(target_text)

            raw_img = Image.open(raw_img_path)
            gen_img_original = Image.open(gen_img_path)

            # Uniformly resize generated image to original size for fair comparison
            if gen_img_original.mode != "RGB":
                gen_img_original = gen_img_original.convert("RGB")

            if gen_img_original.size != raw_img.size:
                gen_img_resized = gen_img_original.resize(raw_img.size, Image.LANCZOS)
            else:
                gen_img_resized = gen_img_original

            # OCR: raw
            raw_ocr = self.models["ocr"].ocr(raw_img_path, cls=True)
            raw_lines_src = raw_ocr[0] if (raw_ocr and raw_ocr[0]) else []
            raw_items = [
                {"text": line[1][0], "bbox": poly_to_bbox(line[0]),
                 'poly': line[0], 'score': line[1][1]}
                for line in raw_lines_src
            ]
            if not raw_items:
                return default_res

            raw_lines = group_into_lines(raw_items)

            best_line_idx = -1
            best_sim = 0.0
            for i, line_items in enumerate(raw_lines):
                line_text = "".join(item["text"] for item in line_items)
                line_norm = norm(line_text)
                
                if source_norm in line_norm:
                    sim = 1.0
                else:
                    sim = self.get_ld(source_norm, line_norm)
                
                if sim > best_sim:
                    best_sim = sim
                    best_line_idx = i

            if best_line_idx == -1 or best_sim < 0.5:
                return default_res

            # OCR: gen (numpy array)
            gen_ocr = self.models["ocr"].ocr(np.array(gen_img_resized), cls=True)
            gen_lines_src = gen_ocr[0] if (gen_ocr and gen_ocr[0]) else []
            gen_items = [
                {"text": line[1][0], "bbox": poly_to_bbox(line[0]),
                 'poly': line[0], 'score': line[1][1]}
                for line in gen_lines_src
            ]
            if not gen_items:
                return default_res

            # Region split
            raw_edit_line = raw_lines[best_line_idx]
            raw_edit_region = union_bboxes([item['bbox'] for item in raw_edit_line])
            gen_edit_region = raw_edit_region
            
            gen_region_items = []
            gen_bg_items = []
            
            for item in gen_items:
                iou = bbox_iou(item['bbox'], gen_edit_region)
                center = bbox_center(item['bbox'])
                if iou > iou_threshold or point_in_bbox(center, gen_edit_region):
                    gen_region_items.append(item)
                else:
                    gen_bg_items.append(item)

            # Target similarity: find target text in the edited region
            best_target_sim = 0.0
            if gen_region_items:
                for it in gen_region_items:
                    best_target_sim = max(best_target_sim, self.get_ld(target_norm, norm(it["text"])))

                for line in group_into_lines(gen_region_items):
                    text_merged = "".join(it["text"] for it in line)
                    best_target_sim = max(best_target_sim, self.get_ld(target_norm, norm(text_merged)))

            # Penalize if source still exists while target does not
            gen_all_text_norm = norm("".join(item["text"] for item in gen_items))
            if source_norm in gen_all_text_norm and target_norm not in gen_all_text_norm:
                best_target_sim *= 0.2

            target_accuracy = best_target_sim

            # Background line alignment similarity
            raw_bg_items = [
                item
                for i, line in enumerate(raw_lines)
                if i != best_line_idx
                for item in line
            ]
            raw_bg_lines = group_into_lines(raw_bg_items)
            gen_bg_lines = group_into_lines(gen_bg_items)

            used_gen_indices = set()
            bg_sims = []
            
            for raw_line in raw_bg_lines:
                raw_l_norm = norm(''.join(it['text'] for it in raw_line))
                current_best_sim = 0.0
                best_gen_idx = -1
                
                for j, gen_line in enumerate(gen_bg_lines):
                    if j in used_gen_indices:
                        continue
                    gen_l_norm = norm(''.join(it['text'] for it in gen_line))
                    sim = self.get_ld(raw_l_norm, gen_l_norm)
                    if sim > current_best_sim:
                        current_best_sim = sim
                        best_gen_idx = j
                
                if best_gen_idx != -1:
                    used_gen_indices.add(best_gen_idx)
                
                bg_sims.append(current_best_sim)

            # Compute Precision / Recall / F1
            w_target = target_weight
            num_gt_bg = len(raw_bg_lines)
            num_pred_bg = len(gen_bg_lines)
            w_bg_unit = (1.0 - w_target) / num_gt_bg if num_gt_bg > 0 else 0.0
            
            tp_score = (target_accuracy * w_target) + (sum(bg_sims) * w_bg_unit)
            gt_total_weight = w_target + (num_gt_bg * w_bg_unit)
            pred_total_weight = w_target + (num_pred_bg * w_bg_unit)
            
            recall = tp_score / gt_total_weight if gt_total_weight > 0 else 0.0
            precision = tp_score / pred_total_weight if pred_total_weight > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            roi_ned_score = self.compute_roi_ned(
                raw_items,
                gen_items,
                source_text,
                target_text,
            )

            return {
                "target_accuracy": target_accuracy,
                'precision': precision,
                'recall': recall,
                "f1": f1,
                "roi_ned": roi_ned_score,
            }

        except Exception as e:
            logging.error(f"Compute OCR metrics failed: {e}")
            return default_res

    def compute_clip_score_batch(self, image_paths: List[str], texts: List[str]) -> List[float]:
        """Batch compute CLIPScore"""
        if not self.clip_available or 'clip_official' not in self.models:
            return [0.0] * len(image_paths)
        
        try:
            import clip
            from sklearn.preprocessing import normalize
            import sklearn.preprocessing
            from packaging import version
            import warnings
            
            processed_texts = []
            for text in texts:
                processed_texts.append(text)
            
            images = []
            for image_path in image_paths:
                try:
                    image = Image.open(image_path)
                    image_input = self.models['clip_official_preprocess'](image).unsqueeze(0)
                    images.append(image_input)
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path}: {e}")
                    images.append(torch.zeros(1, 3, 224, 224))
            
            if not images:
                return []

            images_batch = torch.cat(images, dim=0).to(self.device)
            texts_batch = clip.tokenize(processed_texts, truncate=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.models['clip_official'].encode_image(images_batch)
                text_features = self.models['clip_official'].encode_text(texts_batch)
                
                image_features_np = image_features.cpu().numpy()
                text_features_np = text_features.cpu().numpy()
                
                if version.parse(np.__version__) < version.parse('1.21'):
                    image_features_np = sklearn.preprocessing.normalize(image_features_np, axis=1)
                    text_features_np = sklearn.preprocessing.normalize(text_features_np, axis=1)
                else:
                    warnings.warn('Using compat normalization')
                    image_features_np = image_features_np / np.sqrt(np.sum(image_features_np**2, axis=1, keepdims=True))
                    text_features_np = text_features_np / np.sqrt(np.sum(text_features_np**2, axis=1, keepdims=True))
                
                similarities = np.sum(image_features_np * text_features_np, axis=1)
                clip_scores = 2.5 * np.clip(similarities, 0, None)
                
                return clip_scores.tolist()
        
        except Exception as e:
            logging.error(f"Batch CLIP score computation failed: {e}")
            return [0.0] * len(image_paths)

    def compute_aesthetic_score(self, image_path: str) -> float:
        """Calculate aesthetic score"""
        if not self.openclip_available or 'aesthetic' not in self.models or 'openclip' not in self.models:
            return 0.0
        
        try:
            image = Image.open(image_path)
            image_input = self.models['openclip_preprocess'](image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.models['openclip'].encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                prediction = self.models['aesthetic'](image_features)
                
            return prediction.cpu().numpy().item()
        
        except Exception as e:
            logging.error(f"Aesthetic score computation failed for {image_path}: {e}")
            return 0.0

    def evaluate_group(self, model_name: str, group_name: str, jsonl_files: List[str], 
                       benchmark_dir: str, gt_root_dir: str, model_output_root: str) -> Dict:
        """Evaluate a group of files for a specific model"""

        # Aggregated Metrics
        agg_metrics = {
            'target_accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1s': [],
            'roi_neds': [],
            'clip_scores': [],
            'aesthetic_scores': [],
            'image_count': 0
        }

        # Store per-sample detailed results
        detailed_results = []
        print(f"[{self.device}] Processing {model_name} | Group: {group_name} ({len(jsonl_files)} files)")

        for jsonl_file in jsonl_files:
            full_path = os.path.join(benchmark_dir, jsonl_file)
            if not os.path.exists(full_path):
                logging.warning(f"File not found: {full_path}")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                data_entries = [json.loads(line) for line in f if line.strip()]
            
            # 1. Prepare Batches for this file
            batch_paths, batch_prompts, valid_indices = [], [], []

            for idx, entry in enumerate(data_entries):
                # Resolve Model Generated Image Path
                raw_img_rel = entry.get('original_image', '')
                # class_id = parts[0]   # e.g., '1.1.1'
                # filename = parts[-1]  # e.g., '1901685000029.0.jpg'
                gen_img_path = os.path.join(model_output_root, model_name, raw_img_rel.split('/')[0], raw_img_rel.split('/')[-1])

                if gen_img_path and os.path.exists(gen_img_path):
                    # For CLIP/Aesthetic/OCR
                    batch_paths.append(gen_img_path)
                    batch_prompts.append(entry.get('gt_caption', ''))
                    valid_indices.append(idx)
            
            # 2. Compute CLIP Batch
            if batch_paths:
                clip_scores = self.compute_clip_score_batch(batch_paths, batch_prompts)
                clip_map = {idx: score for idx, score in zip(valid_indices, clip_scores)}
            else:
                clip_map = {}

            # 3. Compute Per-Image Metrics
            for idx, entry in enumerate(tqdm(data_entries, desc=f"  Evaluating {jsonl_file}", leave=False)):
                if idx not in clip_map:
                    continue
                
                raw_img_rel  = entry.get('original_image', '')
                raw_img_path = os.path.join(gt_root_dir, raw_img_rel)
                gt_img_rel   = entry.get('gt_image', '')
                gt_img_path  = os.path.join(gt_root_dir, gt_img_rel)
                gen_img_path = os.path.join(model_output_root, model_name, raw_img_rel.split('/')[0], raw_img_rel.split('/')[-1])

                # Read source_text and target_text directly from entry
                source_text = entry.get('source_text', '')
                target_text = entry.get('target_text', '')

                # CLIPScore
                clip_score = clip_map[idx]
                agg_metrics['clip_scores'].append(clip_score)
                
                # Aesthetic Score
                aesthetic_score = self.compute_aesthetic_score(gen_img_path)
                agg_metrics['aesthetic_scores'].append(aesthetic_score)

                # OCR metrics (pass source_text and target_text)
                ocr_res = self.compute_ocr_metrics_textedit(
                    raw_img_path=raw_img_path,
                    gen_img_path=gen_img_path,
                    source_text=source_text,
                    target_text=target_text,
                    target_weight=0.5,
                    iou_threshold=0.5
                )

                # Aggregate metrics
                agg_metrics['image_count'] += 1
                agg_metrics['target_accuracies'].append(ocr_res['target_accuracy'])
                agg_metrics['precisions'].append(ocr_res['precision'])
                agg_metrics['recalls'].append(ocr_res['recall'])
                agg_metrics['f1s'].append(ocr_res['f1'])
                agg_metrics['roi_neds'].append(ocr_res['roi_ned'])
                
                # Save per-sample detailed information
                detailed_results.append({
                    'id': entry.get('id'),
                    'prompt': entry.get('prompt', ''),
                    'path': {
                        'original_image': raw_img_path,
                        'edited_image': gen_img_path,
                        'gt_image': gt_img_path
                    },
                    'score': {
                        'ocr_accuracy': float(ocr_res['target_accuracy']),
                        'ocr_precision': float(ocr_res['precision']),
                        'ocr_recall': float(ocr_res['recall']),
                        'ocr_f1': float(ocr_res['f1']),
                        'clip_score': float(clip_score),
                        'ned_score': float(ocr_res['roi_ned']),
                        'aesthetic_score': float(aesthetic_score)
                    }
                })

        def sm(l): 
            return np.mean(l) if l else 0.0
        final_group_results = {
            'Group': group_name,
            'OCR Accuracy': sm(agg_metrics['target_accuracies']),
            'OCR Precision': sm(agg_metrics['precisions']),
            'OCR Recall': sm(agg_metrics['recalls']),
            'OCR F1': sm(agg_metrics['f1s']),
            'CLIPScore': sm(agg_metrics['clip_scores']),
            'NED': sm(agg_metrics['roi_neds']),
            'Aesthetic Score': sm(agg_metrics['aesthetic_scores']),
            'Total Images': agg_metrics['image_count'],
            'detailed_results': detailed_results
        }
        
        return final_group_results

def worker_process(model_name, gpu_id, args):
    try:
        device = f"cuda:{gpu_id}"
        print(f"\n>>> Worker started: Model={model_name} on Device={device}")
        os.environ["TORCH_HOME"] = args.cache_dir 
        evaluator = UnifiedMetricsEvaluator(device=device, cache_dir=args.cache_dir)
        
        groups = {
            "Virtual": [
                "1.1.1.jsonl", "1.1.2.jsonl", "1.1.3.jsonl",
                "1.2.1.jsonl", "1.2.2.jsonl",
                "1.3.1.jsonl", "1.3.2.jsonl",
                "1.4.1.jsonl", "1.4.2.jsonl", "1.4.3.jsonl", "1.4.4.jsonl"
            ],
            "Real": [
                "2.1.jsonl", "2.2.jsonl", "2.3.jsonl", "2.4.jsonl", 
                "2.5.jsonl", "2.6.jsonl", "2.7.jsonl"
            ]
        }

        # Evaluate
        res_v = evaluator.evaluate_group(model_name, "Virtual", groups["Virtual"], args.benchmark_dir, args.gt_root_dir, args.model_output_root)
        res_r = evaluator.evaluate_group(model_name, "Real", groups["Real"], args.benchmark_dir, args.gt_root_dir, args.model_output_root)
        
        metrics_order = [
                "OCR Accuracy", "OCR Precision", "OCR Recall", "OCR F1",
                "NED", "CLIPScore", "Aesthetic Score"
            ]
        final_res = {
            'summary_by_model': {model_name: {'Virtual': res_v, 'Real': res_r}}, 
            'metrics_list': metrics_order
            }
        
        # Print result table
        print(f"\n--- Results for {model_name} ---")
        print(f"{'Metric':<20} | {'Real':<15} | {'Virtual':<15}")
        print("-" * 60)
        for m in final_res['metrics_list']:
            print(f"{m:<20} | {res_r.get(m, 0.0):.4f}          | {res_v.get(m, 0.0):.4f}")
        print("-" * 60)

        out_path = os.path.join(args.output_dir, f"{model_name}.json")
        with open(out_path, 'w') as f:
            json.dump(convert_numpy_types(final_res), f, indent=4, ensure_ascii=False)

        print(f">>> Worker finished: {model_name}. Saved to {out_path}")
    except Exception as e:
        print(f"!!! Error in worker: {e}"); import traceback; traceback.print_exc()

@record
def main():
    parser = argparse.ArgumentParser(description='Unified text-to-image generation evaluation tool')
    parser.add_argument('--benchmark_dir', required=True, help='Directory containing the .jsonl files')
    parser.add_argument('--gt_root_dir', required=True, help='Root directory for Ground Truth images')
    parser.add_argument('--model_output_root', required=True, help='Root directory where model outputs are stored')
    parser.add_argument('--output_dir', required=True, help='result output file path')
    parser.add_argument('--models', default='bagel', help='Comma separated list of model names')
    parser.add_argument('--cache_dir', required=True, help='HuggingFace model cache directory path')
    args = parser.parse_args()
    
    if args.cache_dir:
        os.environ['TORCH_HOME'] = args.cache_dir
    mp.set_start_method('spawn', force=True)
    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    gpu_count = torch.cuda.device_count() or 1
    
    procs = []
    for i, model in enumerate(model_list):
        p = mp.Process(target=worker_process, args=(model, i % gpu_count, args))
        p.start()
        procs.append(p)
        time.sleep(30)

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()