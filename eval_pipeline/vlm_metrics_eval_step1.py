#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse
import base64
import re
import time
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair

import datetime
import openai

# Set environment variable
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def get_rank_id():
    """Get current replica rank ID (for multi-replica jobs)."""
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))

# ================= PROMPT TEMPLATES =================

PROMPT_MAIN_EVAL = """
You are an expert Forensic Image Analyst and Design QA Specialist.
Your task is to evaluate the quality of an AI-edited image by comparing three images.

**Images Provided (in order):**
1. **Original Image**: The unedited source image containing the text "{raw_text}".
2. **Ground Truth Image**: A human-created reference showing the ideal result with text "{target_text}".
3. **Edited Image**: The AI-generated result to be evaluated.

**Editing Task Information:**
- **Text to Remove**: "{raw_text}"
- **Text to Add**: "{target_text}"

---

## **EVALUATION RUBRIC (1-5 SCORING SYSTEM)**

Please evaluate the **Edited Image** based on the following 5 dimensions. Use the strict criteria below to assign a score from 1 to 5.

### **Q1. [Target Text Accuracy]**
*Focus: Spelling, Erasure, and Legibility of "{target_text}".*
- **5 (Perfect)**: Exact spelling match (case-sensitive). Old text completely erased. No ghosting.
- **4 (Minor Flaw)**: Text is correct but has 1 character error/typo, OR slight casing issue, OR extremely faint ghosting visible only on close inspection.
- **3 (Readable but Flawed)**: 2-3 character errors but word is recognizable. OR visible ghosting/remnants of old text that affect cleanness.
- **2 (Major Error)**: >3 character errors (misspelled heavily). OR old text is still clearly readable (failed erasure).
- **1 (Failed)**: Text is missing, gibberish, or completely wrong word. Old text remains fully intact.

### **Q2. [Non-Target Text Preservation]**
*Focus: Protection of background text (signs, prices, logos) that should NOT change.*
- **5 (Perfect)**: All non-target text is 100% preserved and legible, identical to Original/GT.
- **4 (Good)**: Main background text is preserved. Minor distinct text (far background) is slightly softened/blurred but still readable.
- **3 (Fair)**: One or two secondary text elements are blurred, damaged, or missing.
- **2 (Poor)**: Critical nearby text (directly adjacent to target) is damaged, erased, or hallucinated.
- **1 (Destructive)**: Widespread destruction or hallucination of background text.

### **Q3. [Global Scene Integrity]**
*Focus: Geometric stability of non-edited areas (background, objects, people).*
- **5 (Perfect)**: Pixel-perfect preservation of background geometry. No distortions.
- **4 (Good)**: Almost perfect, but very minor shift (<1%) in background lines or perspective.
- **3 (Noticeable)**: Visible distortion in straight lines (wavy), or slight warping of objects/faces.
- **2 (Severe)**: Major structural damage (e.g., a person's face is melted, a building collapsed).
- **1 (Chaos)**: The scene structure is completely changed or nonsensical compared to Original.

### **Q4. [Local Realism & Artifacts]**
*Focus: Inpainting quality, edge cleanliness, and seamlessness.*
- **5 (Excellent)**: Invisible edit. Clean edges, no halos, no smudges. Professional quality.
- **4 (Good)**: Very minor artifacts (e.g., slight pixelation on zoom-in), but looks natural at a glance.
- **3 (Fair)**: Visible seams, blurry rectangular patch, or "smudged" look around the text.
- **2 (Poor)**: Obvious artifacts, messy edges, or white/black box artifacts.
- **1 (Garbage)**: The edited area looks like a corrupted file or pure noise.

### **Q5. [Aesthetic & Lighting Harmony]**
*Focus: Style matching (font), lighting, shadow, and texture.*
- **5 (Seamless)**: Font style matches the GT/Context perfectly. Lighting/shadows are physically correct. Texture (grain) matches the photo.
- **4 (Integrated)**: Good style match. Lighting is mostly correct. Texture is slightly too smooth but acceptable.
- **3 (Artificial)**: Text looks "pasted on" (digital sticker look). Font style is generic (e.g., Arial) and clashes with the scene.
- **2 (Disjointed)**: Wrong color, wrong perspective, or no shading where needed.
- **1 (Mismatch)**: Text floats awkwardly, completely ignoring the scene's physics and style.

---

## **FINAL OUTPUT FORMAT (JSON ONLY)**

You must output a valid JSON object containing two dictionaries: `score` (integers) and `reason` (strings).

**Example Output:**
```json
{{
  "score": {{
    "Q1": 5,
    "Q2": 1,
    "Q3": 2,
    "Q4": 5,
    "Q5": 4
  }},
  "reason": {{
    "Q1": "The target text 'PARTY' is spelled correctly and is clearly legible. The specific target text to remove ('MUSIC') is completely gone with no ghosting.",
    "Q2": "The model caused widespread destruction of non-target text. 'NIGHT CLUB', '31 OCT', 'FREE DRINKS', 'LIVE', and 'PRICE' were all erroneously removed, and '10$' was corrupted into the hallucinated text '1TY'.",
    "Q3": "Global scene integrity is severely compromised. The skeleton's arm holding the maraca was erased, leaving the maraca floating in mid-air, which breaks the physical logic of the illustration.",
    "Q4": "Despite the semantic failures, the technical quality of the image is excellent. The edges are sharp, the background inpainting is smooth, and there are no visible pixel artifacts, blur, or noise.",
    "Q5": "The font style selected for 'PARTY' integrates well with the hand-drawn vector aesthetic of the poster, although the color is a darker maroon compared to the bright red of the original text."
  }}
}}
```
Do not output any markdown or conversational text outside the JSON block.
"""

# ====================================================

class GlobalProgress:
    """Global progress tracker across models."""
    def __init__(self, total_entries):
        self.total = total_entries
        self.processed = 0
        self.start_time = time.time()
    
    def update(self, count=1):
        self.processed += count
    
    def get_status_str(self, current_model_name):
        elapsed = time.time() - self.start_time
        progress_pct = (self.processed / self.total * 100) if self.total > 0 else 0
        eta = (elapsed / self.processed * (self.total - self.processed)) if self.processed > 0 else 0
        return f"[{current_model_name}] Overall: {self.processed}/{self.total} ({progress_pct:.1f}%) | ETA: {datetime.timedelta(seconds=int(eta))}"

def image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image_to_target(source_path, target_size):
    """
    Resize an image to the target size and save it as a temporary file.
    Return the temporary file path.
    """
    img = Image.open(source_path)
    if img.size != target_size:
        img_resized = img.resize(target_size, Image.LANCZOS)
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img_resized.save(temp_file.name, quality=100)
        return temp_file.name
    else:
        return source_path


class GeminiInferenceWrapper:
    def __init__(self, api_key, base_url, model_name="gemini-3-pro", max_workers=32):
        print(f"[INFO] Initializing Gemini Client with model: {model_name}...")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.max_workers = max_workers
        print("[INFO] Client initialized successfully.")

    def build_chat_messages(self, image_paths, prompt_text):
        """Build OpenAI-format messages (supports multiple images)."""
        content = [{"type": "text", "text": prompt_text}]
        
        for img_path in image_paths:
            if isinstance(img_path, str) and os.path.exists(img_path):
                image_data = image_to_base64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })
            elif hasattr(img_path, 'save'):  # PIL Image
                temp_path = f"/tmp/temp_img_{id(img_path)}.png"
                img_path.save(temp_path)
                image_data = image_to_base64(temp_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })
                os.remove(temp_path)
        
        messages = [{"role": "user", "content": content}]
        return messages

    def generate_single(self, messages, max_retries=5):
        """Single API call."""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=65536
                )
                results = completion.choices[0].message.content.strip()
                finish_reason = completion.choices[0].finish_reason
                if not results or finish_reason == "length":
                    print(f"completion: {completion}")
                    print(f"finish_reason: {finish_reason}")
                    print(f"Output empty or truncated, retrying...")
                    raise Exception("Output truncated or empty")
                return results
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[WARN] API call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

    def process_single_task(self, task):
        """Process one task."""
        try:
            messages = task["messages"]
            result = self.generate_single(messages)
            return task["entry_id"], result, None
        except Exception as e:
            error_msg = f"Error: {str(e)[:200]}"
            print(f"[ERROR] Task {task['entry_id']} failed: {error_msg}")
            return task["entry_id"], f"Answer: No | {error_msg}", str(e)

    def generate_batch(self, tasks):
        """Run concurrent batch inference."""
        if not tasks:
            return {}
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="API Calls"):
                entry_id, result, error = future.result()
                results[entry_id] = result
        
        return results


def scan_total_workload(args, model_list):
    """Scan total workload."""
    rank_id = get_rank_id()
    total_tasks = 0

    print(f"[Rank {rank_id}] 🔍 Scanning total workload across {len(model_list)} models...")
    
    all_input_files = sorted(glob.glob(os.path.join(args.input_data_dir, "*.jsonl"))) # 18 files
    for jsonl_path in all_input_files:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
            total_tasks += count
            
    final_total = total_tasks * len(model_list)
    print(f"[Rank {rank_id}] ✅ Total pending items across all models: {final_total}")
    return final_total


def extract_and_parse_json(text):
    if not text:
        print("Empty text received for JSON extraction.")
        return None
    try:
        # json_repair automatically handles extraction, cleaning, and parsing
        decoded_object = json_repair.loads(text)
        return decoded_object
    except Exception as e:
        print(f"[JSON Parse Error] Raw text: {text[:200]}... Repair failed: {e}")
        return None

def process_model(model_name, args, wrapper, global_tracker):
    """Process all JSONL files for a single model."""
    rank_id = get_rank_id()
    
    # Create model output directory
    model_output_dir = os.path.join(args.output_base_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Scan all JSONL files
    all_input_files = sorted(glob.glob(os.path.join(args.input_data_dir, "*.jsonl"))) # 18 files
    print(f"\n{'='*60}")
    print(f"[Rank {rank_id}] 🚀 Processing Model: {model_name}")
    print(f"[Rank {rank_id}] Found {len(all_input_files)} JSONL files")
    print(f"{'='*60}\n")
    
    for jsonl_path in all_input_files:
        filename = os.path.basename(jsonl_path)
        output_path = os.path.join(model_output_dir, filename)
        
        # Check whether already completed
        if os.path.exists(output_path) and not args.debug:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f)
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                total_count = sum(1 for _ in f)
            
            if existing_count == total_count:
                print(f"[Rank {rank_id}] ⏭️  Skipping {filename} (already completed)")
                global_tracker.update(total_count)
                continue
        
        print(f"[Rank {rank_id}] 📝 Processing {filename}...")
        
        # Read input data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
        
        # Debug mode: process only the first 3 samples
        if args.debug:
            entries = entries[:3]
            print(f"[Rank {rank_id}] 🐛 Debug mode: Processing only {len(entries)} samples")
        
        # Prepare batch tasks
        tasks = []
        temp_files = []  # Track temporary files
        entry_map = {}
        
        for entry in entries:
            entry_id    = entry.get('id')
            raw_text    = entry.get('source_text', '')
            target_text = entry.get('target_text', '')
            raw_img_rel = entry.get('original_image', '')
            gt_img_rel  = entry.get('gt_image', '')
            
            # Resolve paths
            raw_img_path = os.path.join(args.gt_data_root, raw_img_rel)
            gt_img_path  = os.path.join(args.gt_data_root, gt_img_rel)
            # class_id = parts[0]   # e.g., '1.1.1'
            # filename = parts[-1]  # e.g., '1901685000029.0.jpg'
            gen_img_path = os.path.join(args.model_output_root, model_name, raw_img_rel.split('/')[0], raw_img_rel.split('/')[-1])
            
            # Check file existence
            if not all([os.path.exists(p) for p in [raw_img_path, gt_img_path, gen_img_path]]):
                continue
            
            # Resize generated image to original size
            raw_img = Image.open(raw_img_path)
            raw_size = raw_img.size
            
            gen_img_resized_path = resize_image_to_target(gen_img_path, raw_size)
            if gen_img_resized_path != gen_img_path:
                temp_files.append(gen_img_resized_path)
            
            # Build evaluation prompt
            eval_prompt = PROMPT_MAIN_EVAL.format(
                raw_text=raw_text,
                target_text=target_text
            )

            messages = wrapper.build_chat_messages(
                image_paths=[raw_img_path, gt_img_path, gen_img_resized_path],
                prompt_text=eval_prompt
            )

            tasks.append({
                "entry_id": entry.get('id'),
                "messages": messages
            })
            entry["edited_image"] = gen_img_path
            entry_map[entry.get('id')] = entry


        if not tasks:
            print(f"[Rank {rank_id}] ⚠️  No valid tasks for {filename}")
            continue
        
        # Batch generation
        print(f"[Rank {rank_id}] 🔄 Generating {len(tasks)} evaluations...")
        results = wrapper.generate_batch(tasks)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Process results and save
        output_entries = []
        for entry_id, output in results.items():
            # Build output entry
            entry = entry_map[entry_id]

            output_entry = {
                'id': entry_id,
                'prompt': entry.get('prompt', ''),
                'path': {
                    'original_image': entry.get('original_image', ''),
                    'edited_image': entry.get('edited_image', ''),
                    'gt_image': entry.get('gt_image', '')
                },
                'gt_caption': entry.get('gt_caption', '')
            }

            entry_map[entry_id]
            # Try parsing JSON
            parsed_json = extract_and_parse_json(output)
            output_entry['gemini_raw_response'] = output
            if parsed_json and 'score' in parsed_json and 'reason' in parsed_json:
                output_entry['gemini_eval_score'] = parsed_json['score']
                output_entry['gemini_eval_reason'] = parsed_json['reason']
            else:
                output_entry['gemini_eval_score'] = None
                output_entry['gemini_eval_reason'] = None
            
            output_entries.append(output_entry)
            global_tracker.update(1)
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in output_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"[Rank {rank_id}] ✅ Saved {len(output_entries)} entries to {output_path}")
        print(f"[Rank {rank_id}] {global_tracker.get_status_str(model_name)}\n")

def main():
    parser = argparse.ArgumentParser(description="Gemini API Batch Evaluation with Model Iteration")
    parser.add_argument("--input_data_dir", required=True, help="Directory containing the evaluated prompt files(.jsonl)")
    parser.add_argument("--model_output_root", required=True, help="Root directory where model outputs are stored")
    parser.add_argument("--gt_data_root", required=True, help="Root directory for Ground Truth images")
    parser.add_argument("--output_base_dir", required=True, help="API 推理结果保存目录")

    parser.add_argument("--api_key", type=str, required=True, help="Gemini API Key")
    parser.add_argument("--base_url", type=str, required=True, help="Gemini API Base URL (e.g., https://api.gemini.com/v1)")
    parser.add_argument("--models", required=True, help="待评测模型列表，逗号分隔")
    parser.add_argument("--model_name", default="gemini-3-pro-preview", help="Judge model name")
    parser.add_argument("--num_workers", type=int, default=32, help="并发数")
    parser.add_argument("--debug", action="store_true", help="Debug 模式，只处理前3个样本")

    args = parser.parse_args()

    # Parse model list
    model_list = [m.strip() for m in args.models.split(',') if m.strip()]

    # Initialize wrapper
    wrapper = GeminiInferenceWrapper(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.num_workers
    )

    # Scan total workload
    total_workload = scan_total_workload(args, model_list)
    global_tracker = GlobalProgress(total_workload)

    # Iterate over each model
    for model_name in model_list:
        try:
            process_model(model_name, args, wrapper, global_tracker)
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 All models completed!")
    print(f"Final Status: {global_tracker.get_status_str('ALL')}")
    print("="*60)


if __name__ == "__main__":
    main()