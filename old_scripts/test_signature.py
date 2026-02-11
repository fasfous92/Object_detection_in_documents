import torch
import json
import os
import re
import argparse
import numpy as np
import matplotlib
# Force backend to avoid "no display name" errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# --- IMPORTS FOR PYTHON 3.9 COMPATIBILITY ---
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 1. METRICS ---
def calculate_iou(boxA, boxB):
    # Calculate Intersection over Union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def save_plots(results, output_dir, k=5):
    os.makedirs(output_dir, exist_ok=True)
    results.sort(key=lambda x: x['iou'], reverse=True)
    
    batches = [("Best Predictions", results[:k], "best.png"), 
               ("Worst Predictions", results[-k:], "worst.png")]
    
    for title, subset, filename in batches:
        if not subset: continue
        
        # Adjust figure size based on number of images
        fig, axes = plt.subplots(1, len(subset), figsize=(len(subset) * 4, 5))
        if len(subset) == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            item = subset[i]
            img = item['image'] # This is the resized PIL image
            ax.imshow(img)
            
            # Draw GT (Green)
            if item['gt_box']:
                g = item['gt_box']
                rect_g = patches.Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                         linewidth=2, edgecolor='#00FF00', facecolor='none', label='GT')
                ax.add_patch(rect_g)
                
            # Draw Pred (Red)
            if item['pred_box']:
                p = item['pred_box']
                rect_p = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1],
                                         linewidth=2, edgecolor='red', linestyle='--', facecolor='none', label='Pred')
                ax.add_patch(rect_p)
            
            ax.set_title(f"IoU: {item['iou']:.2f}\n{item['filename']}", fontsize=10)
            ax.axis('off')
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# --- 2. HELPER: PARSE QWEN OUTPUT ---
def parse_qwen_bbox(text_output, width, height):
    try:
        # Regex to find [ymin, xmin, ymax, xmax]
        matches = re.findall(r"\[([\d,\s\.]+)\]", text_output)
        if not matches: return None
        
        coords = [float(x.strip()) for x in matches[-1].split(',')]
        if len(coords) != 4: return None

        # Qwen Format: [ymin, xmin, ymax, xmax] (0-1000 scale)
        ymin, xmin, ymax, xmax = coords
        
        # Convert to Pixels [x1, y1, x2, y2]
        x1 = (xmin / 1000.0) * width
        y1 = (ymin / 1000.0) * height
        x2 = (xmax / 1000.0) * width
        y2 = (ymax / 1000.0) * height
        
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    except:
        return None

# --- 3. MAIN EVALUATION ---
def evaluate(args):
    print(f">> Loading Base Model: {BASE_MODEL_PATH}")
    
    # 1. Load Base Model (4-bit to save memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa" # Use "sdpa" for Python 3.9/RTX 4000
    )
    
    # 2. Load Adapters (If provided)
    if args.checkpoint:
        print(f">> Loading Trained Adapters from: {args.checkpoint}")
        # This wraps the base model with your fine-tuned weights
        model = PeftModel.from_pretrained(model, args.checkpoint)
        output_dir_name = "evaluation_results_trained"
    else:
        print(">> No checkpoint provided. Running BASELINE model.")
        output_dir_name = "evaluation_results_baseline"

    # 3. Load Processor
    # Use the same resolution config as training (512 if you used Speed Mode)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28)

    # 4. Load Data
    with open(args.test_json, 'r') as f:
        data = json.load(f)
    
    # Optional: Limit data for quick testing
    if args.limit > 0:
        data = data[:args.limit]

    results = []
    print(f">> Evaluating on {len(data)} images...")

    for item in tqdm(data, desc="Testing"):
        file_name = item['file_name']
        img_path = os.path.join(args.image_base, file_name)
        
        if not os.path.exists(img_path): continue
            
        try:
            raw_image = Image.open(img_path).convert("RGB")
            
            # --- IMPORTANT: Match Training Pre-processing ---
            # If you trained with the 512x512 resize hack, we must do it here too
            # to verify the trained model correctly.
            w_orig, h_orig = raw_image.size
            processed_image = raw_image.resize((512, 512))
            w_new, h_new = processed_image.size # 512, 512
        except: continue

        # Construct Prompt
        prompt_text = "Detect the signature. Output JSON [ymin, xmin, ymax, xmax] (0-1000 scale)."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Prepare Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # --- PARSE PREDICTION ---
        # Parse based on the resized image dimensions (512x512)
        pred_box_resized = parse_qwen_bbox(output_text, w_new, h_new)
        
        # Scale prediction back to ORIGINAL image size for IoU calculation
        pred_box_original = None
        if pred_box_resized:
            scale_x = w_orig / w_new
            scale_y = h_orig / h_new
            pred_box_original = [
                pred_box_resized[0] * scale_x,
                pred_box_resized[1] * scale_y,
                pred_box_resized[2] * scale_x,
                pred_box_resized[3] * scale_y,
            ]
            
        # --- PARSE GROUND TRUTH ---
        # GT is already in original image coordinates/ratios
        gt_box = None
        if len(item['boxes']) > 0:
            gx, gy, gw, gh = item['boxes'][0]
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            gt_box = [gx, gy, gx + gw, gy + gh] 
            # If GT is normalized (0-1), multiply by w_orig/h_orig
            # Assuming your JSON is pixels based on previous scripts
            # If it's normalized, uncomment below:
            # gt_box = [c * dim for c, dim in zip(gt_box, [w_orig, h_orig, w_orig, h_orig])]

        # Calculate IoU
        iou = 0.0
        if pred_box_original and gt_box:
            iou = calculate_iou(pred_box_original, gt_box)
            
        results.append({
            "filename": file_name,
            "image": raw_image, # Save original for plotting
            "pred_box": pred_box_original,
            "gt_box": gt_box,
            "iou": iou,
            "output": output_text
        })

    # Summary
    if not results: return

    ious = [r['iou'] for r in results]
    mean_iou = sum(ious) / len(ious)
    acc_50 = sum(1 for x in ious if x >= 0.5) / len(ious)
    
    print("\n" + "="*30)
    print(f" RESULTS ({len(results)} images)")
    print("="*30)
    print(f" Model:         {'Baseline' if not args.checkpoint else args.checkpoint}")
    print(f" Mean IoU:      {mean_iou:.4f}")
    print(f" Accuracy @ 50: {acc_50:.2%}")
    print("="*30)
    
    save_plots(results, output_dir=output_dir_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, default="data/signatures_augmented/rod_test.json")
    parser.add_argument("--image_base", type=str, default="data/signatures_augmented/images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to 'best_model' folder. Leave empty for Baseline.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test images (0 for all)")
    
    args = parser.parse_args()
    evaluate(args)
