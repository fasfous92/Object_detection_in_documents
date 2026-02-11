import torch
import argparse
import os
import json
import re
import matplotlib
matplotlib.use('Agg') # Safe for servers/headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 1. COORDINATE PARSING ---
def extract_boxes_from_text(text, img_w, img_h):
    """
    Parses Qwen's <box>(y1,x1),(y2,x2)</box> format.
    Returns a list of [x1, y1, x2, y2] in absolute pixels.
    """
    pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
    matches = re.findall(pattern, text)
    
    boxes = []
    for m in matches:
        # Qwen uses (y, x) ordering on 0-1000 scale
        y1, x1, y2, x2 = map(int, m)
        
        # Convert to Pixels [x, y, x, y]
        px1 = (x1 / 1000.0) * img_w
        py1 = (y1 / 1000.0) * img_h
        px2 = (x2 / 1000.0) * img_w
        py2 = (y2 / 1000.0) * img_h
        
        boxes.append([
            min(px1, px2), min(py1, py2), 
            max(px1, px2), max(py1, py2)
        ])
    return boxes

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# --- 2. VISUALIZATION ---
def save_comparison_plot(results, filename, title):
    if not results: return
    
    n = min(5, len(results))
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1: axes = [axes]
    
    for i in range(n):
        ax = axes[i] if n > 1 else axes[0]
        item = results[i]
        
        ax.imshow(item['image'])
        
        # Draw GT (Green)
        for box in item['gt_boxes']:
            w, h = box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((box[0], box[1]), w, h, 
                                   linewidth=2, edgecolor='#00FF00', facecolor='none', label='GT')
            ax.add_patch(rect)
            
        # Draw Pred (Red)
        for box in item['pred_boxes']:
            w, h = box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((box[0], box[1]), w, h, 
                                   linewidth=2, edgecolor='red', linestyle='--', facecolor='none', label='Pred')
            ax.add_patch(rect)
            
        ax.set_title(f"IoU: {item['iou']:.2f}", fontsize=12)
        ax.axis('off')
    
    # Legend on last plot
    if n > 1: axes[-1].legend(loc='lower right')
    else: axes[0].legend(loc='lower right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

# --- 3. MAIN EVALUATION LOOP ---
def evaluate(args):
    print(f">> Loading Base Model: {BASE_MODEL_PATH}")
    
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)

    # 2. Load Model
    print(">> Loading Model (using 'sdpa' attention)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,       # FIXED: Changed from torch_dtype to dtype
        attn_implementation="sdpa", # FIXED: "sdpa" is native torch attention (stable & fast)
        device_map="auto",
    )
    
    # Load Adapters
    if args.checkpoint:
        print(f">> Loading Adapters from: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
    
    model.eval()

    # 3. Load Data
    print(f">> Reading JSONL: {args.test_jsonl}")
    if not os.path.exists(args.test_jsonl):
        print(f"Error: File not found at {args.test_jsonl}")
        return

    with open(args.test_jsonl, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        data = data[:args.limit]
        print(f"   [!] Limiting to first {args.limit} images.")

    results = []

    print(">> Starting Inference...")
    for item in tqdm(data):
        img_path = os.path.join(args.base_data_path, item['image'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
        except:
            print(f"Missing image: {img_path}")
            continue

        # A. Get Ground Truth
        gt_text = item['conversations'][1]['value']
        gt_boxes = extract_boxes_from_text(gt_text, w, h)

        # B. Get Prediction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Locate the signature."},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        pred_boxes = extract_boxes_from_text(pred_text, w, h)

        # C. Calculate Metrics
        best_iou = 0.0
        if pred_boxes and gt_boxes:
            best_iou = calculate_iou(pred_boxes[0], gt_boxes[0])
        elif not pred_boxes and not gt_boxes:
            best_iou = 1.0 

        results.append({
            "id": item['id'],
            "image": image,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "iou": best_iou
        })

    # 4. Summary
    if not results:
        print("No results generated.")
        return

    ious = [r['iou'] for r in results]
    mean_iou = sum(ious) / len(ious)
    
    print("\n" + "="*30)
    print(f" EVALUATION RESULTS")
    print(f" Mean IoU: {mean_iou:.4f}")
    print("="*30)
    
    results.sort(key=lambda x: x['iou'], reverse=True)
    
    save_comparison_plot(results[:5], "comparison_best.png", "Best Predictions")
    save_comparison_plot(results[-5:], "comparison_worst.png", "Worst Predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output_model/qwen_grounding_specialist/best_model")
    parser.add_argument("--base_data_path", type=str, default="data/qwen_signatures_ready")
    parser.add_argument("--test_jsonl", type=str, default="data/qwen_signatures_ready/test.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    
    args = parser.parse_args()
    evaluate(args)
