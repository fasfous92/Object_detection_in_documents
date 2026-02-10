import torch
import json
import os
import numpy as np
import matplotlib
# Force backend to avoid "no display name" errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from src.model import RODMLLM

# --- CONFIG ---
# Path to your "Best" fine-tuned locator
CHECKPOINT_LOCATOR = "output_model/expert_locator_owlv2" 
# Path to your test set
TEST_JSON = "data/signatures_augmented/rod_test.json" 
# Path to images
IMAGE_BASE = "data/signatures_augmented/images"
# Target size must match training (768)
TARGET_SIZE = 768 

# --- 1. PRE-PROCESSING (Must match Training) ---
def resize_and_pad(image, target_size):
    """
    Resizes image to fit within target_size x target_size while keeping aspect ratio.
    Returns: padded_image, scale_factor
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_image.paste(image, (0, 0)) 
    
    return new_image, scale

# --- 2. METRICS ---
def calculate_iou(boxA, boxB):
    """
    Calculate IoU between two boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def save_plots(results, output_dir="evaluation_results", k=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by IoU (Best to Worst)
    results.sort(key=lambda x: x['iou'], reverse=True)
    
    best_k = results[:k]
    worst_k = results[-k:]
    
    # Define visualization batches
    batches = [("Best Predictions", best_k, "best.png"), 
               ("Worst Predictions", worst_k, "worst.png")]
    
    for title, subset, filename in batches:
        if not subset: continue
        
        fig, axes = plt.subplots(1, len(subset), figsize=(len(subset) * 4, 5))
        if len(subset) == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            item = subset[i]
            img = item['image']
            w, h = img.size
            ax.imshow(img)
            
            # Draw GT (Green)
            if item['gt_box']:
                g = item['gt_box'] # [x1, y1, x2, y2]
                rect_g = patches.Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                         linewidth=3, edgecolor='#00FF00', facecolor='none')
                ax.add_patch(rect_g)
                
            # Draw Pred (Red)
            if item['pred_box']:
                p = item['pred_box'] # [x1, y1, x2, y2]
                rect_p = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1],
                                         linewidth=3, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect_p)
            
            ax.set_title(f"IoU: {item['iou']:.2f}\n{item['filename']}")
            ax.axis('off')
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"   [+] Saved {title} to {output_dir}/{filename}")

# --- 3. MAIN EVALUATION ---
def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Starting Evaluation on {device}...")
    
    # A. Load Model & Processor
    # We load standard classes directly to ensure we have full control over inference
    if os.path.exists(CHECKPOINT_LOCATOR):
        print(f">> Loading Fine-Tuned Locator: {CHECKPOINT_LOCATOR}")
        model = OwlViTForObjectDetection.from_pretrained(CHECKPOINT_LOCATOR).to(device)
        processor = OwlViTProcessor.from_pretrained(CHECKPOINT_LOCATOR)
    else:
        print(f"(!) Error: Checkpoint not found at {CHECKPOINT_LOCATOR}")
        return

    model.eval()

    # B. Load Data
    if not os.path.exists(TEST_JSON):
        print(f"(!) Error: Test JSON not found at {TEST_JSON}")
        return
        
    with open(TEST_JSON, 'r') as f:
        data = json.load(f)
        
    print(f">> Found {len(data)} test images.")
    results = []
    
    # C. Loop
    for item in tqdm(data, desc="Evaluating"):
        file_name = item['file_name']
        img_path = os.path.join(IMAGE_BASE, file_name)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except:
            continue

        w_orig, h_orig = raw_image.size
        
        # --- CRITICAL: Apply Training Transform (Letterbox) ---
        input_image, scale = resize_and_pad(raw_image, TARGET_SIZE)
        
        # --- INFERENCE ---
        inputs = processor(text=[["signature"]], images=input_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process (returns boxes on 768x768 canvas)
        target_sizes = torch.Tensor([[TARGET_SIZE, TARGET_SIZE]]).to(device)
        res = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]
        
        # --- PARSE PREDICTION ---
        pred_box = None
        best_score = -1.0
        
        if len(res['scores']) > 0:
            # Pick best score
            idx = torch.argmax(res['scores'])
            box_padded = res['boxes'][idx].cpu().tolist() # [x1, y1, x2, y2] on 768px
            
            # Inverse Transform: Remove Padding & Scale Back
            px1 = box_padded[0] / scale
            py1 = box_padded[1] / scale
            px2 = box_padded[2] / scale
            py2 = box_padded[3] / scale
            
            # Clip to image boundaries
            pred_box = [
                max(0, px1), max(0, py1), 
                min(w_orig, px2), min(h_orig, py2)
            ]
            
        # --- PARSE GROUND TRUTH ---
        # Assuming JSON is [x, y, w, h] (Standard COCO format)
        gt_box = None
        if len(item['boxes']) > 0:
            gx, gy, gw, gh = item['boxes'][0]
            gt_box = [gx, gy, gx + gw, gy + gh] # Convert to x1, y1, x2, y2

        # --- CALCULATE IoU ---
        iou = 0.0
        if pred_box and gt_box:
            iou = calculate_iou(pred_box, gt_box)
            
        results.append({
            "filename": file_name,
            "image": raw_image, # Store original for plotting
            "pred_box": pred_box,
            "gt_box": gt_box,
            "iou": iou
        })

    # D. Summary
    if not results:
        print("No valid results.")
        return

    ious = [r['iou'] for r in results]
    mean_iou = sum(ious) / len(ious)
    
    # Calculate Accuracy @ IoU Thresholds
    acc_50 = sum(1 for x in ious if x >= 0.5) / len(ious)
    acc_75 = sum(1 for x in ious if x >= 0.75) / len(ious)
    
    print("\n" + "="*30)
    print(f" RESULTS ({len(results)} images)")
    print("="*30)
    print(f" Mean IoU:      {mean_iou:.4f}")
    print(f" Accuracy @ 50: {acc_50:.2%}")
    print(f" Accuracy @ 75: {acc_75:.2%}")
    print("="*30)
    
    save_plots(results)

if __name__ == "__main__":
    evaluate()
