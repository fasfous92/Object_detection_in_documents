import torch
import json
import os
import matplotlib
# Force backend to avoid "no display name" errors in notebooks
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from src.model import RODMLLM

# --- HELPERS ---
def get_iou(boxA, boxB):
    # Ensure both boxes are valid [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Add epsilon to prevent division by zero
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def save_performance_plots(results, folder_path, k=5):
    os.makedirs(folder_path, exist_ok=True)
    
    # Sort by IoU (Descending)
    sorted_res = sorted(results, key=lambda x: x['iou'], reverse=True)
    best_k = sorted_res[:k]
    worst_k = sorted_res[-k:]
    
    for title, subset, filename in [("Best Predictions", best_k, "best_predictions.png"), 
                                    ("Worst Predictions", worst_k, "worst_predictions.png")]:
        
        # Create figure
        fig, axes = plt.subplots(1, k, figsize=(k * 4, 5))
        fig.suptitle(f"{title} (Green=Truth, Red=Pred)", fontsize=16)
        if k == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            if i >= len(subset): 
                ax.axis('off')
                continue
                
            item = subset[i]
            img = item['image']
            w, h = img.size
            ax.imshow(img)
            
            # Ground Truth (Green) - Already Normalized
            if item.get('gt_box'):
                gx1, gy1, gx2, gy2 = item['gt_box']
                rect_g = patches.Rectangle((gx1*w, gy1*h), (gx2-gx1)*w, (gy2-gy1)*h,
                                         linewidth=3, edgecolor='#00FF00', facecolor='none')
                ax.add_patch(rect_g)
                
            # Prediction (Red) - Already Normalized
            if item.get('pred_box'):
                px1, py1, px2, py2 = item['pred_box']
                rect_r = patches.Rectangle((px1*w, py1*h), (px2-px1)*w, (py2-py1)*h,
                                         linewidth=3, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect_r)
            
            ax.set_title(f"IoU: {item['iou']:.2f}\n{item['filename'][-15:]}")
            ax.axis('off')
        
        plt.tight_layout()
        save_file = os.path.join(folder_path, filename)
        plt.savefig(save_file)
        plt.close()
        print(f"   Saved plot: {save_file}")

# --- MAIN EVALUATION ---
def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- UPDATED CONFIG FOR AUGMENTED DATA ---
    # Use the last epoch you trained (e.g., epoch 5)
    checkpoint_path = "output_model/signature_epoch1.pt" 
    
    # Use the augmented paths we created
    test_json = "data/signatures_augmented/rod_test.json"
    image_base = "data/signatures_augmented/images"
    
    print(f">> Loading model from {checkpoint_path}...")
    model = RODMLLM(device=device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle loading weights that might have prefixes from compilation
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
        print("   Weights loaded successfully.")
    else:
        print(f"(!) WARNING: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.eval()
    
    if not os.path.exists(test_json):
        print(f"(!) Error: Test JSON not found at {test_json}")
        return

    with open(test_json, 'r') as f:
        data = json.load(f)
    
    results = []
    print(f">> Running Inference on {len(data)} images...")
    
    for item in tqdm(data):
        file_name = item['file_name']
        img_path = os.path.join(image_base, file_name)
        
        if not os.path.exists(img_path): 
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        w, h = image.size
        gt_boxes = item['boxes'] # Normalized [0-1]
        
        # --- PREDICTION ---
        prompt = "Locate the signature."
        
        # Get raw boxes from Locator (Returns Absolute Pixels)
        pred_boxes, scores = model.locator_module.get_candidate_boxes(image, prompt)
        
        best_iou = 0.0
        best_pred_norm = None
        target_gt = gt_boxes[0] if len(gt_boxes) > 0 else None
        
        if len(pred_boxes) > 0:
            # Take Top-1 Prediction
            pred_abs = pred_boxes[0].cpu().tolist() # Absolute [x1, y1, x2, y2]
            
            # --- NORMALIZE PREDICTION ---
            pred_norm = [
                pred_abs[0] / w,
                pred_abs[1] / h,
                pred_abs[2] / w,
                pred_abs[3] / h
            ]
            best_pred_norm = pred_norm
            
            # Compare against all GT boxes
            if target_gt:
                for gt in gt_boxes:
                    iou = get_iou(pred_norm, gt)
                    if iou > best_iou:
                        best_iou = iou
                        target_gt = gt
        
        results.append({
            "filename": file_name,
            "image": image,
            "pred_box": best_pred_norm,
            "gt_box": target_gt,
            "iou": best_iou
        })

    if len(results) == 0:
        print("No results generated. Check paths.")
        return

    avg_iou = sum(r['iou'] for r in results) / len(results)
    print(f"\n>> EVALUATION COMPLETE")
    print(f"   Mean IoU: {avg_iou:.4f}")
    
    save_performance_plots(results, "results")

if __name__ == "__main__":
    evaluate()
