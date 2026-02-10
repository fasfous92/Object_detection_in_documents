import torch
import os
import json
import matplotlib
matplotlib.use('Agg') # Safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from src.model import RODMLLM

# --- CONFIG ---
# Path to the BEST model saved by your clean training script
LOCATOR_PATH = "output_model/expert_locator_owlv2"
TARGET_SIZE = 768 # Must match training size

# --- 1. HELPER: RESIZE & PAD (CRITICAL: Matches Training Logic) ---
def resize_and_pad(image, target_size):
    """
    Resizes image to fit within target_size x target_size while keeping aspect ratio.
    Pads the rest with black. Matches 'train_locator_clean.py'.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_image.paste(image, (0, 0)) # Top-Left Alignment
    
    return new_image, scale

def get_ground_truth_box(image_filename, annotation_files):
    for json_path in annotation_files:
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item["file_name"] == image_filename and len(item["boxes"]) > 0:
                # Assuming dataset has normalized boxes [x, y, w, h] or [x1, y1, x2, y2]
                # We will check visualization to confirm.
                return item["boxes"][0]
    return None

def debug_single_image(image_path, annotation_files):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize Base RODMLLM
    print(f">> Initializing RODMLLM...")
    model = RODMLLM(device=device)
    
    # 2. INJECT YOUR TRAINED LOCATOR
    # This overwrites the default "google/owlvit-base-patch32" weights
    if os.path.exists(LOCATOR_PATH):
        print(f">> Loading FINE-TUNED Locator from: {LOCATOR_PATH}")
        try:
            # Load your trained weights
            trained_locator = OwlViTForObjectDetection.from_pretrained(LOCATOR_PATH)
            trained_processor = OwlViTProcessor.from_pretrained(LOCATOR_PATH)
            
            # Swap them into the RODMLLM class
            model.locator_module.model = trained_locator.to(device)
            model.locator_module.processor = trained_processor
            print("   [✓] Locator weights injected successfully.")
        except Exception as e:
            print(f"   [!] Failed to inject locator: {e}")
            return
    else:
        print(f"(!) ERROR: Trained locator not found at {LOCATOR_PATH}")
        print("    Run 'src/train_locator_clean.py' first.")
        return

    model.eval()

    # 3. Load & Preprocess Image
    if not os.path.exists(image_path):
        print(f"(!) Error: Image not found at {image_path}")
        return

    print(f">> Processing image: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    w_orig, h_orig = raw_image.size

    # --- APPLY LETTERBOX TRANSFORM ---
    # We cannot just call model.locator_module.get_candidate_boxes() directly 
    # because it might use default resizing. We must do it manually to match training.
    input_image, scale = resize_and_pad(raw_image, TARGET_SIZE)

    # 4. Run Inference
    print(">> Running Inference (with Letterbox padding)...")
    inputs = model.locator_module.processor(
        text=[["signature"]], 
        images=input_image, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model.locator_module.model(**inputs)
    
    # Post-Process (on 768x768 canvas)
    target_sizes = torch.Tensor([[TARGET_SIZE, TARGET_SIZE]]).to(device)
    results = model.locator_module.processor.post_process_object_detection(
        outputs, threshold=0.05, target_sizes=target_sizes
    )[0]
    
    scores = results["scores"].cpu().tolist()
    boxes = results["boxes"].cpu().tolist() # [x1, y1, x2, y2]
    
    # 5. Visualize
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(raw_image) # Show ORIGINAL image
    
    # Draw Ground Truth (Green)
    gt_box = get_ground_truth_box(os.path.basename(image_path), annotation_files)
    if gt_box:
        # Assuming GT is [x, y, w, h] normalized? Or [x1, y1, x2, y2]?
        # Let's try drawing it assuming normalized [x, y, w, h] first.
        # If your data is unnormalized pixels, remove the *w and *h
        gx, gy, gw, gh = gt_box
        
        # Check if Normalized (values <= 1.0)
        if gx <= 1.0 and gw <= 1.0:
            rect_gt = patches.Rectangle((gx*w_orig, gy*h_orig), gw*w_orig, gh*h_orig,
                                      linewidth=3, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect_gt)
            ax.text(gx*w_orig, gy*h_orig-5, "GT", color='lime', weight='bold')
        else:
            # Absolute pixels
            rect_gt = patches.Rectangle((gx, gy), gw, gh,
                                      linewidth=3, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect_gt)

    # Draw Predictions (Red)
    print(f">> Found {len(scores)} predictions:")
    
    for score, box in zip(scores, boxes):
        # --- INVERSE TRANSFORM (Remove Padding) ---
        x1 = box[0] / scale
        y1 = box[1] / scale
        x2 = box[2] / scale
        y2 = box[3] / scale
        
        # Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        print(f"   Score: {score:.4f} | Box: {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}")
        
        rect_pred = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='red', linestyle='--', facecolor='none'
        )
        ax.add_patch(rect_pred)
        ax.text(x1, y1-5, f"{score:.2f}", color='white', backgroundcolor='red', fontsize=9)

    ax.axis('off')
    output_file = "debug_result_loaded.png"
    plt.savefig(output_file)
    print(f"\n>> Saved visualization to '{output_file}'")
    plt.close()

if __name__ == "__main__":
    # Annotations to find GT
    annotation_files = [
        "data/signatures_augmented/rod_test.json"
    ]
    
    # Image to test
    target_image = "data/signatures_augmented/images/test_1.jpg" 
    
    # Check for image existence
    if not os.path.exists(target_image):
        import glob
        files = glob.glob("data/signatures_augmented/images/*.jpg")
        if files: target_image = files[0]

    debug_single_image(target_image, annotation_files)
