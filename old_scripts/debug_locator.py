import torch
import os
import argparse
import matplotlib
matplotlib.use('Agg') # Safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# --- CONFIG ---
TRAINED_MODEL_PATH = "output_model/clean_locator_owlv2/best_model"
TEST_IMAGE = "data/signatures_augmented/images/test_0.jpg" # <--- Change this to your image
TARGET_SIZE = 768
THRESHOLD = 0.02

# --- 1. HELPER: RESIZE & PAD (Same as Training) ---
def resize_and_pad(image, target_size):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_image.paste(image, (0, 0)) 
    
    return new_image, scale

# --- 2. MAIN INFERENCE ---
def test_locator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basemodel", action="store_true", help="Use base Google model instead of trained one")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Path to image to test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Device: {device}")

    # A. Select Model
    if args.basemodel:
        print(">> [MODE] Using BASE Model (google/owlvit-base-patch32)")
        model_path = "google/owlvit-large-patch14"
    else:
        print(f">> [MODE] Using TRAINED Model ({TRAINED_MODEL_PATH})")
        model_path = TRAINED_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"(!) Error: Trained model not found at {model_path}")
            return

    # B. Load
    try:
        model = OwlViTForObjectDetection.from_pretrained(model_path).to(device)
        processor = OwlViTProcessor.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"(!) Failed to load model: {e}")
        return

    # C. Load Image
    if not os.path.exists(args.image):
        print(f"(!) Error: Image not found at {args.image}")
        return

    raw_image = Image.open(args.image).convert("RGB")
    w_orig, h_orig = raw_image.size
    
    # D. Preprocess (Letterbox)
    input_image, scale = resize_and_pad(raw_image, TARGET_SIZE)
    
    # E. Predict
    # Note: We use "signature" as the prompt. 
    # The base model might detect ANY text as signature, yours should be specific.
    inputs = processor(text=[["signature"]], images=input_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # F. Post-Process
    target_sizes = torch.Tensor([[TARGET_SIZE, TARGET_SIZE]]).to(device)
    results = processor.post_process_object_detection(outputs, threshold=THRESHOLD, target_sizes=target_sizes)[0]
    
    scores = results["scores"].cpu().tolist()
    boxes = results["boxes"].cpu().tolist()
    
    # G. Visualize
    print(f"\n>> Found {len(scores)} candidates (Threshold={THRESHOLD})")
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(raw_image)
    
    found_any = False
    for score, box in zip(scores, boxes):
        # Inverse Transform: Remove Padding & Scale up
        x1 = box[0] / scale
        y1 = box[1] / scale
        x2 = box[2] / scale
        y2 = box[3] / scale
        
        # Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        # Draw
        # Base model (Red) vs Trained (Green) for clarity if you compare screenshots
        color = 'red' if args.basemodel else '#00FF00' 
        
        print(f"   Score: {score:.4f} | Box: {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}")
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, f"{score:.2f}", color='white', backgroundcolor=color, fontsize=10, weight='bold')
        found_any = True

    if not found_any:
        print("   No signatures detected above threshold.")

    # Save
    suffix = "base" if args.basemodel else "trained"
    out_file = f"test_result_{suffix}.png"
    plt.axis('off')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"\n>> Saved visualization to '{out_file}'")

if __name__ == "__main__":
    test_locator()
