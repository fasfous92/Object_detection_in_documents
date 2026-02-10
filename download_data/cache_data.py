import torch
import json
import os
import sys
from tqdm import tqdm
from PIL import Image

# Ensure we can import from src/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.locator import LowLevelLocator
except ImportError:
    # Fallback if running from root
    from utils.locator import LowLevelLocator

def extract_category_from_prompt(prompt):
    """
    Extracts the object name from the prompt for the Locator.
    Input: "Locate <p>the airplane</p>."
    Output: "airplane"
    """
    try:
        # Simple string parsing based on your download script format
        if "<p>the " in prompt and "</p>" in prompt:
            start = prompt.find("<p>the ") + len("<p>the ")
            end = prompt.find("</p>")
            return prompt[start:end]
        return "object" # Generic fallback
    except:
        return "object"

def cache_dataset():
    # --- CONFIG ---
    json_path = "data/ROD/rod_dataset.json"
    image_dir = "data/images"
    output_path = "data/ROD/rod_dataset_cached.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Locator on {device}...")
    
    # Initialize the Locator (OWL-ViT)
    locator = LowLevelLocator(device=device)
    
    print(f"Loading dataset from {json_path}...")
    if not os.path.exists(json_path):
        print("(!) Error: rod_dataset.json not found. Run your download script first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Caching candidates for {len(data)} images...")
    cached_data = []
    
    for item in tqdm(data):
        filename = item["file_name"]
        img_path = os.path.join(image_dir, filename)
        
        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping broken image {filename}: {e}")
            continue

        # 2. Fix Ground Truth Boxes (xywh -> xyxy)
        # Your download script saves [x, y, w, h] (normalized).
        # The training model (ROI Align) expects [x1, y1, x2, y2].
        fixed_boxes = []
        for box in item['boxes']:
            x, y, w, h = box
            # Ensure we don't go out of bounds
            x2 = min(1.0, x + w)
            y2 = min(1.0, y + h)
            fixed_boxes.append([x, y, x2, y2])
        item['boxes'] = fixed_boxes

        # 3. Generate Candidate Boxes (The "Caching" Part)
        # We ask the locator to find regions that look like the target category.
        query_text = extract_category_from_prompt(item['prompt'])
        
        try:
            # Run Locator
            candidates, _ = locator.get_candidate_boxes(image, query_text, threshold=0.05)
            
            # Convert to list for JSON
            candidates_list = candidates.cpu().tolist()
            
            # Fallback if locator finds nothing
            if len(candidates_list) == 0:
                candidates_list = [[0.0, 0.0, 1.0, 1.0]]
            
            item["cached_candidates"] = candidates_list
            cached_data.append(item)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. Save
    with open(output_path, 'w') as f:
        json.dump(cached_data, f, indent=4)
        
    print(f"✓ Caching Complete. Saved processed data to {output_path}")

if __name__ == "__main__":
    cache_dataset()
