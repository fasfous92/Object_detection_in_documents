import torch
import json
import os
from tqdm import tqdm
from PIL import Image
from src.locator import LowLevelLocator

def cache_dataset(json_path="data/ROD/rod_dataset.json", image_dir="data/images"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Locator on {device}...")
    
    # Load Locator
    locator = LowLevelLocator(device=device)
    
    # Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Caching boxes for {len(data)} samples...")
    cached_data = []
    
    for item in tqdm(data):
        # Load Image
        filename = item.get("file_name", f"{item['image_id']}.jpg")
        img_path = os.path.join(image_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Run Locator ONCE
            # We pass the PIL image directly (Locator handles it)
            boxes, _ = locator.get_candidate_boxes(image, item["prompt"])
            
            # Move to CPU and convert to list for JSON saving
            boxes_list = boxes.cpu().tolist()
            
            # If no boxes found, add a dummy one (to avoid crashes)
            if len(boxes_list) == 0:
                boxes_list = [[0.0, 0.0, 1.0, 1.0]]
                
            # Save to item
            item["cached_candidates"] = boxes_list
            cached_data.append(item)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save New JSON
    output_path = "data/ROD/rod_dataset_cached.json"
    with open(output_path, 'w') as f:
        json.dump(cached_data, f, indent=4)
        
    print(f"✓ Caching Complete. Saved to {output_path}")

if __name__ == "__main__":
    cache_dataset()
