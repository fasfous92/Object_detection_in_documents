import os
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def create_tiny_dataset(output_dir="data"):
    """
    Downloads 10 real images from COCO and creates a mini rod_dataset.json
    compatible with our training script.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    rod_dir = os.path.join(output_dir, "ROD")
    os.makedirs(rod_dir, exist_ok=True)
    
    # 1. Define 10 Real Samples (Images + Prompts + Boxes)
    # Format: [Image_URL, Filename, Prompt, Box [x, y, w, h]]
    # Note: Boxes are normalized [0-1] for simplicity in this test generator
    samples = [
        # Cat image
        ("http://images.cocodataset.org/val2017/000000039769.jpg", "000000039769.jpg", 
         "Locate <p>the cats</p>.", [0.1, 0.1, 0.8, 0.8]), 
        # Horse
        ("http://images.cocodataset.org/val2017/000000019623.jpg", "000000019623.jpg",
         "Find <p>the person</p>.", [0.3, 0.2, 0.2, 0.5]),
        # Train
        ("http://images.cocodataset.org/val2017/000000255393.jpg", "000000255393.jpg",
         "Locate <p>the train</p>.", [0.1, 0.3, 0.8, 0.5]),
        # Bus
        ("http://images.cocodataset.org/val2017/000000122765.jpg", "000000122765.jpg",
         "Where is <p>the bus</p>?", [0.1, 0.2, 0.7, 0.6]),
        # Zebra
        ("http://images.cocodataset.org/val2017/000000008690.jpg", "000000008690.jpg",
         "Detect <p>the zebra</p>.", [0.2, 0.3, 0.5, 0.5]),
    ]
    
    json_data = []
    
    print(f"Downloading {len(samples)} images for Tiny Dataset...")
    
    for url, filename, prompt, box in tqdm(samples):
        # Download Image
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(os.path.join(images_dir, filename))
            
            # Create Annotation Entry
            # We construct a fake "None" negative sample for variety if needed, 
            # but for now let's stick to positives to ensure gradients flow.
            entry = {
                "image_id": filename.split(".")[0],
                "file_name": filename,
                "prompt": prompt,
                "boxes": [box], # List of boxes
                "is_negative": False
            }
            json_data.append(entry)
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    # Save JSON
    json_path = os.path.join(rod_dir, "rod_dataset.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"\n✓ Tiny Dataset created at {output_dir}")
    print(f"  - Images: {len(json_data)}")
    print(f"  - Annotation File: {json_path}")

if __name__ == "__main__":
    create_tiny_dataset()
