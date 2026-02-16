import os
import json
from PIL import Image
from torch.utils.data import Dataset

class GroundingDataset(Dataset):
    def __init__(self, jsonl_file: str, base_image_dir: str):
        self.base_image_dir = base_image_dir
        self.data = []
        
        print(f"📂 Loading dataset from: {jsonl_file}")
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"Dataset not found: {jsonl_file}")
            
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ Loaded {len(self.data)} raw samples.")

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- SMART PATH LOGIC START ---
        image_filename = item["image"] # e.g., "images/valid_123.jpg"
        
        # 1. Try joining base_dir + filename (Standard)
        path_attempt_1 = os.path.join(self.base_image_dir, image_filename)
        
        # 2. Try removing "images/" from the filename if base_dir already has it
        # (Fixes the ./data/images/images/... double folder issue)
        clean_filename = image_filename.replace("images/", "").replace("images\\", "")
        path_attempt_2 = os.path.join(self.base_image_dir, clean_filename)
        
        # 3. Try looking inside an 'images' subfolder if base_dir is root
        path_attempt_3 = os.path.join(self.base_image_dir, "images", clean_filename)

        final_image_path = None
        if os.path.exists(path_attempt_1):
            final_image_path = path_attempt_1
        elif os.path.exists(path_attempt_2):
            final_image_path = path_attempt_2
        elif os.path.exists(path_attempt_3):
            final_image_path = path_attempt_3
            
        if final_image_path is None:
            # Print helpful error only once per missing file
            print(f"❌ Error: File not found. Tried: \n1.{path_attempt_1}\n2.{path_attempt_2}")
            return None
        # --- SMART PATH LOGIC END ---

        try:
            image = Image.open(final_image_path).convert("RGB")
            img_w, img_h = image.size
        except Exception as e:
            print(f"⚠️ Corrupt Image {final_image_path}: {e}")
            return None
        

        # 2. Parse the "Label" String
        # Your format: "label": "[{\"bbox_2d\": [378, 121, ...], ...}]"
        try:
            # It's a string, so we deserialize it into a Python List
            raw_labels = json.loads(item["label"])
        except:
            print(f"⚠️ Error parsing label string for {item['image']}")
            return None

        # 3. Normalize Coordinates (Pixels -> 0-1000)
        final_boxes = []
        
        for obj in raw_labels:
            # Your data: [xmin, ymin, xmax, ymax] in absolute pixels
            box = obj["bbox_2d"]
            
            # MATH: (Value / Total_Dimension) * 1000
            # We assume box is [xmin, ymin, xmax, ymax]
            norm_box = [
                int((box[0] / img_w) * 1000),
                int((box[1] / img_h) * 1000),
                int((box[2] / img_w) * 1000),
                int((box[3] / img_h) * 1000)
            ]

            # Safety Clip (ensure we don't go <0 or >1000 due to rounding)
            norm_box = [max(0, min(1000, x)) for x in norm_box]

            final_boxes.append({
                "box_2d": norm_box,
                "label": "signature" # Or use obj['label'] if reliable
            })

        # 4. Construct the Prompt Text
        # We format the answer as a JSON string so the model learns to output JSON
        user_text = "Detect the bounding box of the signature."
        assistant_text = json.dumps(final_boxes)

        return {
            "image": image,
            "user_prompt": user_text,
            "assistant_response": assistant_text
        }
