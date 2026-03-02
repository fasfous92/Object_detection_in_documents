import os
import json
import cv2
import shutil
import random
import numpy as np
import albumentations as A
from roboflow import Roboflow
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Project Settings
PROJ1_WORKSPACE = "personal-cquvu"
PROJ1_NAME = "authentications_elements_from_documents"
PROJ1_VERSION = 1

# Final Output Settings
OUTPUT_DIR = "data_final"
FINAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
  
CLASS_MAP = {1: "logo", 2: "signature", 3: "stamp"}

# --- AUGMENTATION ---
train_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 40), p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_visibility=0.3))

def setup_dirs():
    if os.path.exists(OUTPUT_DIR): 
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(FINAL_IMG_DIR, exist_ok=True)

def download_project(workspace, project_name, version, local_dir):
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("coco", location=local_dir)
    return dataset.location

def get_data_from_coco(base_path, split_folder, negatives_only=False):
    """Fetches data exclusively from the specified split folder (train, valid, or test)."""
    samples = []
    path = os.path.join(base_path, split_folder)
    json_p = os.path.join(path, "_annotations.coco.json")
    
    if not os.path.exists(json_p): 
        return samples
        
    with open(json_p, 'r') as f:
        coco = json.load(f)
    
    ann_map = {}
    for ann in coco['annotations']:
        ann_map.setdefault(ann['image_id'], []).append({"bbox": ann['bbox'], "cls": ann['category_id']})
        
    for img in coco['images']:
        anns = ann_map.get(img['id'], [])
        
        # Filter logic: Skip images with annotations if we only want negatives
        if negatives_only and len(anns) > 0: 
            continue
        
        full_p = os.path.join(path, img['file_name'])
        if os.path.exists(full_p):
            samples.append({"path": full_p, "anns": anns, "name": img['file_name']})
            
    return samples

def process_set(samples, split_name):
    entries = []
    is_train = (split_name == "train")
    
    for s in tqdm(samples, desc=f"Processing {split_name}"):
        img = cv2.imread(s["path"])
        if img is None: 
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augs (1x for val/test, 2x for train to boost variety)
        loops = 2 if is_train else 1
        for i in range(loops):
            h, w = img.shape[:2]
            boxes = [a["bbox"] for a in s["anns"]]
            labels = [a["cls"] for a in s["anns"]]
            
            curr_img, curr_boxes, curr_labels = img, boxes, labels
            if is_train and i > 0:
                try:
                    res = train_aug(image=img, bboxes=boxes, labels=labels)
                    curr_img, curr_boxes, curr_labels = res['image'], res['bboxes'], res['labels']
                except: 
                    continue

            # Save File
            fname = f"{split_name}_{i}_{s['name']}"
            cv2.imwrite(os.path.join(FINAL_IMG_DIR, fname), cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
            
            # 🚀 QWEN2.5-VL OPTIMIZATION: Absolute Pixels [xmin, ymin, xmax, ymax]
            res_list = []
            for b, l in zip(curr_boxes, curr_labels):
                # 1. COCO format to Absolute Pixels [x_min, y_min, x_max, y_max]
                x_min_abs = max(0, int(b[0]))
                y_min_abs = max(0, int(b[1]))
                x_max_abs = min(w, int(b[0] + b[2]))
                y_max_abs = min(h, int(b[1] + b[3]))
                
                # 2. Normalize to Qwen's 0-1000 scale
                # We use min/max again just to be absolutely safe against floating point rounding errors
                x_min = max(0, min(1000, int((x_min_abs / w) * 1000)))
                y_min = max(0, min(1000, int((y_min_abs / h) * 1000)))
                x_max = max(0, min(1000, int((x_max_abs / w) * 1000)))
                y_max = max(0, min(1000, int((y_max_abs / h) * 1000)))
                
                # Using the standard 'bbox_2d' key
                res_list.append({
                    "bbox_2d": [x_min, y_min, x_max, y_max], 
                    "label": CLASS_MAP.get(l, "unknown") # Safe fallback if label is missing
                })
            targets = ["signatures", "stamps", "logos"]
            random.shuffle(targets)
            
            # 🚀 PROMPT UPDATED: Explicitly requesting 'bbox_2d' and the correct coordinate order
            prompt_text = f"Detect all {targets[0]}, {targets[1]}, and {targets[2]}. Output a JSON list with 'bbox_2d' [xmin, ymin, xmax, ymax] and 'label'. If none are found, output an empty list []."
                        
            entries.append({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": os.path.abspath(os.path.join(FINAL_IMG_DIR, fname))},
                        {"type": "text", "text": prompt_text}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": json.dumps(res_list)}]}
                ]
            })
            
    with open(os.path.join(OUTPUT_DIR, f"{split_name}.jsonl"), 'w') as f:
        for e in entries: 
            f.write(json.dumps(e) + "\n")

def run():
    setup_dirs()
    loc1 = download_project(PROJ1_WORKSPACE, PROJ1_NAME, PROJ1_VERSION, "data_final/raw_proj1")
    
    # Read directly from Roboflow's pre-made splits
    train_data = get_data_from_coco(loc1, "train", negatives_only=False)
    valid_data = get_data_from_coco(loc1, "valid", negatives_only=False)
    test_data = get_data_from_coco(loc1, "test", negatives_only=False)
    
    # Process and build splits
    process_set(train_data, "train")
    process_set(valid_data, "valid")
    process_set(test_data, "test")
    
    print("Done! Dataset ready for Qwen training.")

if __name__ == "__main__":
    run()
