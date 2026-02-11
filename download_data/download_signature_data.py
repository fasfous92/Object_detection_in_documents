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

# 1. Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "tech-ysdkk"
PROJECT = "signature-detection-hlx8j"
VERSION = 3 

# Split Ratios (Must add up to 1.0)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO  = 0.10

OUTPUT_DIR = "data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_coco")
FINAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# --- 2. AUGMENTATION PIPELINE (Train Only) ---
train_aug_pipeline = A.Compose([
    A.SafeRotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.6),
    A.Perspective(scale=(0.05, 0.1), pad_mode=cv2.BORDER_CONSTANT, pad_val=255, p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 40), p=0.3),
], bbox_params=A.BboxParams(
    format='coco', 
    label_fields=['labels'],
    min_visibility=0.3
))

def setup_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(FINAL_IMG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

# --- USER PROVIDED DOWNLOAD FUNCTION ---
def download_data():
    print(">> Downloading dataset from Roboflow...")
    if not ROBOFLOW_API_KEY:
        print("Error: ROBOFLOW_API_KEY not found.")
        return None

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(model_format="coco", location=RAW_DIR, overwrite=True)
    return dataset.location

def safe_clip_boxes(boxes, img_w, img_h):
    cleaned = []
    for b in boxes:
        x, y, w, h = b
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w - 1, x + w)
        y2 = min(img_h - 1, y + h)
        
        new_w, new_h = x2 - x1, y2 - y1
        if new_w > 1 and new_h > 1:
            cleaned.append([x1, y1, new_w, new_h])
    return cleaned

def aggregate_all_data(base_path):
    """
    Looks inside the downloaded path for ANY train/valid/test folders
    and merges them into a single list of samples.
    """
    print(f">> Aggregating data from {base_path}...")
    all_samples = []
    
    # Roboflow structure usually has 'train', 'valid', 'test' folders
    # But sometimes it's flat. We check common subfolders.
    subfolders = ["train", "valid", "test", "."]
    
    for folder in subfolders:
        current_dir = os.path.join(base_path, folder)
        json_path = os.path.join(current_dir, "_annotations.coco.json")
        
        if not os.path.exists(json_path):
            continue
            
        # Load COCO JSON
        with open(json_path, 'r') as f:
            coco = json.load(f)
            
        # Create a map: image_id -> list of bboxes
        ann_map = {}
        for ann in coco['annotations']:
            ann_map.setdefault(ann['image_id'], []).append(ann['bbox'])
            
        # Iterate images in this split
        for img_info in coco['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            full_img_path = os.path.join(current_dir, file_name)
            
            # Verify image exists
            if os.path.exists(full_img_path):
                boxes = ann_map.get(img_id, [])
                if boxes:
                    all_samples.append({
                        "path": full_img_path,
                        "boxes": boxes,
                        "file_name": file_name
                    })

    print(f"   Total images found: {len(all_samples)}")
    return all_samples

def process_and_save(samples, split_name):
    """
    Augments (if train) and saves to JSONL format.
    """
    entries = []
    is_train = (split_name == "train")
    aug_factor = 3 if is_train else 1
    
    print(f">> Processing {split_name} ({len(samples)} images)...")
    
    for sample in tqdm(samples):
        image = cv2.imread(sample["path"])
        if image is None: continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        raw_boxes = sample["boxes"]
        cleaned_boxes = safe_clip_boxes(raw_boxes, w, h)
        if not cleaned_boxes: continue
        
        for i in range(aug_factor):
            # Keep original on first pass, augment on others (only for train)
            if i == 0 or not is_train:
                final_img, final_boxes = image, cleaned_boxes
                suffix = "orig"
            else:
                try:
                    res = train_aug_pipeline(image=image, bboxes=cleaned_boxes, labels=[1]*len(cleaned_boxes))
                    final_img, final_boxes = res['image'], res['bboxes']
                    suffix = f"aug_{i}"
                except ValueError:
                    continue
            
            if not final_boxes: continue
            
            # Save Image
            base_name = os.path.splitext(sample["file_name"])[0]
            out_name = f"{split_name}_{base_name}_{suffix}.jpg"
            out_path = os.path.join(FINAL_IMG_DIR, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
            
            # Format JSON Output (The target label)
            json_targets = []
            for b in final_boxes:
                bx, by, bw, bh = b
                bbox_2d = [int(bx), int(by), int(bx + bw), int(by + bh)]
                json_targets.append({
                    "bbox_2d": bbox_2d,
                    "label": "signatures"
                })
            
            label_str = json.dumps(json_targets)
            
            entries.append({
                "image": f"images/{out_name}",
                "label": label_str
            })
            
    # Write JSONL
    out_jsonl = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(out_jsonl, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
            
    return len(entries)

def run_pipeline():
    setup_dirs()
    
    # 1. Download Data
    dataset_location = download_data()
    if not dataset_location:
        return

    # 2. Aggregate everything into one list (ignoring original splits)
    all_data = aggregate_all_data(dataset_location)
    
    if not all_data:
        print("Error: No data found after download.")
        return

    # 3. Shuffle and Split Manually
    random.shuffle(all_data)
    
    total = len(all_data)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)
    
    train_samples = all_data[:train_end]
    valid_samples = all_data[train_end:valid_end]
    test_samples  = all_data[valid_end:]
    
    print(f">> Split Plan: Train={len(train_samples)}, Valid={len(valid_samples)}, Test={len(test_samples)}")
    
    # 4. Process and Save
    n_train = process_and_save(train_samples, "train")
    n_valid = process_and_save(valid_samples, "valid")
    n_test  = process_and_save(test_samples, "test")

    print(f"\n>> Pipeline Complete!")
    print(f">> Train: {n_train} (Augmented)")
    print(f">> Valid: {n_valid}")
    print(f">> Test:  {n_test}")

if __name__ == "__main__":
    run_pipeline()
