import os
import json
import cv2
import shutil
import numpy as np
import albumentations as A
from roboflow import Roboflow
from tqdm import tqdm
from glob import glob
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')




# --- CONFIGURATION ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") 
WORKSPACE = "tech-ysdkk"
PROJECT = "signature-detection-hlx8j"
VERSION = 3

# Paths
RAW_DIR = "data/raw_download"
OUTPUT_DIR = "data/signatures_augmented"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# Augmentation Settings
AUGMENT_FACTOR = 3  # How many new images to create per original training image
DO_AUGMENT_TRAIN = True
DO_AUGMENT_VAL = False # Never augment validation/test

# --- AUGMENTATION PIPELINE ---
# specific for Documents (No flipping, focus on scan quality)
aug_pipeline = A.Compose([
    # 1. Pixel-level (Scan Simulation)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4), # Simulates grainy paper
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),    # Simulates bad focus
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3), # JPEG artifacts
    
    # 2. Geometric (Safe deformations)
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
    A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=255, p=0.2), # Slight skew
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.3))

def setup_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

def download_data():
    print(">> Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(model_format="coco", location=RAW_DIR, overwrite=True)
    return dataset.location

def process_split(split_name, json_path, source_img_dir, augment=False):
    print(f"\n>> Processing split: {split_name} (Augment={augment})")
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    # Map Image ID to filename/dims
    img_map = {img['id']: img for img in coco['images']}
    
    # Map Annotations to Image ID
    ann_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map: ann_map[img_id] = []
        ann_map[img_id].append(ann['bbox']) # COCO format: [x, y, w, h]

    rod_entries = []
    
    for img_id, img_info in tqdm(img_map.items()):
        file_name = img_info['file_name']
        src_path = os.path.join(source_img_dir, file_name)
        
        if not os.path.exists(src_path):
            continue
            
        # Load Image
        image = cv2.imread(src_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        bboxes_coco = ann_map.get(img_id, [])
        if not bboxes_coco: continue # Skip empty images
        
        # --- HELPER: Save Entry ---
        def save_entry(img_arr, boxes_coco, suffix=""):
            # Save Image to disk
            new_filename = f"{split_name}_{img_id}{suffix}.jpg"
            save_path = os.path.join(IMG_DIR, new_filename)
            cv2.imwrite(save_path, cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
            
            # Convert Boxes to Normalized [x1, y1, x2, y2]
            boxes_norm = []
            for b in boxes_coco:
                x, y, bw, bh = b
                # Clip to image bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                
                boxes_norm.append([
                    x1 / w, y1 / h, x2 / w, y2 / h
                ])
            
            rod_entries.append({
                "file_name": new_filename,
                "prompt": "<image>\nLocate the signature.",
                "boxes": boxes_norm,
                "cached_candidates": [[0.0,0.0,1.0,1.0]] # Dummy
            })

        # 1. Save Original
        save_entry(image, bboxes_coco, "")
        
        # 2. Augment (Only if enabled)
        if augment:
            for i in range(AUGMENT_FACTOR):
                try:
                    # Albumentations needs class labels
                    labels = [1] * len(bboxes_coco)
                    transformed = aug_pipeline(image=image, bboxes=bboxes_coco, class_labels=labels)
                    
                    if len(transformed['bboxes']) > 0:
                        save_entry(transformed['image'], transformed['bboxes'], f"_aug{i}")
                except Exception as e:
                    # Skip if augmentation fails (e.g. box transformed out of image)
                    pass
                    
    # Save JSON
    out_json = os.path.join(OUTPUT_DIR, f"rod_{split_name}.json")
    with open(out_json, 'w') as f:
        json.dump(rod_entries, f, indent=4)
    print(f"   Saved {len(rod_entries)} entries to {out_json}")

def main():
    setup_dirs()
    data_path = download_data()
    
    # Process Splits matches Roboflow folder structure
    splits = [
        ("train", "train", DO_AUGMENT_TRAIN),
        ("valid", "valid", DO_AUGMENT_VAL),
        ("test", "test", False)
    ]
    
    for folder, name, do_aug in splits:
        json_path = os.path.join(data_path, folder, "_annotations.coco.json")
        img_source = os.path.join(data_path, folder)
        
        if os.path.exists(json_path):
            process_split(name, json_path, img_source, augment=do_aug)
        else:
            print(f"(!) Missing split: {folder}")

    print("\n>> Data Preparation & Augmentation Complete.")
    print(f">> Dataset ready at: {OUTPUT_DIR}")
    print(">> Use 'data/signatures_augmented/images' as image_dir in training.")

if __name__ == "__main__":
    main()
