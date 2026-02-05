
import os
import json
from roboflow import Roboflow
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient

# --- Configuration ---
# Replace with your actual key or set env var ROBOFLOW_API_KEY
user_secrets = UserSecretsClient()
API_KEY =  user_secrets.get_secret("ROBOFLOW_API_KEY")
WORKSPACE = "tech-ysdkk"
PROJECT = "signature-detection-hlx8j"
VERSION = 3
DOWNLOAD_DIR = "data/signatures_raw"
OUTPUT_DIR = "data/signatures_rod"

def setup_directories():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data():
    """Downloads dataset using code extracted from signature_detection_train.ipynb"""
    print(f">> Downloading dataset {PROJECT} v{VERSION}...")
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE)
    project = workspace.project(PROJECT)
    version = project.version(VERSION)
    
    # Download in COCO format as used in the notebook
    dataset = version.download(
        model_format="coco",
        location=DOWNLOAD_DIR,
        overwrite=True
    )
    return dataset.location

def convert_coco_to_rod(coco_json_path, image_folder_name, split_name):
    """
    Parses the COCO JSON and converts it to ROD-MLLM format.
    """
    print(f"   Processing {split_name} split...")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Map Image IDs to File Names and Dimensions
    images_info = {}
    for img in coco_data['images']:
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # 2. Group Annotations by Image ID
    annotations_map = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_map:
            annotations_map[image_id] = []
        
        # COCO bbox is [x_min, y_min, width, height]
        x, y, w, h = ann['bbox']
        
        # ROD requires Normalized [x1, y1, x2, y2]
        img_w = images_info[image_id]['width']
        img_h = images_info[image_id]['height']
        
        norm_box = [
            x / img_w,
            y / img_h,
            (x + w) / img_w,
            (y + h) / img_h
        ]
        annotations_map[image_id].append(norm_box)

    # 3. Create ROD Entries
    rod_data = []
    for image_id, info in images_info.items():
        # Get boxes (if none, skip or add empty list depending on preference)
        boxes = annotations_map.get(image_id, [])
        
        if not boxes: 
            continue

        # We construct the path relative to where ingestion.py will look
        # ingestion.py typically takes a root dir. We will assume root is 'data/signatures_raw'
        # so file_name should include the subfolder (e.g., "train/image.jpg")
        rel_path = os.path.join(image_folder_name, info['file_name'])

        rod_data.append({
            "image_id": str(image_id),
            "file_name": rel_path,
            "prompt": "<image>\nLocate the signature.", # Standard prompt
            "boxes": boxes,
            "cached_candidates": [[0.0, 0.0, 1.0, 1.0]] # Dummy to skip locator
        })

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"rod_{split_name}.json")
    with open(out_path, 'w') as f:
        json.dump(rod_data, f, indent=4)
    
    print(f"   Saved {len(rod_data)} samples to {out_path}")

def main():
    setup_directories()
    
    # 1. Download
    dataset_location = download_data()
    
    # 2. Process Splits (Train, Valid, Test)
    # The notebook defines these 3 splits explicitly
    splits = [
        ("train", "train"), 
        ("valid", "valid"), 
        ("test", "test")
    ]
    
    for split_folder, split_name in splits:
        json_path = os.path.join(dataset_location, split_folder, "_annotations.coco.json")
        if os.path.exists(json_path):
            convert_coco_to_rod(json_path, split_folder, split_name)
        else:
            print(f"Warning: Could not find annotations for {split_name} at {json_path}")

    print("\n>> Data Preparation Complete.")
    print(f">> Images located in: {dataset_location}")
    print(f">> ROD JSONs located in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
