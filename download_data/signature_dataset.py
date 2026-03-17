import os
import json
import cv2
import shutil
from roboflow import Roboflow
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Project Settings
PROJ1_WORKSPACE = "personal-cquvu"
PROJ1_NAME = "signature-detection-hlx8j-pvgiq"
PROJ1_VERSION = 1

# Final Output Settings
OUTPUT_DIR = "data_final_signature"
FINAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# 🎯 Target Class Definition
# Only this class will receive bounding box tokens. All others are ignored.
TARGET_CLASS = "signature" 

def setup_dirs():
    if os.path.exists(OUTPUT_DIR): 
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(FINAL_IMG_DIR, exist_ok=True)

def download_project(workspace, project_name, version, local_dir):
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("coco", location=local_dir)
    return dataset.location

def get_data_from_coco(base_path, split_folder):
    """Fetches data and dynamically maps Roboflow category IDs to string labels."""
    samples = []
    path = os.path.join(base_path, split_folder)
    json_p = os.path.join(path, "_annotations.coco.json")
    
    if not os.path.exists(json_p): 
        return samples
        
    with open(json_p, 'r') as f:
        coco = json.load(f)
    
    # Dynamic class mapping
    cat_map = {cat['id']: cat['name'].lower() for cat in coco['categories']}
    
    ann_map = {}
    for ann in coco['annotations']:
        label_name = cat_map.get(ann['category_id'], "unknown")
        ann_map.setdefault(ann['image_id'], []).append({"bbox": ann['bbox'], "cls": label_name})
        
    for img in coco['images']:
        anns = ann_map.get(img['id'], [])
        full_p = os.path.join(path, img['file_name'])
        if os.path.exists(full_p):
            samples.append({"path": full_p, "anns": anns, "name": img['file_name']})
            
    return samples

def process_set(samples, split_name):
    entries = []
    
    for s in tqdm(samples, desc=f"Processing {split_name}"):
        img = cv2.imread(s["path"])
        if img is None: 
            continue
        h, w = img.shape[:2]
        
        fname = f"{split_name}_{s['name']}"
        dst_path = os.path.join(FINAL_IMG_DIR, fname)
        shutil.copy(s["path"], dst_path)
        
        # Build Qwen's Native Spatial String
        response_text = ""
        
        for ann in s["anns"]:
            b = ann["bbox"]
            l = ann["cls"]
            
            # 🚀 THE FIX: Ignore any class that is NOT our target
            if l != TARGET_CLASS.lower():
                continue
                
            # 1. COCO format to Absolute Pixels
            x_min_abs = max(0, int(b[0]))
            y_min_abs = max(0, int(b[1]))
            x_max_abs = min(w, int(b[0] + b[2]))
            y_max_abs = min(h, int(b[1] + b[3]))
            
            # 2. Normalize to Qwen's 0-1000 scale
            x_min = max(0, min(1000, int((x_min_abs / w) * 1000)))
            y_min = max(0, min(1000, int((y_min_abs / h) * 1000)))
            x_max = max(0, min(1000, int((x_max_abs / w) * 1000)))
            y_max = max(0, min(1000, int((y_max_abs / h) * 1000)))
            
            # 3. Concatenate target objects using Qwen grounding tokens
            response_text += f"<|object_ref_start|>{l}<|object_ref_end|><|box_start|>({x_min},{y_min}),({x_max},{y_max})<|box_end|>"
            
        # If the response is still empty, it means no target class was found in the image
        if not response_text:
            response_text = "None of the requested objects were found."
            
        prompt_text = f"Detect all {TARGET_CLASS}s. If none are found, respond with: None of the requested objects were found."
                    
        entries.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": os.path.abspath(dst_path)},
                    {"type": "text", "text": prompt_text}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
            ]
        })
        
    with open(os.path.join(OUTPUT_DIR, f"{split_name}.jsonl"), 'w') as f:
        for e in entries: 
            f.write(json.dumps(e) + "\n")

def run():
    setup_dirs()
    # Download directly to a raw holding folder within our output structure
    raw_dl_dir = os.path.join(OUTPUT_DIR, "raw_proj1")
    loc1 = download_project(PROJ1_WORKSPACE, PROJ1_NAME, PROJ1_VERSION, raw_dl_dir)
    
    # Read directly from Roboflow's pre-made splits
    train_data = get_data_from_coco(loc1, "train")
    valid_data = get_data_from_coco(loc1, "valid")
    test_data = get_data_from_coco(loc1, "test")
    
    # Process and build splits
    process_set(train_data, "train")
    process_set(valid_data, "valid")
    process_set(test_data, "test")
    
    # Clean up the raw download to save disk space
    shutil.rmtree(raw_dl_dir)
    
    print("✅ Done! Qwen JSONL files successfully generated with selective negative sampling.")

if __name__ == "__main__":
    run()
