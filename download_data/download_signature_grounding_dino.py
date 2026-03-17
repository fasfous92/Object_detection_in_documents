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
PROJ1_NAME = "authentications_elements_from_documents"
PROJ1_VERSION = 1

# Final Output Settings
OUTPUT_DIR = "data_grounding_dino"
FINAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

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
    """Fetches data and dynamically maps Roboflow category IDs to string labels."""
    samples = []
    path = os.path.join(base_path, split_folder)
    json_p = os.path.join(path, "_annotations.coco.json")
    
    if not os.path.exists(json_p): 
        return samples
        
    with open(json_p, 'r') as f:
        coco = json.load(f)
    
    cat_map = {cat['id']: cat['name'].lower() for cat in coco['categories']}
    
    ann_map = {}
    for ann in coco['annotations']:
        label_name = cat_map.get(ann['category_id'], "unknown")
        ann_map.setdefault(ann['image_id'], []).append({"bbox": ann['bbox'], "cls": label_name})
        
    for img in coco['images']:
        anns = ann_map.get(img['id'], [])
        
        if negatives_only and len(anns) > 0: 
            continue
        
        full_p = os.path.join(path, img['file_name'])
        if os.path.exists(full_p):
            samples.append({"path": full_p, "anns": anns, "name": img['file_name']})
            
    return samples

def process_set(samples, split_name):
    entries = []
    
    # Grounding DINO requires categories separated by periods.
    # We use a static prompt so the model learns to identify these specific classes every time.
    dino_prompt = "signature . stamp . logo ."
    
    for s in tqdm(samples, desc=f"Processing {split_name}"):
        img = cv2.imread(s["path"])
        if img is None: 
            continue
        h, w = img.shape[:2]
        
        boxes = [a["bbox"] for a in s["anns"]]
        labels = [a["cls"] for a in s["anns"]]
        
        fname = f"{split_name}_{s['name']}"
        dst_path = os.path.join(FINAL_IMG_DIR, fname)
        shutil.copy(s["path"], dst_path)
        
        # Grounding DINO OPTIMIZATION: Keep absolute pixel coordinates [xmin, ymin, xmax, ymax]
        res_list = []
        for b, l in zip(boxes, labels):
            # COCO format is [x_min, y_min, width, height]
            x_min = max(0, int(b[0]))
            y_min = max(0, int(b[1]))
            x_max = min(w, int(b[0] + b[2]))
            y_max = min(h, int(b[1] + b[3]))
            
            res_list.append({
                "bbox": [x_min, y_min, x_max, y_max], 
                "label": l.lower() # Ensure label strictly matches the lowercase prompt
            })
                    
        # Create a clean, flat JSON structure suitable for standard PyTorch/HF dataloaders
        entries.append({
            "image_path": os.path.abspath(dst_path),
            "width": w,
            "height": h,
            "prompt": dino_prompt,
            "annotations": res_list
        })
        
    # Save as JSONL
    with open(os.path.join(OUTPUT_DIR, f"{split_name}.jsonl"), 'w') as f:
        for e in entries: 
            f.write(json.dumps(e) + "\n")

def run():
    setup_dirs()
    
    print("⏳ Downloading dataset from Roboflow...")
    loc1 = download_project(PROJ1_WORKSPACE, PROJ1_NAME, PROJ1_VERSION, "data_grounding_dino/raw_proj1")
    
    print("⏳ Parsing COCO annotations...")
    train_data = get_data_from_coco(loc1, "train", negatives_only=False)
    valid_data = get_data_from_coco(loc1, "valid", negatives_only=False)
    test_data = get_data_from_coco(loc1, "test", negatives_only=False)
    
    print("⚙️ Building Grounding DINO JSONL splits...")
    process_set(train_data, "train")
    process_set(valid_data, "valid")
    process_set(test_data, "test")
    
    print("🧹 Cleaning up temporary files...")
    shutil.rmtree("data_grounding_dino/raw_proj1")
    
    print("✅ Done! Dataset beautifully formatted for Grounding DINO.")

if __name__ == "__main__":
    run()
