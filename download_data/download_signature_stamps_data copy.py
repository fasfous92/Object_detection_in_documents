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

# The source project containing the signatures
PROJ_WORKSPACE = "personal-cquvu"
PROJ_NAME = "yolo5tst123-irqzg"
PROJ_VERSION = 3

# Final Output Settings
OUTPUT_DIR = "negative_examples_export"
FINAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
ZIP_NAME = "negative_signatures_for_roboflow"

# The exact class name of the signatures in your source dataset
TARGET_CLASS = "signature" 

def setup_dirs():
    if os.path.exists(OUTPUT_DIR): 
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(FINAL_IMG_DIR, exist_ok=True)

def download_project(workspace, project_name, version, local_dir):
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(workspace).project(project_name)
    # Downloading as COCO is best for easily parsing the JSON structure
    dataset = project.version(version).download("coco", location=local_dir)
    return dataset.location

def get_target_images_from_coco(base_path, split_folder, target_class):
    """Fetches images from COCO JSON that explicitly contain the target class."""
    samples = []
    path = os.path.join(base_path, split_folder)
    json_p = os.path.join(path, "_annotations.coco.json")
    
    if not os.path.exists(json_p): 
        return samples
        
    with open(json_p, 'r') as f:
        coco = json.load(f)
    
    # Map category IDs to string labels
    cat_map = {cat['id']: cat['name'].lower() for cat in coco['categories']}
    
    # Map image IDs to a list of class labels present in that image
    ann_map = {}
    for ann in coco['annotations']:
        label_name = cat_map.get(ann['category_id'], "unknown")
        ann_map.setdefault(ann['image_id'], []).append(label_name)
        
    for img in coco['images']:
        labels = ann_map.get(img['id'], [])
        
        # Filter logic: Only keep the image if it contains our target class
        if target_class.lower() in labels:
            full_p = os.path.join(path, img['file_name'])
            if os.path.exists(full_p):
                samples.append({"path": full_p, "name": img['file_name']})
                
    return samples

def process_split(samples, split_name):
    for s in tqdm(samples, desc=f"Extracting {split_name} negatives"):
        # Validate image integrity
        img = cv2.imread(s["path"])
        if img is None: 
            continue
        
        # Prefix the filename with the split to prevent accidental overwriting
        fname = f"{split_name}_{s['name']}"
        dst_path = os.path.join(FINAL_IMG_DIR, fname)
        
        # Copy ONLY the image. We drop the bounding box data entirely.
        shutil.copy(s["path"], dst_path)

def run():
    setup_dirs()
    raw_dl_dir = os.path.join(OUTPUT_DIR, "raw_proj")
    
    print("Downloading source project...")
    loc1 = download_project(PROJ_WORKSPACE, PROJ_NAME, PROJ_VERSION, raw_dl_dir)
    
    # Extract image paths across all splits
    train_data = get_target_images_from_coco(loc1, "train", TARGET_CLASS)
    valid_data = get_target_images_from_coco(loc1, "valid", TARGET_CLASS)
    test_data = get_target_images_from_coco(loc1, "test", TARGET_CLASS)
    
    total_found = len(train_data) + len(valid_data) + len(test_data)
    print(f"Found {total_found} images containing '{TARGET_CLASS}'.")
    
    if total_found == 0:
        print("No images found. Check if TARGET_CLASS matches the exact label in your dataset.")
        return

    # Process and copy the raw images
    process_split(train_data, "train")
    process_split(valid_data, "valid")
    process_split(test_data, "test")
    
    # Zip the directory containing only images
    print("\nZipping negative examples...")
    shutil.make_archive(ZIP_NAME, 'zip', FINAL_IMG_DIR)
    
    # Cleanup temporary directories
    shutil.rmtree(raw_dl_dir)
    shutil.rmtree(OUTPUT_DIR) 
    
    print(f"✅ Success! Generated {ZIP_NAME}.zip.")
    print("Drag and drop this ZIP directly into your new Roboflow stamp project.")

if __name__ == "__main__":
    run()
