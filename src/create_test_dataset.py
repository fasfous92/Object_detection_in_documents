import os
import json
import requests
import zipfile
import shutil
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_file(url, save_path):
    """Helper to download large files."""
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def create_dataset(output_dir="data", limit=200):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    rod_dir = os.path.join(output_dir, "ROD")
    os.makedirs(rod_dir, exist_ok=True)

    # 1. Download Annotations
    anno_zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = "annotations_trainval2017.zip"
    download_file(anno_zip_url, zip_path)

    # 2. Unzip and Load JSON
    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # We only need instances_val2017.json
        zip_ref.extract("annotations/instances_val2017.json", path=".")
    
    anno_file = "annotations/instances_val2017.json"
    print(f"Loading {anno_file}...")
    with open(anno_file, 'r') as f:
        coco = json.load(f)

    # Create Maps for fast lookup
    # category_id -> name (e.g., 1 -> "person")
    cat_map = {c['id']: c['name'] for c in coco['categories']}
    # image_id -> file_name
    img_map = {i['id']: i['file_name'] for i in coco['images']}
    # image_id -> list of annotations
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # 3. Build Dataset
    rod_data = []
    
    # We filter for images that have annotations
    valid_image_ids = list(img_to_anns.keys())
    
    print(f"Found {len(valid_image_ids)} images with annotations. Selecting first {limit}...")
    
    count = 0
    for img_id in tqdm(valid_image_ids):
        if count >= limit:
            break
            
        filename = img_map[img_id]
        anns = img_to_anns[img_id]
        
        # Pick the largest object in the image to be the "main" target
        # (Heuristic to get clear training examples)
        best_ann = max(anns, key=lambda x: x['bbox'][2] * x['bbox'][3])
        category_name = cat_map[best_ann['category_id']]
        bbox = best_ann['bbox'] # [x, y, w, h] (Absolute pixels)
        
        # COCO Bbox is [x, y, w, h] absolute. 
        # Our locator usually expects normalized [0-1] relative to image size.
        # We need image dimensions to normalize.
        # Ideally we load image first, or check coco['images'] metadata.
        img_meta = next(item for item in coco['images'] if item["id"] == img_id)
        width, height = img_meta['width'], img_meta['height']
        
        norm_box = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]
        
        # Download Image
        img_url = f"http://images.cocodataset.org/val2017/{filename}"
        save_path = os.path.join(images_dir, filename)
        
        try:
            # Only download if we don't have it
            if not os.path.exists(save_path):
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                else:
                    continue # Skip if download fails
            
            # Verify image is valid
            try:
                Image.open(save_path).verify()
            except:
                continue

            # Add to dataset
            rod_data.append({
                "image_id": str(img_id),
                "file_name": filename,
                "prompt": f"Locate <p>the {category_name}</p>.",
                "boxes": [norm_box], # List of boxes
                "is_negative": False
            })
            count += 1
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # 4. Save JSON
    json_path = os.path.join(rod_dir, "rod_dataset.json")
    with open(json_path, "w") as f:
        json.dump(rod_data, f, indent=4)
        
    # Cleanup
    if os.path.exists("annotations"):
        shutil.rmtree("annotations")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"\n✓ Dataset created with {len(rod_data)} samples.")
    print(f"  - Saved to: {json_path}")

if __name__ == "__main__":
    create_dataset(limit=4900)
