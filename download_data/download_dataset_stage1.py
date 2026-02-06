import os
import requests
import zipfile
from tqdm import tqdm

# Project structure
DATA_DIR = "data"
SUB_DIRS = ["images", "annotations", "ROD"]

def setup_directories():
    """Create the project's data structure."""
    for folder in SUB_DIRS:
        path = os.path.join(DATA_DIR, folder)
        os.makedirs(path, exist_ok=True)
    print(f"✓ Project structure created in ./{DATA_DIR}")

def download_file(url, destination):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract downloaded zip files."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

def main():
    setup_directories()

    # 1. Download COCO 2017 Annotations (Required for regional alignment)
    coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    coco_ann_dest = os.path.join(DATA_DIR, "annotations/coco_ann.zip")
    
    print("\n--- Downloading COCO Annotations ---")
    if not os.path.exists(os.path.join(DATA_DIR, "annotations/instances_train2017.json")):
        download_file(coco_ann_url, coco_ann_dest)
        extract_zip(coco_ann_dest, os.path.join(DATA_DIR, "annotations"))
    else:
        print("COCO Annotations already exist.")

    # 2. ROD Dataset Placeholder
    # Note: As of CVPR 2025, ensure you have the official link from the authors.
    # Replace the URL below with the official Hugging Face or Project URL.
    print("\n--- Downloading ROD Dataset (500K pairs) ---")
    rod_url = "REPLACE_WITH_OFFICIAL_ROD_DATASET_URL" 
    rod_dest = os.path.join(DATA_DIR, "ROD/rod_dataset.json")
    
    if rod_url != "REPLACE_WITH_OFFICIAL_ROD_DATASET_URL":
        download_file(rod_url, rod_dest)
    else:
        print("(!) Please provide the official ROD dataset URL provided by the paper's repository.")

    # 3. Objects365 Sampling (Section 4.1)
    # The paper samples 100K images from Objects365.
    print("\n--- Note on Objects365 ---")
    print("Objects365 is massive (>500GB). It is recommended to download the ")
    print("specific 100K samples used in the paper manually from the official source.")

if __name__ == "__main__":
    main()
