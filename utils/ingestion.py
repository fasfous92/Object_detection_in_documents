import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class RODDataset(Dataset):
    def __init__(self, annotation_file, image_folder, processor=None):
        self.image_folder = image_folder
        self.processor = processor
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item["file_name"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))

        # Return raw data; Collator will handle processing
        return {
            "image": image,
            "boxes": torch.tensor(item["boxes"]),
            "prompt": item["prompt"],
            "file_name": item["file_name"]
        }

class RealDataCollator:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        # If processor is not passed, we assume the model handles it or standard transforms
        self.processor = processor 

    def __call__(self, batch):
        # 1. Prepare Images (CLIP requires specific resizing/normalization)
        images = [item["image"] for item in batch]
        
        # We manually apply standard CLIP transforms if processor isn't available
        # But ideally, we use the processor from the CLIP model.
        # For this script, we assume the model logic handles pixel_values via a processor.
        # Here is a generic fallback that works for most CLIP models:
        try:
            from transformers import CLIPProcessor
            # Use a default processor if none provided (standard for OpenAI CLIP)
            proc = self.processor if self.processor else CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            pixel_values = proc(images=images, return_tensors="pt").pixel_values
        except:
            # Absolute fallback if transformers fails (rare)
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((336, 336)),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
            pixel_values = torch.stack([transform(img) for img in images])

        # 2. Prepare Text (Input IDs & Labels)
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        input_ids = encodings.input_ids
        
        # Create Labels (same as input_ids for Causal LM, ignoring pad)
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # 3. Pass through other metadata
        boxes = [item["boxes"] for item in batch]
        filenames = [item["file_name"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "boxes": boxes,     # List of tensors (variable length per image)
            "prompts": prompts, # List of strings
            "filenames": filenames
        }

class RODDataset(Dataset):
    def __init__(self, annotation_file, image_dir, tokenizer=None):
        self.image_dir = image_dir
        
        # Load the JSON annotations
        with open(annotation_file, "r") as f:
            self.data = json.load(f)
            
        print(f"Loaded {len(self.data)} samples from {annotation_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Image Path Handling
        filename = item.get("file_name", f"{item['image_id']}.jpg")
        img_path = os.path.join(self.image_dir, filename)
        
        # Robust check to prevent crashes
        if not os.path.exists(img_path):
             # Try appending .jpg if missing
             if os.path.exists(img_path + ".jpg"):
                 img_path += ".jpg"
             else:
                 # Return a dummy black image if file is truly missing (prevents crash)
                 print(f"(!) Warning: Image missing {img_path}")
                 image = Image.new("RGB", (336, 336), (0, 0, 0))
                 pixel_values = torch.zeros((3, 336, 336)) 
                 # We still need to return valid types
                 return {
                    "pixel_values": pixel_values,
                    "prompt": item["prompt"], 
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "raw_image": image
                 }

        # 2. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"(!) Error loading image {img_path}: {e}")
            image = Image.new("RGB", (336, 336), (0, 0, 0))

        # 3. Transform Image (CLIP Standard)
        image = image.resize((336, 336))
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])
        pixel_values = transform(image)
        
        # --- THE FIX: Ensure <image> token exists ---
        raw_prompt = item["prompt"]
        if "<image>" not in raw_prompt:
            prompt = f"<image>\n{raw_prompt}"
        else:
            prompt = raw_prompt
        # --------------------------------------------

        # 5. Retrieve Cached Boxes
        cached_boxes = item.get("cached_candidates", [[0.0, 0.0, 1.0, 1.0]])
        boxes_tensor = torch.tensor(cached_boxes, dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "prompt": prompt, 
            "boxes": boxes_tensor, 
            "raw_image": image 
        }
