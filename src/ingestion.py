import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RODDataset(Dataset):
    """
    Unified Dataset loader for ROD-MLLM.
    Handles Stage 1 (Alignment) and Stage 2 (Instruction Tuning).
    Supports: ROD, COCO, Objects365, and Flickr30K Entities.
    """
    def __init__(self, annotation_file, image_dir, transform=None, tokenizer=None):
        """
        Args:
            annotation_file (str): Path to the JSON annotation file.
            image_dir (str): Directory with all images.
            transform (callable, optional): Standard vision transforms (e.g., CLIP preprocessing).
            tokenizer (callable, optional): LLM tokenizer for text formatting.
        """
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Standard CLIP/ViT preprocessing (336px as per paper) [cite: 767]
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((336, 336), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load Image
        image_path = os.path.join(self.image_dir, item['image_id'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # 2. Extract Language Description (The 'Query')
        # Format follows Figure 4: "Locate <p>description</p> in the image." [cite: 661]
        description = item.get('description', '')
        prompt = f"Locate <p>{description}</p> in the image."

        # 3. Extract Ground Truth Bounding Boxes
        # The paper uses normalized coordinates [0, 1000] [cite: 632]
        boxes = item.get('bboxes', []) # List of [x1, y1, x2, y2]
        
        # Handle "Absence" (Non-existent objects) [cite: 682]
        if not boxes:
            target_text = "None"
        else:
            # Format: <box>[<a1><a2>]</box> [cite: 662]
            anchor_tokens = "".join([f"<a{i}>" for i in range(len(boxes))])
            target_text = f"<box>[{anchor_tokens}]</box>"

        return {
            "pixel_values": image,
            "input_ids": prompt,
            "labels": target_text,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4)
        }

def get_dataloader(ann_path, img_dir, batch_size=4, shuffle=True, num_workers=2):
    """
    Factory function to create the DataLoader.
    On Kaggle, keep num_workers low to avoid memory fragmentation.
    """
    dataset = RODDataset(ann_path, img_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )

# --- Example Usage for Kaggle ---
if __name__ == "__main__":
    # Example paths - adjust based on your Kaggle Input setup
    DATA_INPUT = "/kaggle/input/rod-dataset-v1"
    
    loader = get_dataloader(
        ann_path=f"{DATA_INPUT}/annotations/train.json",
        img_dir=f"{DATA_INPUT}/images/",
        batch_size=2 # Small batch size for T4 GPUs
    )
    
    print(f"Loaded {len(loader.dataset)} samples.")
