# wrapper/dataset_handler.py
import json
import os
from torch.utils.data import Dataset

class GroundingDataset(Dataset):
    def __init__(self, jsonl_file, base_image_dir):
        self.base_image_dir = base_image_dir
        with open(jsonl_file, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.base_image_dir, item["image"])
        
        # 1. Create the STRICT Qwen Message Format
        # We do NOT load the image here with PIL. We pass the path.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": (
                            "You are an object detector. Detect the signature. "
                            "Return a strictly valid JSON list. "
                            "Format: [{'box_2d': [xmin, ymin, xmax, ymax], 'label': 'signature'}]. "
                            "Coordinates must be normalized 0-1000."
                        )
                    }
                ]
            }
        ]

        # 2. Get the Label (Assistant Response)
        label_str = item["label"]
        if not isinstance(label_str, str):
            label_str = json.dumps(label_str)

        # Return the raw message list. The Collator will handle the heavy lifting.
        return {
            "messages": messages,
            "assistant_response": label_str
        }
