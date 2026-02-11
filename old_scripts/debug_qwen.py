import torch
import json
import os
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
# Path to your images and JSON
TEST_JSON = "data/signatures_augmented/rod_test.json" 
IMAGE_BASE = "data/signatures_augmented/images"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    print(f">> Loading {MODEL_PATH} for DEBUGGING...")
    
    # 1. Load Model (Python 3.9 Compatible)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 2. Load Processor
    # Min/Max pixels help manage VRAM and ensure the model sees the image clearly
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)

    # 3. Load Data
    if not os.path.exists(TEST_JSON):
        print(f"ERROR: {TEST_JSON} not found.")
        return
        
    with open(TEST_JSON, 'r') as f:
        data = json.load(f)
    
    # --- DEBUG LOOP: ONLY PROCESS FIRST 3 IMAGES ---
    print(f">> Testing first 3 images out of {len(data)}...")
    
    for i, item in enumerate(data[:3]):
        file_name = item['file_name']
        img_path = os.path.join(IMAGE_BASE, file_name)
        
        if not os.path.exists(img_path):
            print(f"Skipping {file_name} (Not found)")
            continue
            
        print(f"\n" + "="*40)
        print(f" IMAGE {i+1}: {file_name}")
        print("="*40)
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        print(f"Original Size: {w} x {h}")
        
        # Print Ground Truth
        if len(item['boxes']) > 0:
            # GT is usually [x, y, w, h]
            gx, gy, gw, gh = item['boxes'][0]
            print(f"Ground Truth (xywh): [{gx}, {gy}, {gw}, {gh}]")
        else:
            print("Ground Truth: None")

        # --- PREPARE INPUT ---
        # We try a very specific prompt to force JSON output
        prompt_text = (
            "Detect the signature in this image. "
            "Return the bounding box as a JSON list [ymin, xmin, ymax, xmax] "
            "using a 0-1000 coordinate scale."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Process Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # --- GENERATE ---
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode Output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\n--- RAW MODEL OUTPUT ---")
        print(f"'{output_text}'")
        print("------------------------")
        
        # --- TEST PARSING ---
        # Try to find the JSON list
        matches = re.findall(r"\[([\d,\s\.]+)\]", output_text)
        
        if matches:
            print(f"Regex Matched: {matches[-1]}")
            coords = [float(x.strip()) for x in matches[-1].split(',')]
            print(f"Parsed Coords: {coords}")
            
            # Check Scale
            if all(c <= 1.0 for c in coords):
                print("WARNING: Coordinates look like 0-1 scale (not 0-1000).")
            elif any(c > 1000 for c in coords):
                print("WARNING: Coordinates look like absolute pixels (not 0-1000).")
            else:
                print("Scale looks correct (0-1000).")
        else:
            print("FAIL: Regex could not find a JSON list in the output.")

if __name__ == "__main__":
    main()
