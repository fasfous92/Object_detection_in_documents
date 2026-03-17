import os
import re
from typing import Dict

import json
import cv2
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm
SYSTEM_PROMPT='''
You are an expert document analysis AI. 
Classify visual elements as strictly ONE of the following:
- "signature": Handwritten names, initials, or cursive ink strokes.
- "logo": Printed graphic symbols, company emblems, or brand marks.
- "stamp": Inked rubber stamps, official seals, or stamped dates/text.
Do not include markdown formatting, conversational text, or explanations.
'''


class Qwen_llm:
    def __init__(self, base_model_path, adapter_path=None, cache_dir="/tmp/huggingface_cache", system_prompt=SYSTEM_PROMPT):
        """
        Initializes the configuration for the Qwen2.5-VL model.
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
        # Add the system prompt to the instance
        # If None is provided, we default to a standard helpful assistant prompt, 
        # or you can leave it entirely blank depending on your needs.
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        # Set the Hugging Face cache directory
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            
        self.model = None
        self.processor = None

    def load(self, is_trainable=False):
        """
        Loads the base model, applies the LoRA adapter (if provided), and loads the processor.
        If is_trainable=True, prepares the adapter weights for continued fine-tuning.
        """
        print(f"⏳ Loading Base Model from {self.base_model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa" #"flash_attention_2"
        )
        
        processor_path = self.base_model_path

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"⏳ Merging LoRA Adapter from {self.adapter_path}...")
            # CRITICAL: Pass is_trainable to PeftModel so the weights aren't frozen
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.adapter_path, 
                is_trainable=is_trainable 
            )
            processor_path = self.adapter_path 
            
        # Toggle train/eval mode based on the boolean
        if is_trainable:
            self.model.train()
            print("⚙️ Model loaded in TRAINING mode.")
        else:
            self.model.eval()
            print("🛡️ Model loaded in EVALUATION (Inference) mode.")

        print("⏳ Loading Processor...")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        print("✅ Model and Processor loaded successfully.")
        
    def predict(self, image_path, prompt="Detect all signatures and stamps. Output a JSON list with 'bbox' [ymin, xmin, ymax, xmax] and 'label'."):
        """
        End-to-end inference:
        1. Reads original dimensions for post-processing.
        2. Passes the raw image path directly to Qwen (letting max_pixels handle safe scaling).
        3. Delegates the coordinate mapping to self.postprocess().
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # --- 1. GET ORIGINAL DIMENSIONS ---
        # We only use cv2 here to get the true W and H for the postprocess math.
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Could not read image at: {image_path}")
        
        original_height, original_width = orig_img.shape[:2]
        
        img_resized = cv2.resize(orig_img, (640, 640))
        resized_img_path = "resized_test_image.jpg"
        cv2.imwrite(resized_img_path, img_resized)


        # --- 2. FORWARD PASS ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_img_path}, # Feed the RAW image path!
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256, # Increased to prevent cutting off long JSON lists
                do_sample=False,
                temperature=0.0     # Forces strict, uncreative JSON output
            )

        # Isolate newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the raw text output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # --- 3. POST-PROCESSING (Mapping back to HD) ---
        # Ensure is_ground_truth is False so it strips markdown/stop tokens
        # pixel_boxes = self.postprocess(
        #     raw_output=output_text, 
        #     original_width=original_width, 
        #     original_height=original_height,
        #     is_ground_truth=False
        # )

        return  output_text,original_width, original_height
    

    def postprocess_prediction(self, raw_output, original_width, original_height, normalize=True):
        """
        Cleans LLM artifacts from the prediction output, extracts JSON bounding boxes, 
        conditionally un-normalizes coordinates, and formats them.
        """
        clean_output = raw_output.strip()
        
        # Strip specific Qwen stop tokens
        clean_output = clean_output.replace("<|im_end|>", "").strip()
        clean_output = clean_output.replace("<|endoftext|>", "").strip()
        
        # Handle the case where the model hallucinates markdown JSON wrappers
        if clean_output.startswith("```json"):
            clean_output = clean_output[7:]
        if clean_output.endswith("```"):
            clean_output = clean_output[:-3]
            
        clean_output = clean_output.strip()
        
        try:
            detections = json.loads(clean_output)
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Failed to parse Prediction JSON. Error: {e}")
            print(f"Raw output was: {raw_output}")
            return []
            
        pixel_results = []
        if not isinstance(detections, list) or len(detections) == 0:
            return pixel_results
            
        for det in detections:
            bbox = det.get("bbox_2d", det.get("bbox", []))
            label = det.get("label", "unknown")
            
            if len(bbox) != 4:
                continue
                
            x1_n, y1_n, x2_n, y2_n = bbox
            
            if normalize:
                x1 = int((x1_n / 1000.0) * original_width)
                y1 = int((y1_n / 1000.0) * original_height)
                x2 = int((x2_n / 1000.0) * original_width)
                y2 = int((y2_n / 1000.0) * original_height)
            else:
                x1, y1, x2, y2 = int(x1_n), int(y1_n), int(x2_n), int(y2_n)
            
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(0, min(original_width - 1, x2))
            y2 = max(0, min(original_height - 1, y2))
            
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                pixel_results.append({"bbox": [x1, y1, x2, y2], "label": label})
                
        return pixel_results
    def postprocess_ground_truth(self, raw_output, original_width, original_height, normalize=True):
        """
        Parses clean ground truth JSON, conditionally un-normalizes coordinates, 
        and formats them. Bypasses all LLM-specific text cleaning.
        """
        # Robustness check: If the dataset already parsed the JSON into a Python list, use it directly.
        if isinstance(raw_output, list):
            detections = raw_output
        else:
            # Otherwise, assume it's a JSON string and parse it.
            clean_output = raw_output.strip()
            try:
                detections = json.loads(clean_output)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Failed to parse Ground Truth JSON. Error: {e}")
                print(f"Raw GT was: {raw_output}")
                return []
                
        pixel_results = []
        if not isinstance(detections, list) or len(detections) == 0:
            return pixel_results
            
        for det in detections:
            bbox = det.get("bbox_2d", det.get("bbox", []))
            label = det.get("label", "unknown")
            
            if len(bbox) != 4:
                continue
                
            x1_n, y1_n, x2_n, y2_n = bbox
            
            if normalize:
                x1 = int((x1_n / 1000.0) * original_width)
                y1 = int((y1_n / 1000.0) * original_height)
                x2 = int((x2_n / 1000.0) * original_width)
                y2 = int((y2_n / 1000.0) * original_height)
            else:
                x1, y1, x2, y2 = int(x1_n), int(y1_n), int(x2_n), int(y2_n)
            
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(0, min(original_width - 1, x2))
            y2 = max(0, min(original_height - 1, y2))
            
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                pixel_results.append({"bbox": [x1, y1, x2, y2], "label": label})
                
        return pixel_results

    def plot_bounding_boxes(self, image_path, predictions=None, ground_truths=None, save_path=None):
        """
        Visualizes the image, drawing predictions in Green and ground truths in Red.
        Assumes bounding boxes are already in absolute pixel coordinates [xmin, ymin, xmax, ymax].
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image at {image_path}")
            return
            
        # Convert to RGB for accurate color plotting in Matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Safely handle None inputs
        ground_truths = ground_truths or []
        predictions = predictions or []

        # Helper to safely extract coordinates and labels from various dictionary formats
        def extract_bbox_and_label(item, default_prefix):
            if isinstance(item, dict):
                # Check for 'bbox' first, fallback to 'bbox_2d'
                bbox = item.get("bbox", item.get("bbox_2d", []))
                label = f"{default_prefix}: {item.get('label', 'unknown')}"
                return bbox, label
            return item, default_prefix # Fallback for flat lists

        # --- 1. Plot Ground Truths (RED) ---
        for gt in ground_truths:
            bbox, label_text = extract_bbox_and_label(gt, "GT")
                
            if len(bbox) != 4:
                print(f"⚠️ Skipping malformed GT bbox: {bbox}")
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
            
            # Place GT text ABOVE the box
            cv2.putText(image, label_text, (x1, max(y1 - 8, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- 2. Plot Predictions (GREEN) ---
        for pred in predictions:
            bbox, label_text = extract_bbox_and_label(pred, "Pred")
                
            if len(bbox) != 4:
                print(f"⚠️ Skipping malformed Pred bbox: {bbox}")
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box
            
            # Place Prediction text BELOW the box to prevent overlap with GT text
            cv2.putText(image, label_text, (x1, min(y2 + 20, image.shape[0] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 3. Save or Display ---
        if save_path:
            # Convert back to BGR because OpenCV expects BGR for saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f"📸 Saved visualization to {save_path}")
        else:
            plt.figure(figsize=(12, 12))
            plt.imshow(image)
            plt.axis("off")    
            plt.show()
            
######New approach ##################

    def parse_qwen_ground_truth(self,gt_string: str, img_width: int = None, img_height: int = None):
        """
        Parses Qwen-VL's native bounding box format (with special tokens) 
        into a structured list of dictionaries.
        """
        # Regex breakdown:
        # <\|object_ref_start\|>(.*?)<\|object_ref_end\|> -> Captures the label text non-greedily
        # <\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|> -> Captures the (xmin, ymin), (xmax, ymax) numbers
        # Note: We use \| to escape the pipe character, as it is a special regex operator.
        pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
        
        matches = re.findall(pattern, gt_string)
        results = []
        
        for match in matches:
            label = match[0]
            
            # Extract the normalized [0-1000] coordinates
            norm_xmin = int(match[1])
            norm_ymin = int(match[2])
            norm_xmax = int(match[3])
            norm_ymax = int(match[4])
            
            # If image dimensions are provided, scale back to actual image pixels
            if img_width and img_height:
                xmin = int((norm_xmin / 1000.0) * img_width)
                ymin = int((norm_ymin / 1000.0) * img_height)
                xmax = int((norm_xmax / 1000.0) * img_width)
                ymax = int((norm_ymax / 1000.0) * img_height)
            else:
                # Otherwise, return the raw 0-1000 scale
                xmin, ymin, xmax, ymax = norm_xmin, norm_ymin, norm_xmax, norm_ymax
                
            results.append({
                "label": label,
                "bbox": [xmin, ymin, xmax, ymax]
            })
            
        return results

    
    def parse_qwen_detections(self, model_output: str, img_width: int = None, img_height: int = None):
        """
        Parses the continuous string output from Qwen-VL into a structured list of dictionaries.
        Optionally converts normalized [0-1000] coordinates back to absolute image pixels.
        """
        # Regex breakdown: 
        # ([a-zA-Z]+)  -> Captures the label (e.g., 'stamp', 'signature')
        # \((\d+),(\d+)\),\((\d+),(\d+)\) -> Captures the (xmin, ymin), (xmax, ymax) numbers
        pattern = r"([a-zA-Z]+)\((\d+),(\d+)\),\((\d+),(\d+)\)"
        
        matches = re.findall(pattern, model_output)
        results = []
        
        for match in matches:
            label = match[0]
            
            # Extract the normalized [0-1000] coordinates
            norm_xmin = int(match[1])
            norm_ymin = int(match[2])
            norm_xmax = int(match[3])
            norm_ymax = int(match[4])
            
            # If image dimensions are provided, scale back to actual image pixels
            if img_width and img_height:
                xmin = int((norm_xmin / 1000.0) * img_width)
                ymin = int((norm_ymin / 1000.0) * img_height)
                xmax = int((norm_xmax / 1000.0) * img_width)
                ymax = int((norm_ymax / 1000.0) * img_height)
            else:
                # Otherwise, just return the raw 0-1000 scale
                xmin, ymin, xmax, ymax = norm_xmin, norm_ymin, norm_xmax, norm_ymax
                
            results.append({
                "label": label,
                "bbox": [xmin, ymin, xmax, ymax]
            })
            
        return results
