import os
import re
from typing import Dict
import json
import cv2
import torch
import matplotlib.pyplot as plt
# Updated import for Qwen2_5_VL and Qwen3 compatibility
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm

class Qwen_llm:
    def __init__(self, base_model_path="Qwen/Qwen3-VL-2B-Instruct", adapter_path=None, cache_dir="/tmp/huggingface_cache"):
        """
        Initializes the configuration for Qwen3-VL (or Qwen2.5-VL).
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            
        self.model = None
        self.processor = None

    def load(self, is_trainable=False):
        """
        Loads the model with support for Qwen3's specific architecture and BFloat16.
        """
        print(f"⏳ Loading Model: {self.base_model_path}...")
        
        # Qwen3-VL works best with bfloat16 and sdpa/flash_attention_2
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="sdpa", 
            trust_remote_code=True
        )
        
        processor_path = self.base_model_path

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"⏳ Applying LoRA Adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.adapter_path, 
                is_trainable=is_trainable 
            )
            processor_path = self.adapter_path 
            
        if is_trainable:
            self.model.train()
        else:
            self.model.eval()

        print("⏳ Loading Processor...")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        
        # 🚀 FIX: Qwen3-VL Tokenizer Alignment
        # Set pad_token if not defined to avoid issues during batched inference
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            
        print("✅ Qwen3-VL and Processor loaded.")

    def predict(self, image_path, prompt="Detect all signatures and stamps. Output a JSON list with 'bbox' [ymin, xmin, ymax, xmax] and 'label'."):
        """
        Inference optimized for Qwen3-VL's spatial reasoning.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded.")

        # Get original dims for mapping
        orig_img = cv2.imread(image_path)
        if orig_img is None: raise ValueError(f"Invalid image: {image_path}")
        original_height, original_width = orig_img.shape[:2]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
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
                max_new_tokens=512, # Qwen3 can handle more complex reasoning
                do_sample=False,
                use_cache=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode output
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        pixel_boxes = self.postprocess(
            raw_output=output_text, 
            original_width=original_width, 
            original_height=original_height
        )

        return {"boxes": pixel_boxes, "raw_text": output_text}

    def postprocess(self, raw_output, original_width, original_height, is_ground_truth=False):
        """
        Robust JSON extraction for V3 output.
        """
        clean_output = raw_output.strip()
        
        # Remove common chat tokens and markdown
        clean_output = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", clean_output)
        if "```json" in clean_output:
            clean_output = re.search(r"```json(.*?)```", clean_output, re.DOTALL).group(1)
        elif "```" in clean_output:
            clean_output = re.search(r"```(.*?)```", clean_output, re.DOTALL).group(1)
        
        clean_output = clean_output.strip()
        
        try:
            detections = json.loads(clean_output)
        except json.JSONDecodeError:
            # Fallback: Try finding anything that looks like a list
            match = re.search(r"\[\s*\{.*\}\s*\]", clean_output, re.DOTALL)
            if match:
                try: detections = json.loads(match.group(0))
                except: return []
            else: return []
            
        pixel_results = []
        if not isinstance(detections, list): return pixel_results
            
        for det in detections:
            bbox = det.get("bbox", [])
            label = det.get("label", "unknown")
            
            if len(bbox) != 4: continue
                
            # Un-normalize from 0-1000 scale to pixel space
            ymin, xmin, ymax, xmax = bbox
            
            x1 = int((xmin / 1000.0) * original_width)
            y1 = int((ymin / 1000.0) * original_height)
            x2 = int((xmax / 1000.0) * original_width)
            y2 = int((ymax / 1000.0) * original_height)
            
            # Clip and format
            x1, x2 = sorted([max(0, min(original_width - 1, x1)), max(0, min(original_width - 1, x2))])
            y1, y2 = sorted([max(0, min(original_height - 1, y1)), max(0, min(original_height - 1, y2))])
            
            pixel_results.append({"bbox": [x1, y1, x2, y2], "label": label})
                
        return pixel_results

    # ... [plot_bounding_boxes and evaluate functions remain largely the same] ...
