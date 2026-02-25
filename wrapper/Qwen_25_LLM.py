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


class Qwen_llm:
    def __init__(self, base_model_path, adapter_path=None, cache_dir="/tmp/huggingface_cache"):
        """
        Initializes the configuration for the Qwen2.5-VL model.
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
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
            attn_implementation="flash_attention_2"
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

        # --- 2. FORWARD PASS ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, # Feed the RAW image path!
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
        pixel_boxes = self.postprocess(
            raw_output=output_text, 
            original_width=original_width, 
            original_height=original_height,
            is_ground_truth=False
        )

        return {
            "boxes": pixel_boxes, 
            "raw_text": output_text
        }


    def postprocess(self, raw_output, original_width, original_height, is_ground_truth=False):
        """
        Extracts JSON structured bounding boxes and converts them to absolute pixel coordinates.
        
        Args:
            raw_output (str): The raw text output (e.g., '[{"bbox":...}]<|im_end|>') or clean GT JSON.
            original_width (int): The width of the original, un-resized document.
            original_height (int): The height of the original, un-resized document.
            is_ground_truth (bool): If True, bypasses LLM-specific text cleaning.
            
        Returns:
            list: A list of dictionaries containing 'bbox' [x1, y1, x2, y2] and 'label'.
        """
        if not is_ground_truth:
            # 1. Clean the raw output (Predictions ONLY)
            clean_output = raw_output.strip()
            
            # Strip Qwen's specific stop tokens
            clean_output = clean_output.replace("<|im_end|>", "").strip()
            clean_output = clean_output.replace("<|endoftext|>", "").strip()
            
            # Handle the case where the model hallucinates markdown JSON wrappers
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
                
            clean_output = clean_output.strip()
        else:
            # Ground truth is already clean
            clean_output = raw_output.strip()
        
        # 2. Parse the JSON string
        try:
            detections = json.loads(clean_output)
        except json.JSONDecodeError as e:
            prefix = "Ground Truth" if is_ground_truth else "Prediction"
            print(f"Warning: Failed to parse {prefix} JSON. Error: {e}")
            print(f"Raw output was: {raw_output}")
            return []
            
        pixel_results = []
        
        if not isinstance(detections, list) or len(detections) == 0:
            return pixel_results
            
        # 3. Process each detection
        for det in detections:
            bbox = det.get("bbox", [])
            label = det.get("label", "unknown")
            
            if len(bbox) != 4:
                continue
                
            # Qwen JSON format ordering: [ymin, xmin, ymax, xmax] on a 0-1000 scale
            y1_n, x1_n, y2_n, x2_n = bbox
            
            # 4. Un-normalize back to original dimensions
            x1 = int((x1_n / 1000.0) * original_width)
            y1 = int((y1_n / 1000.0) * original_height)
            x2 = int((x2_n / 1000.0) * original_width)
            y2 = int((y2_n / 1000.0) * original_height)
            
            # 5. Safety Check: Clip to image boundaries
            x1 = max(0, min(original_width - 1, x1))
            y1 = max(0, min(original_height - 1, y1))
            x2 = max(0, min(original_width - 1, x2))
            y2 = max(0, min(original_height - 1, y2))
            
            # 6. Ensure correct ordering (prevents inverted boxes)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Only append valid, non-zero area boxes
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                pixel_results.append({
                    "bbox": [x1, y1, x2, y2], 
                    "label": label
                })
                
        return pixel_results
 
    def plot_bounding_boxes(self,image_path, predictions=None, ground_truths=None, save_path=None):
        """
        Visualizes the image, drawing predictions in Green and ground truths in Red.
        Includes robust error handling for missing or malformed bounding boxes.
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

        # Plot Ground Truths (RED)
        for gt in ground_truths:
            if isinstance(gt, dict):
                bbox = gt.get("bbox", [])
                label_text = f"GT: {gt.get('label', 'unknown')}"
            else:
                bbox = gt  # Fallback for old list format
                label_text = "GT"
                
            # 🚀 THE FIX: Ensure we actually have 4 coordinates before unpacking
            if len(bbox) != 4:
                print(f"⚠️ Skipping malformed GT bbox: {bbox}")
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
            cv2.putText(image, label_text, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Plot Predictions (GREEN)
        for pred in predictions:
            if isinstance(pred, dict):
                bbox = pred.get("bbox", [])
                label_text = f"Pred: {pred.get('label', 'unknown')}"
            else:
                bbox = pred  # Fallback for old list format
                label_text = "Pred"
                
            # 🚀 THE FIX: Ensure we actually have 4 coordinates before unpacking
            if len(bbox) != 4:
                print(f"⚠️ Skipping malformed Pred bbox: {bbox}")
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box
            cv2.putText(image, label_text, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save or Display
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

    
    def evaluate(self,dataset) -> Dict:
        """
        Computes IoU scores between predicted and ground truth boxes.
        Returns a list of IoU scores for each prediction.
        """
            
        def calculate_iou(boxA, boxB):
            # Standard IoU calculation
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
             
        
        def evaluate_detection(pred_box, gt_box, img_width=1, img_height=1):
            """
            Evaluates prediction against ground truth with distance metrics.
            
            Args:
                pred_box (list): [xmin, ymin, xmax, ymax]
                gt_box (list):   [xmin, ymin, xmax, ymax]
                img_width (int): Width of the image (for normalization)
                img_height (int): Height of the image (for normalization)
                
            Returns:
                dict: IoU, IoP, Center Distance (Pixels), Normalized Center Distance (0-1)
            """
            # --- 1. IoU Calculation (Standard) ---
            xA = max(pred_box[0], gt_box[0])
            yA = max(pred_box[1], gt_box[1])
            xB = min(pred_box[2], gt_box[2])
            yB = min(pred_box[3], gt_box[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            predArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gtArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            
            iou = interArea / float(predArea + gtArea - interArea + 1e-6)
            iop = interArea / float(predArea + 1e-6) # Intersection over Prediction

            # --- 2. Center Point Calculation ---
            pred_cx = (pred_box[0] + pred_box[2]) / 2.0
            pred_cy = (pred_box[1] + pred_box[3]) / 2.0
            
            gt_cx = (gt_box[0] + gt_box[2]) / 2.0
            gt_cy = (gt_box[1] + gt_box[3]) / 2.0
            
            # --- 3. Euclidean Distance (Pixels) ---
            # Pythagorean theorem: a^2 + b^2 = c^2
            dist_pixels = math.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
            
            # --- 4. Normalized Distance (0.0 to 1.0) ---
            # Distance relative to the image diagonal. 
            # 0.05 means the center is off by 5% of the image size.
            # This helps compare errors across images of different resolutions.
            img_diagonal = math.sqrt(img_width**2 + img_height**2) + 1e-6
            norm_dist = dist_pixels / img_diagonal

            return {
                "iou": round(iou, 4),
                "iop": round(iop, 4),
                "center_dist_px": round(dist_pixels, 1),
                "norm_center_dist": round(norm_dist, 4)
            }
   
        
        def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5):
            """
            Matches predictions to ground truths using greedy IoU strategy.
            
            Args:
                pred_boxes (list): List of [xmin, ymin, xmax, ymax]
                gt_boxes (list):   List of [xmin, ymin, xmax, ymax]
                iou_threshold (float): Minimum IoU to consider a match valid
                
            Returns:
                matches (list): List of dicts {'pred': box, 'gt': box, 'iou': float}
                unmatched_preds (list): List of pred_boxes that matched nothing
                unmatched_gts (list): List of gt_boxes that were missed
            """
            matches = []
            # pred_boxes = [box['bbox_2d'] for box in pred_boxes]
            
            # Keep track of which indices have been matched
            matched_pred_indices = set()
            matched_gt_indices = set()
            
            # 1. Calculate IoU for ALL pairs
            # Format: (iou, pred_index, gt_index)
            all_pairs = []
            for i, p_box in enumerate(pred_boxes):
                for j, g_box in enumerate(gt_boxes):
                    iou = calculate_iou(p_box, g_box)
                    if iou > 0.0: # Only consider pairs that overlap at least a little
                        all_pairs.append((iou, i, j))
            
            # 2. Sort pairs by IoU (Highest first)
            all_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # 3. Greedy Matching
            for iou, p_idx, g_idx in all_pairs:
                if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
                    # Found the best remaining match!
                    if iou >= iou_threshold:
                        matches.append({
                            'pred': pred_boxes[p_idx],
                            'gt': gt_boxes[g_idx],
                            'iou': iou
                        })
                        matched_pred_indices.add(p_idx)
                        matched_gt_indices.add(g_idx)
            
            # 4. Gather leftovers
            unmatched_preds = [p for i, p in enumerate(pred_boxes) if i not in matched_pred_indices]
            unmatched_gts = [g for i, g in enumerate(gt_boxes) if i not in matched_gt_indices]
            
            return matches, unmatched_preds, unmatched_gts

        
        metrics_summary = {
            "iou": [],
            "iop": [],
            "norm_dist": [],
            "false_positives": 0,
            "missed_signatures": 0,
            "total_images": 0
        }
        conficting_predictions = []

        for item in tqdm(dataset):
            image_path = item['image_path']
            ground_truth = item['groundTruth']
            
            metrics_summary["total_images"] += 1

            
            output_text = self.predict(image_path, "Locate the signature")
            output_text = self.postprocess(output_text, image_width=640, image_height=640)
            ground_truth = self.postprocess(ground_truth, image_width=640, image_height=640)
            
            matches, false_positives, misses = match_predictions_to_ground_truth(output_text, ground_truth, iou_threshold=0.1)
            metrics_summary["false_positives"] += len(false_positives)
            metrics_summary["missed_signatures"] += len(misses)
            if len(false_positives) > 0 or len(misses) > 0: #append only if there is a conflict to analyze
                conficting_predictions.append({
                    "image_path": image_path,
                    "predictions": output_text,
                    "ground_truths": ground_truth,
                    "false_positives": false_positives,
                    "misses": misses
                })

            
            # Collect Metrics for Matches
            for match in matches:
                # Evaluate using the normalized distance metric
                m = evaluate_detection(match['pred'], match['gt'], img_width=640, img_height=640)
                
                metrics_summary["iou"].append(m['iou'])
                metrics_summary["iop"].append(m['iop'])
                metrics_summary["norm_dist"].append(m['norm_center_dist'])
            
        # --- 3. Calculate Averages ---
        total_matches = len(metrics_summary["iou"])

        if total_matches > 0:
            avg_iou = sum(metrics_summary["iou"]) / total_matches
            avg_iop = sum(metrics_summary["iop"]) / total_matches
            avg_norm_dist = sum(metrics_summary["norm_dist"]) / total_matches
        else:
            avg_iou = avg_iop = avg_norm_dist = 0.0

        # --- 4. Final Report ---
        print("\n" + "="*50)
        print(f" FINAL EVALUATION REPORT ({metrics_summary['total_images']} Images)")
        print("="*50)
        print(f"Total Matches Found:      {total_matches}")
        print(f"Total Missed Signatures:  {metrics_summary['missed_signatures']}")
        print(f"Total False Positives:    {metrics_summary['false_positives']}")
        print("-" * 50)
        print(f"Mean IoU (Overlap):             {avg_iou:.4f}")
        print(f"Mean IoP (Tightness/Precision): {avg_iop:.4f}")
        print(f"Mean Normalized Center Error:   {avg_norm_dist:.4f} ({(avg_norm_dist*100):.2f}% of image diagonal)")
        print("="*50)
        
        #save the conflicting predictions for error analysis
        with open("conflicting_predictions.json", "w") as f:
            json.dump(conficting_predictions, f, indent=4)
            print(f"📁 Saved conflicting predictions to conflicting_predictions.json for error analysis.")
        
        return metrics_summary
        