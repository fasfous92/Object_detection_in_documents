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

    def predict(self, image_path, prompt="Locate the signature."):
        """
        Takes an image and a text prompt, runs the forward pass, and returns the raw text output.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Call load() first.")

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
                max_new_tokens=100,
                do_sample=False
            )

        # Isolate the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def postprocess(self, raw_output, image_width, image_height):
        """
        Extracts Qwen's normalized coordinates and converts them to absolute pixel bounding boxes.
        Returns a list of [x1, y1, x2, y2] pixel coordinates.
        """
        pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
        matches = re.findall(pattern, raw_output)
        
        pixel_boxes = []
        for match in matches:
            y1_n, x1_n, y2_n, x2_n = map(int, match)
            
            # Convert from [0, 1000] scale to actual pixels
            x1 = int((x1_n / 1000) * image_width)
            y1 = int((y1_n / 1000) * image_height)
            x2 = int((x2_n / 1000) * image_width)
            y2 = int((y2_n / 1000) * image_height)
            
            pixel_boxes.append([x1, y1, x2, y2])
            
        return pixel_boxes

    def plot_bounding_boxes(self, image_path, predictions, ground_truths=None, save_path=None):
        """
        Visualizes the image, drawing predictions in Green and ground truths in Red.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image at {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot Ground Truths (RED)
        if ground_truths:
            for gt_box in ground_truths:
                x1, y1, x2, y2 = map(int, gt_box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
                cv2.putText(image, "GT", (x1, max(y1 - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Plot Predictions (GREEN)
        for pred_box in predictions:
            x1, y1, x2, y2 = map(int, pred_box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box
            cv2.putText(image, "Pred", (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"📸 Saved visualization to {save_path}")
        else:
            plt.show()
        return plt

    
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
        