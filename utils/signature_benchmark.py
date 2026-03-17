import math
import json
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import cv2
import matplotlib.pyplot as plt


class SignatureBenchmark:
    def __init__(self, model_wrapper, img_size: int = 640, iou_threshold: float = 0.70):
        self.model = model_wrapper
        self.img_size = img_size
        self.iou_threshold = iou_threshold 

    @staticmethod
    def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    @staticmethod
    def evaluate_detection(pred_box, gt_box, img_width, img_height):
        xA, yA = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
        xB, yB = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        predArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gtArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        
        iou = interArea / float(predArea + gtArea - interArea + 1e-6)
        iop = interArea / float(predArea + 1e-6)

        pred_cx, pred_cy = (pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0
        gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
        
        dist_pixels = math.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
        img_diagonal = math.sqrt(img_width**2 + img_height**2) + 1e-6
        norm_dist = dist_pixels / img_diagonal

        return {
            "iou": round(iou, 4), "iop": round(iop, 4),
            "center_dist_px": round(dist_pixels, 1), "norm_center_dist": round(norm_dist, 4)
        }

    def match_predictions_to_ground_truth(self, pred_boxes, gt_boxes, iou_threshold):
        matches, matched_p, matched_g = [], set(), set()
        all_pairs = []

        for i, p_item in enumerate(pred_boxes):
            for j, g_item in enumerate(gt_boxes):
                iou = self.calculate_iou(p_item['bbox'], g_item['bbox'])
                if iou > 0.0:
                    all_pairs.append((iou, i, j))

        all_pairs.sort(key=lambda x: x[0], reverse=True)

        for iou, p_idx, g_idx in all_pairs:
            if p_idx not in matched_p and g_idx not in matched_g:
                if iou >= iou_threshold:
                    matches.append({'pred': pred_boxes[p_idx], 'gt': gt_boxes[g_idx], 'iou': iou})
                    matched_p.add(p_idx)
                    matched_g.add(g_idx)

        unmatched_preds = [p for i, p in enumerate(pred_boxes) if i not in matched_p]
        unmatched_gts = [g for i, g in enumerate(gt_boxes) if i not in matched_g]
        
        return matches, unmatched_preds, unmatched_gts

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], save_path: str = "confusion_matrix.png"):
        """
        Generates a visual confusion matrix ONLY for correctly localized boxes.
        """
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        
        plt.title(f'Classification Accuracy (On Successfully Localized Boxes)', pad=20, fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"📸 Saved Confusion Matrix plot to {save_path}")
            
        plt.show()

    def evaluate_single_label(self, dataset: List[Dict], target_label: str, output_json: str = None, cm_save_path: str = None) -> Dict:
        """
        Evaluates the model strictly on a single label. 
        Tracks True Negatives (images correctly identified as NOT containing the label).
        """
        output_json = output_json or f"conflicts_only_{target_label}.json"
        cm_save_path = cm_save_path or f"confusion_matrix_only_{target_label}.png"
        
        metrics = {
            "iou": [], "iop": [], "norm_dist": [], 
            "tp": 0, "fp": 0, "fn": 0, "total_imgs": 0,
            "true_negatives": 0, "total_negatives": 0,  # <-- NEW TRACKERS
            "class_correct": 0, "class_incorrect": 0,
            "class_stats": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        }
        conflicts = []
        y_true_matches = []
        y_pred_matches = []

        prompt = f"Detect all {target_label}s. If none are found, respond with: None of the requested objects were found."

        for item in tqdm(dataset, desc=f"Evaluating strictly on: '{target_label}'"):
            metrics["total_imgs"] += 1
            
            try:
                raw_gts = self.model.parse_qwen_ground_truth(item['groundTruth'], self.img_size, self.img_size)
                # Keep ONLY the target label
                gts = [gt for gt in raw_gts if gt.get('label') == target_label]
                
                raw_output, _, _ = self.model.predict(item['image_path'], prompt)
                preds = self.model.parse_qwen_detections(raw_output, self.img_size, self.img_size)
                
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                preds, gts = [], []

            # -------------------------------------------------------------
            # NEW LOGIC: Track Image-Level True Negatives
            # If the GT is empty for this label, it's a "None Example"
            # -------------------------------------------------------------
            if len(gts) == 0:
                metrics["total_negatives"] += 1
                if len(preds) == 0:
                    metrics["true_negatives"] += 1 # The model correctly stayed quiet

            matches, fps, misses = self.match_predictions_to_ground_truth(
                preds, gts, iou_threshold=self.iou_threshold 
            )
            
            metrics["tp"] += len(matches)
            metrics["fp"] += len(fps)
            metrics["fn"] += len(misses)

            for match in matches:
                pred_label = match['pred'].get('label', 'unknown')
                gt_label = match['gt'].get('label', 'unknown')
                
                y_true_matches.append(gt_label)
                y_pred_matches.append(pred_label)

                if pred_label == gt_label:
                    metrics["class_correct"] += 1
                    metrics["class_stats"][gt_label]["tp"] += 1
                else:
                    metrics["class_incorrect"] += 1
                    metrics["class_stats"][gt_label]["fn"] += 1
                    metrics["class_stats"][pred_label]["fp"] += 1

                m = self.evaluate_detection(match['pred']['bbox'], match['gt']['bbox'], self.img_size, self.img_size)
                metrics["iou"].append(m['iou'])
                metrics["iop"].append(m['iop'])
                metrics["norm_dist"].append(m['norm_center_dist'])

            for miss in misses:
                metrics["class_stats"][miss.get('label', 'unknown')]["fn"] += 1
                
            for fp_box in fps:
                metrics["class_stats"][fp_box.get('label', 'unknown')]["fp"] += 1

            if fps or misses or any(m['pred'].get('label') != m['gt'].get('label') for m in matches):
                conflicts.append({
                    "image": item['image_path'],
                    "preds": preds, "gts": gts,
                    "fp_count": len(fps), "miss_count": len(misses)
                })

        if y_true_matches and y_pred_matches:
            self.plot_confusion_matrix(y_true_matches, y_pred_matches, save_path=cm_save_path)

        return self._summarize(metrics, conflicts, output_json)
    
    def _summarize(self, metrics, conflicts, output_json):
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_classified = metrics["class_correct"] + metrics["class_incorrect"]
        class_accuracy = metrics["class_correct"] / total_classified if total_classified > 0 else 0.0

        total_m = len(metrics["iou"])
        avg = lambda x: sum(x) / total_m if total_m > 0 else 0.0
        
        per_class_metrics = {}
        if "class_stats" in metrics:
            for label, stats in metrics["class_stats"].items():
                c_tp = stats["tp"]
                c_fp = stats["fp"]
                c_fn = stats["fn"]
                
                c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
                
                per_class_metrics[label] = {
                    "precision": round(c_prec, 4),
                    "recall": round(c_rec, 4),
                    "f1": round(c_f1, 4),
                    "tp": c_tp,
                    "fp": c_fp,
                    "fn": c_fn
                }

        iou_key = f"localization_f1_at_{int(self.iou_threshold * 100)}"
        
        summary = {
            iou_key: round(f1_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou": avg(metrics["iou"]),
            "total_tp": tp, 
            "total_fp": fp, 
            "total_fn": fn,
            "true_negatives": metrics.get("true_negatives", 0), # NEW
            "total_negatives": metrics.get("total_negatives", 0), # NEW
            "classification_accuracy": round(class_accuracy, 4),
            "per_class_metrics": per_class_metrics
        }

        with open(output_json, "w") as f:
            json.dump(conflicts, f, indent=4)
            
        self._print_report(summary, metrics["total_imgs"])
        return summary
    
    def evaluate_multi_label(self, dataset: List[Dict], output_json: str = "conflicts_multi_label.json", cm_save_path: str = "confusion_matrix_multi.png") -> Dict:
        """
        Evaluates the model on ALL labels simultaneously using a single full prompt.
        Tracks global localization, per-class classification, and Image-Level True Negatives 
        (documents correctly identified as containing none of the target objects).
        """
        metrics = {
            "iou": [], "iop": [], "norm_dist": [], 
            "tp": 0, "fp": 0, "fn": 0, "total_imgs": 0,
            "true_negatives": 0, "total_negatives": 0,  # Image-level empty document trackers
            "class_correct": 0, "class_incorrect": 0,
            "class_stats": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        }
        conflicts = []
        y_true_matches = []
        y_pred_matches = []

        # 🚀 The Full Multi-Label Prompt
        prompt ="Detect all stamps, signatures, and logos. If none are found, respond with: None of the requested objects were found."

        for item in tqdm(dataset, desc="Evaluating ALL Labels (Multi-Class)"):
            metrics["total_imgs"] += 1
            
            try:
                # 1. Parse ALL Ground Truths (No filtering)
                gts = self.model.parse_qwen_ground_truth(item['groundTruth'], self.img_size, self.img_size)
                
                # 2. Predict using the full multi-label prompt
                raw_output, _, _ = self.model.predict(item['image_path'], prompt)
                preds = self.model.parse_qwen_detections(raw_output, self.img_size, self.img_size)
                
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                preds, gts = [], []

            # -------------------------------------------------------------
            # IMAGE-LEVEL TRUE NEGATIVES
            # If the document has NO stamps, NO signatures, and NO logos
            # -------------------------------------------------------------
            if len(gts) == 0:
                metrics["total_negatives"] += 1
                if len(preds) == 0:
                    metrics["true_negatives"] += 1 # The model correctly identified a completely blank doc

            # --- Standard Matching and Scoring Logic ---
            matches, fps, misses = self.match_predictions_to_ground_truth(
                preds, gts, iou_threshold=self.iou_threshold 
            )
            
            metrics["tp"] += len(matches)
            metrics["fp"] += len(fps)
            metrics["fn"] += len(misses)

            for match in matches:
                pred_label = match['pred'].get('label', 'unknown')
                gt_label = match['gt'].get('label', 'unknown')
                
                y_true_matches.append(gt_label)
                y_pred_matches.append(pred_label)

                if pred_label == gt_label:
                    metrics["class_correct"] += 1
                    metrics["class_stats"][gt_label]["tp"] += 1
                else:
                    metrics["class_incorrect"] += 1
                    metrics["class_stats"][gt_label]["fn"] += 1
                    metrics["class_stats"][pred_label]["fp"] += 1

                m = self.evaluate_detection(match['pred']['bbox'], match['gt']['bbox'], self.img_size, self.img_size)
                metrics["iou"].append(m['iou'])
                metrics["iop"].append(m['iop'])
                metrics["norm_dist"].append(m['norm_center_dist'])

            for miss in misses:
                metrics["class_stats"][miss.get('label', 'unknown')]["fn"] += 1
                
            for fp_box in fps:
                metrics["class_stats"][fp_box.get('label', 'unknown')]["fp"] += 1

            if fps or misses or any(m['pred'].get('label') != m['gt'].get('label') for m in matches):
                conflicts.append({
                    "image": item['image_path'],
                    "preds": preds, "gts": gts,
                    "fp_count": len(fps), "miss_count": len(misses)
                })

        # Generate the visual confusion matrix purely for matches
        if y_true_matches and y_pred_matches:
            self.plot_confusion_matrix(y_true_matches, y_pred_matches, save_path=cm_save_path)

        # Re-use the _summarize logic which now handles 'true_negatives' perfectly
        return self._summarize(metrics, conflicts, output_json)

    def _print_report(self, s, total_imgs):
        iou_key = f"localization_f1_at_{int(self.iou_threshold * 100)}"
        
        print("\n" + "="*75)
        print(f" 📊 QWEN EVALUATION REPORT ({total_imgs} Images)")
        print(f" Threshold IoU: {self.iou_threshold}")
        print("="*75)
        print(" 1. LOCALIZATION METRICS (Finding the box)")
        print(f" Precision: {s['precision']:.4f} | Recall: {s['recall']:.4f}")
        print(f" F1-Score:  {s[iou_key]:.4f}")
        print("-" * 75)
        print(f" Successfully Localized (TP):         {s['total_tp']}")
        print(f" Wrongly Predicted Boxes (FP):        {s['total_fp']}  <-- Failed to hit {self.iou_threshold} IoU")
        print(f" Missed Ground Truth Boxes (FN):      {s['total_fn']}")
        
        # --- Print True Negatives if there were any 'None Examples' ---
        if s.get("total_negatives", 0) > 0:
            tn_rate = s['true_negatives'] / s['total_negatives']
            print(f" Correctly Empty Images (TN):         {s['true_negatives']} / {s['total_negatives']} ({tn_rate:.2%})")
            
        print(f" Mean IoU of Successes:               {s['mean_iou']:.4f}")
        print("-" * 75)
        print(" 2. CLASSIFICATION METRICS (Labeling the correctly found boxes)")
        print(f" Accuracy:  {s['classification_accuracy']:.2%}")
        
        if s.get("per_class_metrics"):
            print("-" * 75)
            print(" 3. DETAILED PER-CLASS METRICS")
            print("-" * 75)
            print(f" {'Label':<15} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
            print("-" * 75)
            for label, cm in sorted(s["per_class_metrics"].items()):
                print(f" {label:<15} | {cm['precision']:<9.4f} | {cm['recall']:<9.4f} | {cm['f1']:<9.4f} | {cm['tp']:<4} | {cm['fp']:<4} | {cm['fn']:<4}")
        print("="*75)
