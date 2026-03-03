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

    def evaluate(self, dataset: List[Dict], output_json: str = "conflicts.json", cm_save_path: str = "confusion_matrix.png") -> Dict:
        metrics = {
            "iou": [], "iop": [], "norm_dist": [], 
            "tp": 0, "fp": 0, "fn": 0, "total_imgs": 0,
            "class_correct": 0, "class_incorrect": 0
        }
        conflicts = []
        
        # Arrays for sklearn confusion matrix (STRICTLY for matched boxes)
        y_true_matches = []
        y_pred_matches = []

        for item in tqdm(dataset, desc="Evaluating Qwen"):
            metrics["total_imgs"] += 1
            
            try:
                raw_output, _, _ = self.model.predict(item['image_path'], "Detect all stamps, signatures, and logos.")
                preds = self.model.parse_qwen_detections(raw_output, self.img_size, self.img_size)
                gts = self.model.parse_qwen_ground_truth(item['groundTruth'], self.img_size, self.img_size)
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                preds, gts = [], [] 

            matches, fps, misses = self.match_predictions_to_ground_truth(
                preds, gts, iou_threshold=self.iou_threshold 
            )
            
            # 1. Localization Metrics (The "Did we find it?" test)
            metrics["tp"] += len(matches)
            metrics["fp"] += len(fps)
            metrics["fn"] += len(misses)

            # 2. Classification Metrics (The "Did we name it right?" test)
            for match in matches:
                pred_label = match['pred'].get('label', 'unknown')
                gt_label = match['gt'].get('label', 'unknown')
                
                # Append to our strictly-matched lists for the confusion matrix
                y_true_matches.append(gt_label)
                y_pred_matches.append(pred_label)

                if pred_label == gt_label:
                    metrics["class_correct"] += 1
                else:
                    metrics["class_incorrect"] += 1

                m = self.evaluate_detection(match['pred']['bbox'], match['gt']['bbox'], self.img_size, self.img_size)
                metrics["iou"].append(m['iou'])
                metrics["iop"].append(m['iop'])
                metrics["norm_dist"].append(m['norm_center_dist'])

            # Log conflicts (includes missed localizations AND bad classifications)
                if fps or misses or any(m['pred'].get('label') != m['gt'].get('label') for m in matches):
                                conflicts.append({
                                    "image": item['image_path'],
                                    "preds": preds, "gts": gts,
                                    "fp_count": len(fps), "miss_count": len(misses)
                                })
        # Generate the visual confusion matrix purely for matches
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
        
        summary = {
            f"localization_f1_at_{self.iou_threshold*100}": round(f1_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou": avg(metrics["iou"]),
            "total_tp": tp, "total_fp": fp, "total_fn": fn,
            "classification_accuracy": round(class_accuracy, 4)
        }

        with open(output_json, "w") as f:
            json.dump(conflicts, f, indent=4)
            
        self._print_report(summary, metrics["total_imgs"])
        return summary

    def _print_report(self, s, total_imgs):
        print("\n" + "="*65)
        print(f" 📊 QWEN EVALUATION REPORT ({total_imgs} Images)")
        print(f" Threshold IoU: {self.iou_threshold}")
        print("="*65)
        print(" 1. LOCALIZATION METRICS (Finding the box)")
        print(f" Precision: {s['precision']:.4f} | Recall: {s['recall']:.4f}")
        print(f" F1-Score:  {s[f'localization_f1_at_{self.iou_threshold*100}']:.4f}")
        print("-" * 65)
        print(f" Successfully Localized (TP):         {s['total_tp']}")
        print(f" Wrongly Predicted Boxes (FP):        {s['total_fp']}  <-- Failed to hit {self.iou_threshold} IoU")
        print(f" Missed Ground Truth Boxes (FN):      {s['total_fn']}")
        print(f" Mean IoU of Successes:               {s['mean_iou']:.4f}")
        print("-" * 65)
        print(" 2. CLASSIFICATION METRICS (Labeling the correctly found boxes)")
        print(f" Accuracy:  {s['classification_accuracy']:.2%}")
        print("="*65)

    def plot_all_conflicts(self, conflicts_json_path: str = "conflicts.json", save_path: str = "all_conflicts_grid.png", cols: int = 4):

            if not os.path.exists(conflicts_json_path):
                print(f"❌ Error: Could not find {conflicts_json_path}.")
                return

            with open(conflicts_json_path, 'r') as f:
                conflicts = json.load(f)

            if not conflicts:
                print("✅ No conflicts found in the JSON. Your model is perfect! Nothing to plot.")
                return

            num_images = len(conflicts)
            cols = min(cols, num_images)
            rows = math.ceil(num_images / cols)

            print(f"Generating a {rows}x{cols} grid for {num_images} conflicting images. This might take a moment...")

            # Create a massive figure. 10x10 inches per image ensures extremely high resolution for zooming.
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))
            
            # Ensure axes is always a flat list we can iterate over
            if num_images == 1:
                axes = [axes]
            elif rows > 1 or cols > 1:
                axes = axes.flatten()

            def extract_bbox_and_label(item, default_prefix):
                if isinstance(item, dict):
                    bbox = item.get("bbox", item.get("bbox_2d", []))
                    label = f"{default_prefix}: {item.get('label', 'unknown')}"
                    return bbox, label
                return item, default_prefix

            for idx, conflict in enumerate(conflicts):
                ax = axes[idx]
                image_path = conflict.get('image')
                
                if not image_path or not os.path.exists(image_path):
                    ax.set_title(f"❌ Missing Image:\n{os.path.basename(str(image_path))}", fontsize=18)
                    ax.axis('off')
                    continue
                    
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw Ground Truths (RED)
                for gt in conflict.get('gts', []):
                    bbox, label = extract_bbox_and_label(gt, "GT")
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
                        cv2.putText(img, label, (x1, max(y1 - 15, 20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

                # Draw Predictions (GREEN)
                for pred in conflict.get('preds', []):
                    bbox, label = extract_bbox_and_label(pred, "Pred")
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(img, label, (x1, min(y2 + 40, img.shape[0] - 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

                ax.imshow(img)
                
                # Create a bold title indicating exactly what went wrong
                title_text = f"FP: {conflict.get('fp_count', 0)} | Misses: {conflict.get('miss_count', 0)}\n{os.path.basename(image_path)}"
                ax.set_title(title_text, fontsize=20, fontweight='bold', pad=15)
                ax.axis("off")

            # Hide any unused subplots (if num_images isn't a perfect multiple of cols)
            for i in range(num_images, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            
            # Save with a high DPI so you can zoom deeply into the final PNG
            plt.savefig(save_path, dpi=200, bbox_inches='tight') 
            print(f"📸 Saved giant conflict grid to {save_path}")
            
            # Free memory (crucial when dealing with massive matplotlib figures)
            plt.close(fig)
