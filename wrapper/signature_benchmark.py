import math
import json
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

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
                # Extract the bbox list from the dictionary format
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

    def evaluate(self, dataset: List[Dict], output_json: str = "conflicts.json") -> Dict:
        metrics = {
            "iou": [], "iop": [], "norm_dist": [], 
            "tp": 0, "fp": 0, "fn": 0, "total_imgs": 0,
            "class_correct": 0, "class_incorrect": 0,
            "confusion": defaultdict(int) # Tracks (GT_label, Pred_label)
        }
        conflicts = []

        for item in tqdm(dataset, desc="Evaluating Qwen"):
            metrics["total_imgs"] += 1
            
            try:
                raw_output = self.model.predict(item['image_path'], "Locate and classify the signatures.")
                preds = self.model.postprocess(raw_output, self.img_size, self.img_size)
                gts = self.model.postprocess(item['groundTruth'], self.img_size, self.img_size)
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                preds, gts = [], [] 

            matches, fps, misses = self.match_predictions_to_ground_truth(
                preds, gts, iou_threshold=self.iou_threshold 
            )
            
            # 1. Localization Metrics
            metrics["tp"] += len(matches)
            metrics["fp"] += len(fps)
            metrics["fn"] += len(misses)

            # 2. Classification Metrics (Only evaluated on correctly localized boxes)
            for match in matches:
                pred_label = match['pred'].get('label', 'unknown')
                gt_label = match['gt'].get('label', 'unknown')
                
                # Track confusion
                metrics["confusion"][f"GT:{gt_label} -> Pred:{pred_label}"] += 1

                if pred_label == gt_label:
                    metrics["class_correct"] += 1
                else:
                    metrics["class_incorrect"] += 1

                # Gather distance metrics
                m = self.evaluate_detection(match['pred']['bbox'], match['gt']['bbox'], self.img_size, self.img_size)
                metrics["iou"].append(m['iou'])
                metrics["iop"].append(m['iop'])
                metrics["norm_dist"].append(m['norm_center_dist'])

            # Log conflicts
            if fps or misses or (metrics["class_incorrect"] > 0):
                conflicts.append({
                    "image": item['image_path'],
                    "preds": preds, "gts": gts,
                    "fp_count": len(fps), "miss_count": len(misses)
                })

        return self._summarize(metrics, conflicts, output_json)

    def _summarize(self, metrics, conflicts, output_json):
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate Classification Accuracy
        total_classified = metrics["class_correct"] + metrics["class_incorrect"]
        class_accuracy = metrics["class_correct"] / total_classified if total_classified > 0 else 0.0

        total_m = len(metrics["iou"])
        avg = lambda x: sum(x) / total_m if total_m > 0 else 0.0
        
        summary = {
            "localization_f1_at_70": round(f1_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou": avg(metrics["iou"]),
            "total_tp": tp, "total_fp": fp, "total_fn": fn,
            "classification_accuracy": round(class_accuracy, 4),
            "confusion_matrix": dict(metrics["confusion"])
        }

        with open(output_json, "w") as f:
            json.dump(conflicts, f, indent=4)
            
        self._print_report(summary, metrics["total_imgs"])
        return summary

    def _print_report(self, s, total_imgs):
        print("\n" + "="*55)
        print(f" 📊 QWEN EVALUATION REPORT ({total_imgs} Images)")
        print(f" Threshold IoU: {self.iou_threshold}")
        print("="*55)
        print(" 1. LOCALIZATION METRICS (Finding the box)")
        print(f" Precision: {s['precision']:.4f} | Recall: {s['recall']:.4f}")
        print(f" F1-Score:  {s['localization_f1_at_70']:.4f}")
        print(f" TP: {s['total_tp']} | FP: {s['total_fp']} | FN: {s['total_fn']}")
        print(f" Mean IoU:  {s['mean_iou']:.4f}")
        print("-" * 55)
        print(" 2. CLASSIFICATION METRICS (Labeling the box)")
        print(f" Accuracy:  {s['classification_accuracy']:.2%} (on correctly localized boxes)")
        if s['confusion_matrix']:
            print("\n Confusion Breakdown:")
            for mapping, count in sorted(s['confusion_matrix'].items()):
                print(f"  - {mapping}: {count}")
        print("="*55)
