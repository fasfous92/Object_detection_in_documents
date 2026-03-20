"""
train_signature.py
==================
Entraînement YOLO11n pour la détection de signatures.
Conçu pour Lightning AI.

Usage :
    python train_signature.py
    python train_signature.py --epochs 100 --batch 32

Inférence depuis un notebook :
    from train_signature import load_model, predict
    model = load_model()
    results = predict(model, "chemin/vers/image.jpg")
"""

import os
import shutil
import zipfile
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR    = "/teamspace/studios/this_studio"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RUNS_DIR    = os.path.join(BASE_DIR, "runs")
ZIP_PATH    = os.path.join(BASE_DIR, "Signature Detection.v1i.yolov8.zip")

# ── Nom du modèle exporté ─────────────────────────────────────────────────────
MODEL_SIGNATURE_PATH = "best_model_signature.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers IoU
# ─────────────────────────────────────────────────────────────────────────────

def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(cls), [x1, y1, x2, y2]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"ZIP introuvable : {ZIP_PATH}\n"
            "Uploadez votre dataset via l'interface Lightning AI avant de continuer."
        )

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print(f"Extraction terminée dans : {DATASET_DIR}")

    train_labels_dir = os.path.join(DATASET_DIR, "train", "labels")
    if os.path.exists(train_labels_dir):
        print(f"Nombre de labels d'entraînement : {len(os.listdir(train_labels_dir))}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(epochs=50, imgsz=640, batch=16):
    model = YOLO("yolo11n.pt")
    model.train(
        data    = os.path.join(DATASET_DIR, "data.yaml"),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        project = os.path.join(RUNS_DIR, "detect"),
        name    = "signature_v11_final",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(best_model):
    metrics = best_model.val(data=os.path.join(DATASET_DIR, "data.yaml"))
    print(f"Précision (mAP@50) : {metrics.box.map50:.3f}")

    metrics_test = best_model.val(
        data  = os.path.join(DATASET_DIR, "data.yaml"),
        split = "test"
    )
    return metrics, metrics_test


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Inférence
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(best_model):
    if os.path.exists(os.path.join(DATASET_DIR, "test", "images")):
        test_images = os.path.join(DATASET_DIR, "test", "images")
    else:
        test_images = os.path.join(DATASET_DIR, "valid", "images")

    print(f"Dossier utilisé pour l'inférence : {test_images}")
    results = best_model.predict(source=test_images, conf=0.25, save=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(results, n=10):
    for result in results[:n]:
        res_img = result.plot()
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(res_img_rgb)
        plt.axis("off")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Export / sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def export_model(src_weights_path=None):
    """Copie best.pt vers MODEL_SIGNATURE_PATH."""
    if src_weights_path is None:
        src_weights_path = os.path.join(
            RUNS_DIR, "detect", "signature_v11_final", "weights", "best.pt"
        )

    if os.path.exists(src_weights_path):
        shutil.copy(src_weights_path, MODEL_SIGNATURE_PATH)
        print(f"Modèle sauvegardé : {MODEL_SIGNATURE_PATH}")
    else:
        print(f"Fichier introuvable : {src_weights_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 — Métriques IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_metrics(best_model):
    if (os.path.exists(os.path.join(DATASET_DIR, "test", "images")) and
            os.path.exists(os.path.join(DATASET_DIR, "test", "labels"))):
        images_dir = os.path.join(DATASET_DIR, "test", "images")
        labels_dir = os.path.join(DATASET_DIR, "test", "labels")
    else:
        images_dir = os.path.join(DATASET_DIR, "valid", "images")
        labels_dir = os.path.join(DATASET_DIR, "valid", "labels")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    all_ious = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:
                        cls_id, box = yolo_to_xyxy(values, img_w, img_h)
                        gt_boxes.append((cls_id, box))

        pred_result = best_model.predict(source=img_path, conf=0.25, verbose=False)[0]
        pred_boxes = []
        if pred_result.boxes is not None and len(pred_result.boxes) > 0:
            for box, cls_id in zip(pred_result.boxes.xyxy.cpu().numpy(),
                                   pred_result.boxes.cls.cpu().numpy()):
                pred_boxes.append((int(cls_id), box.tolist()))

        used_preds = set()
        for gt_cls, gt_box in gt_boxes:
            best_iou, best_j = 0.0, -1
            for j, (pred_cls, pred_box) in enumerate(pred_boxes):
                if j in used_preds or pred_cls != gt_cls:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1:
                used_preds.add(best_j)
                all_ious.append(best_iou)

    if all_ious:
        print(f"IoU moyenne  des éléments identifiés : {np.mean(all_ious):.4f}")
        print(f"IoU médiane  des éléments identifiés : {np.median(all_ious):.4f}")
        print(f"Nombre d'éléments appariés           : {len(all_ious)}")
    else:
        print("Aucun élément apparié.")

    return all_ious


def compute_precision_recall(best_model, iou_threshold=0.70):
    if (os.path.exists(os.path.join(DATASET_DIR, "test", "images")) and
            os.path.exists(os.path.join(DATASET_DIR, "test", "labels"))):
        images_dir = os.path.join(DATASET_DIR, "test", "images")
        labels_dir = os.path.join(DATASET_DIR, "test", "labels")
    else:
        images_dir = os.path.join(DATASET_DIR, "valid", "images")
        labels_dir = os.path.join(DATASET_DIR, "valid", "labels")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    total_tp = total_fp = total_fn = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:
                        cls_id, box = yolo_to_xyxy(values, img_w, img_h)
                        gt_boxes.append((cls_id, box))

        pred_result = best_model.predict(source=img_path, conf=0.25, verbose=False)[0]
        pred_boxes = []
        if pred_result.boxes is not None and len(pred_result.boxes) > 0:
            for box, cls_id in zip(pred_result.boxes.xyxy.cpu().numpy(),
                                   pred_result.boxes.cls.cpu().numpy()):
                pred_boxes.append((int(cls_id), box.tolist()))

        matched_gt   = set()
        matched_pred = set()

        for i, (gt_cls, gt_box) in enumerate(gt_boxes):
            best_iou, best_j = 0.0, -1
            for j, (pred_cls, pred_box) in enumerate(pred_boxes):
                if j in matched_pred or pred_cls != gt_cls:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1 and best_iou >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_j)
                total_tp += 1

        total_fn += len(gt_boxes)   - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"Seuil IoU pour True Positive : {iou_threshold:.0%}")
    print(f"TP : {total_tp}  |  FP : {total_fp}  |  FN : {total_fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# API d'inférence — utilisable depuis un notebook
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path=MODEL_SIGNATURE_PATH):
    """
    Charge le modèle depuis les poids sauvegardés.

    Exemple notebook :
        from train_signature import load_model, predict
        model = load_model()                          # poids par défaut
        model = load_model("mon_chemin/best.pt")      # poids custom
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Poids introuvables : {weights_path}\n"
            "Lancez d'abord train_signature.py pour entraîner le modèle."
        )
    return YOLO(weights_path)


def predict(model, source, conf=0.25, save=False, verbose=False):
    """
    Lance une prédiction sur une image, un dossier ou une URL.

    Paramètres
    ----------
    model   : YOLO  — modèle chargé via load_model()
    source  : str   — chemin image / dossier / URL
    conf    : float — seuil de confiance (défaut 0.25)
    save    : bool  — sauvegarder les résultats annotés
    verbose : bool  — afficher les logs YOLO

    Retourne
    --------
    Liste de résultats ultralytics (result.boxes, result.plot(), …)

    Exemple notebook :
        results = predict(model, "image.jpg")
        for r in results:
            print(r.boxes.xyxy)          # coordonnées
            plt.imshow(r.plot()[...,::-1])
    """
    return model.predict(source=source, conf=conf, save=save, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement détection de signatures")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Ne pas réextraire le dataset si déjà présent")
    args = parser.parse_args()

    # 1. Dataset
    if not args.skip_dataset:
        load_dataset()

    # 2. Entraînement
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # 3. Charger les meilleurs poids
    weights_path = os.path.join(
        RUNS_DIR, "detect", "signature_v11_final", "weights", "best.pt"
    )
    best_model = YOLO(weights_path)

    # 4. Validation
    validate(best_model)

    # 5. Inférence + visualisation
    results = run_inference(best_model)
    visualize(results)

    # 6. Export
    export_model(weights_path)

    # 7. Métriques
    compute_iou_metrics(best_model)
    compute_precision_recall(best_model)


if __name__ == "__main__":
    main()
