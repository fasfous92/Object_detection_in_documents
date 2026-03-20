"""
train_stamp.py
==============
Entraînement YOLO11n pour la détection de tampons (stamps).
Conçu pour un environnement local ou serveur (compatible Lightning AI / Colab).

Usage :
    python train_stamp.py
    python train_stamp.py --epochs 100 --batch 32 --zip Stamp_detection.v8i.yolov8

Inférence depuis un notebook :
    from train_stamp import load_model, predict
    model = load_model()
    results = predict(model, "chemin/vers/image.jpg")
"""

import os
import subprocess
import glob
import shutil
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
ZIP_NAME     = "Stamp_detection.v8i.yolov8"
EXTRACT_PATH = "./"
PROJECT_NAME = "stamps_project"
RUN_NAME     = "stamp_yolov11"

# ── Nom du modèle exporté ─────────────────────────────────────────────────────
MODEL_STAMP_PATH = "best_model_stamp.pt"


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
    inter_w    = max(0, xB - xA)
    inter_h    = max(0, yB - yA)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Extraction & configuration YAML
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(zip_name=ZIP_NAME, extract_path=EXTRACT_PATH):
    """
    Extrait le ZIP du dataset et corrige les chemins dans data.yaml.
    Retourne (yaml_path, data_root).
    """
    # Nettoyage et extraction via commande système (évite l'erreur 22)
    subprocess.run(["rm", "-rf", extract_path], check=False)
    subprocess.run(["unzip", "-q", zip_name, "-d", extract_path], check=True)
    print(f"Extraction terminée dans : {extract_path}")

    # Recherche de data.yaml
    yaml_path = None
    data_root = None
    for root, _, files in os.walk(extract_path):
        if "data.yaml" in files:
            yaml_path = os.path.join(root, "data.yaml")
            data_root = root
            break

    if yaml_path is None:
        raise FileNotFoundError("data.yaml introuvable après extraction.")

    # Correction des chemins absolus
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    data_cfg["path"] = os.path.abspath(data_root)

    with open(yaml_path, "w") as f:
        yaml.dump(data_cfg, f)

    print(f"Fichier YAML configuré : {yaml_path}")
    return yaml_path, data_root


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(yaml_path, epochs=50, imgsz=640, batch=16):
    model = YOLO("yolo11n.pt")
    model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        project = PROJECT_NAME,
        name    = RUN_NAME,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(best_model, yaml_path):
    metrics = best_model.val(data=yaml_path)
    print(f"Précision (mAP@50) : {metrics.box.map50:.3f}")

    # Test split si disponible
    try:
        metrics_test = best_model.val(data=yaml_path, split="test")
        print(f"mAP@50 (test)      : {metrics_test.box.map50:.3f}")
    except Exception:
        pass

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Inférence
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(best_model, data_root):
    test_path  = os.path.join(data_root, "test",  "images")
    valid_path = os.path.join(data_root, "valid", "images")

    if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
        test_images = test_path
    else:
        test_images = valid_path

    print(f"Dossier utilisé pour l'inférence : {test_images}")
    results = best_model.predict(source=test_images, conf=0.25, save=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(results, n=10):
    for result in results[:n]:
        res_img = result.plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Export / sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def export_model(src_weights_path=None):
    """Copie best.pt vers MODEL_STAMP_PATH."""
    if src_weights_path is None:
        pattern = os.path.join(
            "runs", "detect", PROJECT_NAME, RUN_NAME + "*", "weights", "best.pt"
        )
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not matches:
            print("Aucun poids trouvé — vérifiez le dossier runs/.")
            return
        src_weights_path = matches[-1]

    shutil.copy(src_weights_path, MODEL_STAMP_PATH)
    print(f"Modèle sauvegardé : {MODEL_STAMP_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 — Métriques IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_metrics(best_model, data_root):
    test_images_dir = os.path.join(data_root, "test",  "images")
    test_labels_dir = os.path.join(data_root, "test",  "labels")
    valid_images_dir = os.path.join(data_root, "valid", "images")
    valid_labels_dir = os.path.join(data_root, "valid", "labels")

    if (os.path.exists(test_images_dir) and
            os.path.exists(test_labels_dir) and
            len(os.listdir(test_images_dir)) > 0):
        images_dir = test_images_dir
        labels_dir = test_labels_dir
    else:
        images_dir = valid_images_dir
        labels_dir = valid_labels_dir

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
        print(f"IoU moyenne des éléments identifiés : {np.mean(all_ious):.4f}")
        print(f"IoU médiane des éléments identifiés : {np.median(all_ious):.4f}")
        print(f"Nombre d'éléments appariés          : {len(all_ious)}")
    else:
        print("Aucun élément apparié.")

    return all_ious


def compute_precision_recall(best_model, data_root, iou_threshold=0.70):
    test_images_dir  = os.path.join(data_root, "test",  "images")
    test_labels_dir  = os.path.join(data_root, "test",  "labels")
    valid_images_dir = os.path.join(data_root, "valid", "images")
    valid_labels_dir = os.path.join(data_root, "valid", "labels")

    if (os.path.exists(test_images_dir) and
            os.path.exists(test_labels_dir) and
            len(os.listdir(test_images_dir)) > 0):
        images_dir = test_images_dir
        labels_dir = test_labels_dir
    else:
        images_dir = valid_images_dir
        labels_dir = valid_labels_dir

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    total_gt = total_pred = total_tp = 0

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

        total_gt   += len(gt_boxes)
        total_pred += len(pred_boxes)

        used_preds = set()
        for gt_cls, gt_box in gt_boxes:
            best_iou, best_j = 0.0, -1
            for j, (pred_cls, pred_box) in enumerate(pred_boxes):
                if j in used_preds or pred_cls != gt_cls:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1 and best_iou >= iou_threshold:
                used_preds.add(best_j)
                total_tp += 1

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall    = total_tp / total_gt   if total_gt   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"Seuil IoU pour TP  : {iou_threshold:.0%}")
    print(f"Total GT           : {total_gt}")
    print(f"Total prédictions  : {total_pred}")
    print(f"True Positives     : {total_tp}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 score           : {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# API d'inférence — utilisable depuis un notebook
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path=MODEL_STAMP_PATH):
    """
    Charge le modèle depuis les poids sauvegardés.

    Exemple notebook :
        from train_stamp import load_model, predict
        model = load_model()
        model = load_model("mon_chemin/best.pt")
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Poids introuvables : {weights_path}\n"
            "Lancez d'abord train_stamp.py pour entraîner le modèle."
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
        results = predict(model, "tampon.jpg")
        for r in results:
            print(r.boxes.xyxy)
            plt.imshow(r.plot()[...,::-1])
    """
    return model.predict(source=source, conf=conf, save=save, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement détection de tampons")
    parser.add_argument("--zip",    type=str, default=ZIP_NAME,
                        help="Nom du fichier ZIP du dataset (sans extension)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()

    # 1. Dataset
    yaml_path, data_root = load_dataset(zip_name=args.zip)

    # 2. Entraînement
    train(yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # 3. Charger les meilleurs poids
    pattern = os.path.join(
        "runs", "detect", PROJECT_NAME, RUN_NAME + "*", "weights", "best.pt"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    best_model = YOLO(matches[-1])

    # 4. Validation
    validate(best_model, yaml_path)

    # 5. Inférence + visualisation
    results = run_inference(best_model, data_root)
    visualize(results)

    # 6. Export
    export_model(matches[-1])

    # 7. Métriques
    compute_iou_metrics(best_model, data_root)
    compute_precision_recall(best_model, data_root)


if __name__ == "__main__":
    main()
