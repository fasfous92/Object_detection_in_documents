"""
train_logo.py
=============
Entraînement YOLO11n pour la détection de logos — dataset IIIT-AR-13K (Kaggle).
Pipeline complet : téléchargement Kaggle → VOC XML → YOLO TXT → entraînement.

Usage :
    python train_logo.py
    python train_logo.py --epochs 100 --batch 32 --bg-rate 0.05

Inférence depuis un notebook :
    from train_logo import load_model, predict
    model = load_model()
    results = predict(model, "chemin/vers/image.jpg")
"""

import os
import json
import glob
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
KAGGLE_DATASET      = "gabrieletazza/iiitar13k"
KAGGLE_DOWNLOAD_DIR = "dataset_kaggle"
OUTPUT_DIR          = Path("logo_dataset")
TARGET_LABELS       = {"logo"}
BACKGROUND_KEEP_RATE = 0.05  # 5 % des images sans logo conservées

PROJECT_NAME = "logo_project"
RUN_NAME     = "logo_yolov11n"

# ── Nom du modèle exporté ─────────────────────────────────────────────────────
MODEL_LOGO_PATH = "best_model_logo_v2.pt"


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
# Étape 1 — Téléchargement Kaggle
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset():
    os.makedirs(KAGGLE_DOWNLOAD_DIR, exist_ok=True)
    print("Téléchargement IIIT-AR-13K depuis Kaggle…")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} "
              f"-p {KAGGLE_DOWNLOAD_DIR} --unzip")
    print("Téléchargement terminé.")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Conversion VOC XML → JSON intermédiaire
# ─────────────────────────────────────────────────────────────────────────────

def _index_files(root_dir):
    index = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            index[f] = os.path.join(root, f)
    return index


def _parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size  = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text.lower().strip()
        if label not in TARGET_LABELS:
            continue
        bnd  = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append({
            "label": label,
            "bbox" : [xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h]
        })
    return boxes


def _convert_split(image_root, xml_root, output_json,
                   bg_keep_rate=BACKGROUND_KEEP_RATE, seed=42):
    random.seed(seed)
    image_index = _index_files(image_root)
    xml_index   = _index_files(xml_root)

    data             = []
    count_with_logo  = 0
    count_background = 0
    count_dropped    = 0

    for xml_name, xml_path in tqdm(xml_index.items(),
                                   desc=os.path.basename(output_json)):
        image_name = os.path.splitext(xml_name)[0] + ".jpg"
        if image_name not in image_index:
            continue
        boxes = _parse_voc_xml(xml_path)
        if not boxes:
            if random.random() > bg_keep_rate:
                count_dropped += 1
                continue
        data.append({
            "file_name"   : image_name,
            "image_path"  : image_index[image_name],
            "annotations" : boxes
        })
        if boxes:
            count_with_logo  += 1
        else:
            count_background += 1

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Total gardé   : {len(data)}")
    print(f"  Avec logo     : {count_with_logo}")
    print(f"  Sans logo     : {count_background}")
    print(f"  Supprimées    : {count_dropped}")
    print(f"  Sauvegardé    : {output_json}\n")


def convert_voc_to_json(bg_keep_rate=BACKGROUND_KEEP_RATE):
    # Nettoyage des runs précédents
    for folder in ["logo_dataset"]:
        p = Path(folder)
        if p.exists():
            shutil.rmtree(p)
    for json_file in ["rtdetr_train.json", "rtdetr_val.json", "rtdetr_test.json"]:
        p = Path(json_file)
        if p.exists():
            p.unlink()

    base_dir = KAGGLE_DOWNLOAD_DIR
    print("=== TRAIN ===")
    _convert_split(
        image_root  = os.path.join(base_dir, "training_images"),
        xml_root    = os.path.join(base_dir, "training_xml"),
        output_json = "rtdetr_train.json",
        bg_keep_rate = bg_keep_rate,
    )
    print("=== VAL ===")
    _convert_split(
        image_root  = os.path.join(base_dir, "validation_images"),
        xml_root    = os.path.join(base_dir, "validation_xml"),
        output_json = "rtdetr_val.json",
        bg_keep_rate = bg_keep_rate,
    )
    print("=== TEST ===")
    _convert_split(
        image_root  = os.path.join(base_dir, "test_images"),
        xml_root    = os.path.join(base_dir, "test_xml"),
        output_json = "rtdetr_test.json",
        bg_keep_rate = bg_keep_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Conversion JSON → YOLO TXT
# ─────────────────────────────────────────────────────────────────────────────

def convert_json_to_yolo():
    SPLITS = {
        "train": "rtdetr_train.json",
        "val"  : "rtdetr_val.json",
        "test" : "rtdetr_test.json",
    }

    for split_name, json_path in SPLITS.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        with open(json_path) as f:
            data = json.load(f)

        for sample in data:
            shutil.copy(sample["image_path"], img_out / sample["file_name"])
            stem     = Path(sample["file_name"]).stem
            lbl_file = lbl_out / (stem + ".txt")
            with open(lbl_file, "w") as f:
                for ann in sample["annotations"]:
                    xmin, ymin, xmax, ymax = ann["bbox"]
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    width    =  xmax - xmin
                    height   =  ymax - ymin
                    f.write(f"0 {x_center:.6f} {y_center:.6f} "
                            f"{width:.6f} {height:.6f}\n")

        print(f"[{split_name:5s}]  {len(data):4d} images → {OUTPUT_DIR / split_name}")

    print("Structure YOLO prête !")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Création du data.yaml
# ─────────────────────────────────────────────────────────────────────────────

def create_yaml():
    data_cfg = {
        "path" : str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val"  : "val/images",
        "test" : "test/images",
        "nc"   : 1,
        "names": ["logo"],
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    print(f"data.yaml créé : {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(yaml_path, epochs=50, imgsz=640, batch=16, patience=10, save_period=10):
    model = YOLO("yolo11n.pt")
    model.train(
        data        = str(yaml_path),
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        project     = PROJECT_NAME,
        name        = RUN_NAME,
        patience    = patience,
        save_period = save_period,
    )
    print("Entraînement terminé !")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def _get_best_weights():
    run_dirs = sorted(
        glob.glob(os.path.join("runs", "detect", PROJECT_NAME, RUN_NAME + "*/")),
        key=os.path.getmtime
    )
    if not run_dirs:
        raise FileNotFoundError("Aucun run trouvé dans runs/detect/.")
    best_weights = os.path.join(run_dirs[-1], "weights", "best.pt")
    print(f"Run        : {run_dirs[-1]}")
    print(f"Modèle     : {best_weights}")
    return best_weights


def validate(best_model, yaml_path):
    metrics = best_model.val(data=str(yaml_path))
    print("=" * 35)
    print("         RÉSULTATS")
    print("=" * 35)
    print(f"  mAP@50    : {metrics.box.map50:.3f}")
    print(f"  mAP@50:95 : {metrics.box.map:.3f}")
    print(f"  Précision : {metrics.box.mp:.3f}")
    print(f"  Rappel    : {metrics.box.mr:.3f}")
    print("=" * 35)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 — Inférence & visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(best_model):
    test_dir = str(OUTPUT_DIR / "test" / "images")
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        test_dir = str(OUTPUT_DIR / "val" / "images")
        print("Dossier test vide → utilisation de val")

    print(f"Prédiction sur : {test_dir}")
    results = best_model.predict(source=test_dir, conf=0.25, save=True)
    return results


def visualize(results, n=10):
    for i, result in enumerate(results[:n]):
        res_img  = result.plot()
        n_logos  = len(result.boxes)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {i + 1} — {n_logos} logo(s) détecté(s)", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Étape 8 — Export / sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def export_model(src_weights_path=None):
    """Copie best.pt vers MODEL_LOGO_PATH."""
    if src_weights_path is None:
        src_weights_path = _get_best_weights()

    shutil.copy(src_weights_path, MODEL_LOGO_PATH)
    print(f"Modèle sauvegardé : {MODEL_LOGO_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 9 — Métriques IoU & Precision/Recall/F1
# ─────────────────────────────────────────────────────────────────────────────

def _get_eval_dirs():
    test_images_dir = OUTPUT_DIR / "test"  / "images"
    test_labels_dir = OUTPUT_DIR / "test"  / "labels"
    val_images_dir  = OUTPUT_DIR / "val"   / "images"
    val_labels_dir  = OUTPUT_DIR / "val"   / "labels"

    if (test_images_dir.exists() and test_labels_dir.exists()
            and len(os.listdir(test_images_dir)) > 0):
        return test_images_dir, test_labels_dir, "test"
    return val_images_dir, val_labels_dir, "val"


def compute_iou_metrics(best_model):
    images_dir, labels_dir, split_used = _get_eval_dirs()
    print(f"Split utilisé pour le calcul IoU : {split_used}")

    image_paths = sorted(glob.glob(str(images_dir / "*.*")))
    all_ious    = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        base_name  = Path(img_path).stem
        label_path = labels_dir / f"{base_name}.txt"

        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:
                        cls_id, box = yolo_to_xyxy(values, img_w, img_h)
                        gt_boxes.append((cls_id, box))

        pred_result = best_model.predict(source=img_path, conf=0.25, verbose=False)[0]
        pred_boxes  = []
        if pred_result.boxes is not None and len(pred_result.boxes) > 0:
            for box, cls_id in zip(pred_result.boxes.xyxy.cpu().numpy(),
                                   pred_result.boxes.cls.cpu().numpy()):
                pred_boxes.append((int(cls_id), box.tolist()))

        used_preds = set()
        for gt_cls, gt_box in gt_boxes:
            best_iou, best_j = -1, -1
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


def compute_precision_recall(best_model, iou_threshold=0.70):
    images_dir, labels_dir, _ = _get_eval_dirs()
    image_paths = sorted(glob.glob(str(images_dir / "*.*")))

    matched_ious = []
    total_tp = total_fp = total_fn = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        base_name  = Path(img_path).stem
        label_path = labels_dir / f"{base_name}.txt"

        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) == 5:
                        gt_boxes.append(yolo_to_xyxy(vals, img_w, img_h))

        pred_result = best_model.predict(source=img_path, conf=0.25, verbose=False)[0]
        pred_boxes  = []
        if pred_result.boxes is not None and len(pred_result.boxes) > 0:
            for box, cls in zip(pred_result.boxes.xyxy.cpu().numpy(),
                                pred_result.boxes.cls.cpu().numpy()):
                pred_boxes.append((int(cls), box.tolist()))

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
                matched_ious.append(best_iou)
                if best_iou > iou_threshold:
                    total_tp += 1
                else:
                    total_fp += 1
                    total_fn += 1
            else:
                total_fn += 1

        total_fp += len(pred_boxes) - len(used_preds)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("=" * 45)
    print("      MÉTRIQUES IoU & DÉTECTION")
    print("=" * 45)
    if matched_ious:
        print(f"  Paires appariées          : {len(matched_ious)}")
        print(f"  IoU moyenne               : {np.mean(matched_ious):.4f}")
        print(f"  IoU médiane               : {np.median(matched_ious):.4f}")
    print(f"\n  Seuil TP                  : IoU > {iou_threshold:.0%}")
    print(f"  TP                        : {total_tp}")
    print(f"  FP                        : {total_fp}")
    print(f"  FN                        : {total_fn}")
    print(f"\n  Precision                 : {precision:.4f}")
    print(f"  Recall                    : {recall:.4f}")
    print(f"  F1 score                  : {f1:.4f}")
    print("=" * 45)

    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# API d'inférence — utilisable depuis un notebook
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path=MODEL_LOGO_PATH):
    """
    Charge le modèle depuis les poids sauvegardés.

    Exemple notebook :
        from train_logo import load_model, predict
        model = load_model()
        model = load_model("mon_chemin/best.pt")
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Poids introuvables : {weights_path}\n"
            "Lancez d'abord train_logo.py pour entraîner le modèle."
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
        results = predict(model, "logo.jpg")
        for r in results:
            print(r.boxes.xyxy)
            plt.imshow(r.plot()[...,::-1])
    """
    return model.predict(source=source, conf=conf, save=save, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement détection de logos")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--bg-rate",  type=float, default=BACKGROUND_KEEP_RATE,
                        help="Taux de conservation des images sans logo (défaut 0.05)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Ne pas retélécharger le dataset si déjà présent")
    args = parser.parse_args()

    # 1. Téléchargement
    if not args.skip_download:
        download_dataset()

    # 2. VOC XML → JSON
    convert_voc_to_json(bg_keep_rate=args.bg_rate)

    # 3. JSON → YOLO TXT
    convert_json_to_yolo()

    # 4. data.yaml
    yaml_path = create_yaml()

    # 5. Entraînement
    train(yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # 6. Charger les meilleurs poids
    best_weights = _get_best_weights()
    best_model   = YOLO(best_weights)

    # 7. Évaluation
    validate(best_model, yaml_path)

    # 8. Inférence + visualisation
    results = run_inference(best_model)
    visualize(results)

    # 9. Export
    export_model(best_weights)

    # 10. Métriques
    compute_iou_metrics(best_model)
    compute_precision_recall(best_model)


if __name__ == "__main__":
    main()
