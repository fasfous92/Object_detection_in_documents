# Object Detection in Documents (ROD-MLLM)

This project implements a **Retrieval-based Object Detection Multimodal LLM (ROD-MLLM)** specialized for detecting signatures in scanned documents. It combines a vision encoder (CLIP), an open-vocabulary locator (OWLv2), and a large language model (Vicuna) to achieve precise localization and contextual understanding.

## Implementation Phase

### 1. Environment Setup
Install the necessary dependencies to run the training and inference scripts.
```bash
pip install -r requirements.txt

```

### 2. Data Preparation

The training pipeline consists of two stages. You must download and prepare the data for each stage separately.

**Stage 1: Alignment (Projector Pre-training)**
This stage aligns the visual encoder (CLIP) with the LLM using the COCO dataset. The model learns to map visual features to text embeddings.

```bash
# Download COCO dataset
python src/download_dataset_stage1.py

# Cache image embeddings for 10x faster training
python src/cache_data.py

```

**Stage 2: Fine-Tuning (Signature Detection)**
This stage fine-tunes the **Locator** and the **LLM** using the augmented Signature dataset (Tobacco800 + DL-Signatures). The script downloads the data from Roboflow and applies augmentation (noise, blur, rotation).

```bash
# Download, Augment (3x), and Format the signature dataset
python src/prepare_and_augment.py

```

*Note: This creates the `data/signatures_augmented` folder ready for training.*

### 3. Training Stage 1 (Alignment)

Train the **Projectors** (MLP) to connect the frozen Vision Encoder to the frozen LLM. This establishes the "vision-language" communication channel.

```bash
python -m src.train \
    --stage 1 \
    --epochs 2 \
    --batch_size 32 \
    --lr 1e-3

```

### 4. Training Stage 2 (Signature Fine-Tuning)

Train the **Locator (OWLv2)** and **LLM (LoRA)**.

* **Unfreezes the Locator:** Allows the model to adapt to document-specific noise (scans, blur, paper texture).
* **Instruction Tuning:** Teaches the LLM to specifically look for "signatures" rather than generic objects.

```bash
python -m src.train_signature \
    --stage 2 \
    --train_annotation data/signatures_augmented/rod_train.json \
    --val_annotation data/signatures_augmented/rod_valid.json \
    --image_dir data/signatures_augmented/images \
    --resume output_model/stage1_epoch2.pt \
    --epochs 5 \
    --lr 1e-5 \
    --per_device_batch_size 2 \
    --target_batch_size 32

```

*Arguments:*

* `--resume`: Points to the best checkpoint from Stage 1.
* `--lr 1e-5`: Uses a lower learning rate to stabilize the fine-tuning of the sensitive Locator module.

### 5. Evaluation & Visualization

Run inference on the test set to calculate the Mean IoU and generate visualization plots.

```bash
python -m src.test_signature

```

**Output:**

* **Console:** Prints the final Mean IoU score.
* **Visuals:** Check the `results/` folder for:
* `best_predictions.png`: Top 5 accurate detections (Green = Truth, Red = Prediction).
* `worst_predictions.png`: Bottom 5 failing cases (useful for debugging).
