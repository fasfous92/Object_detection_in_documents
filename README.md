# Signature Detection with Qwen2.5-VL

This repository implements a signature detection pipeline using the **Qwen2.5-VL** Vision-Language Model. It includes a complete workflow for dataset preparation, fine-tuning (optional), and robust evaluation.

## 🚀 Quick Start

### Instalation

**Create virtual env**: use the requirement.txt to create and activate the python env


### 1. Prepare the Dataset

**Before running any notebooks**, you must run the data preparation script. This script handles:

1. Downloading the raw dataset from Roboflow.
2. Aggregating all images (ignoring original splits).
3. Creating a custom **70% Train / 20% Valid / 10% Test** split.
4. Applying **3x Augmentation** (Rotation, Noise, Perspective) **only** to the training set.
5. Formatting annotations into the required JSONL format.

```bash
# Run this once to generate the 'data/' folder
python  python download_data/download_signature_data.py

```

### 2. Run the Notebook

Once the `data/` folder is generated, the Jupyter Notebook usage is straightforward. It will automatically load the test set and run the evaluation pipeline.

---

## 📊 Evaluation Metrics

Standard object detection metrics often fail to capture the nuances of signature detection (e.g., human annotators drawing loose boxes vs. models drawing tight boxes).

To get a true picture of performance, we use a composite of three metrics:

### 1. IoU (Intersection over Union)

* **What it is:** The standard academic metric measuring absolute pixel overlap.
* **Why we use it:** To compare against standard benchmarks.
* **Limitation:** It unfairly penalizes "tight crops" where the model ignores white space that a human annotator included.

### 2. IoP (Intersection over Prediction)

* **What it is:** Measures how much of the *prediction* is inside the Ground Truth.
* **Why we use it:** This is our **Precision / Tightness** score.
* **Interpretation:** A score of **1.0** means the predicted box is perfectly inside the ground truth. This confirms the model found the signature and didn't hallucinate background pixels, solving the "conservative annotation" bias.

### 3. Normalized Center Distance

* **What it is:** The distance between the center of the predicted box and the ground truth, divided by the image diagonal.
* **Why we use it:** This measures **Localization Error** independent of box size.
* **Interpretation:** A value of **0.05** means the prediction is off by only **5%** relative to the image size. This proves the model is looking at the right location, even if the box shape isn't perfect.
