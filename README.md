# Authentication elements detection with Qwen2.5-VL

This repository implements a signature, logo and stamp detection pipeline using the **Qwen2.5-VL** Vision-Language Model. It includes a complete workflow for dataset preparation, fine-tuning (optional), and robust evaluation.

## 📁 Project Structure

```
Object_detection_in_documents/
├── utils/                          # Utility scripts and helper functions
│   ├── extract_pdf.py              # PDF to image extraction
│   ├── Qwen_25_LLM.py              # Wrapper to handel loading and inference of the model for HuggingFace
│   ├── signature_benchmark.py              # Wrapper to handel evaluating of the model on the test data
│   └── train_qwen.py               # Script to train the base model 
│
├── download_data/                  # Dataset download and preparation scripts
│   └── download_signature_data.py  # Downloads raw data from Roboflow
├── data_final/                     # Final processed dataset (70% Train / 20% Valid / 10% Test) created after you run the script of donwload
├── pdf_to_image/                   # Output folder for PDF page extractions
├── output/                         # Model inference and evaluation results
├── Inference_Qwen_25_VL.ipynb      # Main Jupyter notebook for inference 
├── main.py                         # Main script entry point
├── pdf_sample.pdf                  # Sample PDF for testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Key Folders

- **`utils/`**: Utility functions including PDF extraction and data processing
- **`download_data/`**: Scripts for downloading and preparing the dataset
- **`data_final/`**: Processed dataset with train/validation/test splits and augmentation
- **`output/`**: Results from model inference and evaluation metrics

---

## 🚀 Quick Start

### Installation

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
python download_data/download_signature_data.py
```

### 2. Run the Notebook

Once the `data/` folder is generated, the Jupyter Notebook usage is straightforward. It will automatically load the test set and run the evaluation pipeline.

```bash
jupyter notebook Inference_Qwen_25_VL.ipynb
```

---
## 📊 Evaluation Metrics

The `SignatureBenchmark` script evaluates the Qwen-VL model's performance across two primary dimensions: **Localization** (how accurately it finds objects) and **Classification** (how accurately it identifies what it found).

### 1. Localization Metrics (Spatial Accuracy)

These metrics evaluate the model's ability to draw bounding boxes exactly where the ground-truth objects (signatures, logos, stamps) are located, independent of whether it guessed the correct label. A prediction is only considered a "True Positive" if its overlap with the ground truth exceeds the configured `iou_threshold` (default is 0.70).

* **IoU (Intersection over Union):** The core spatial metric. It measures the overlap between the predicted bounding box and the ground truth box, divided by their total combined area.
    * **Mean IoU / Median IoU:** The average and median overlap score calculated *only* for successfully matched boxes (True Positives). This indicates how "tight" the bounding boxes are when the model gets it right.
* **Precision:** The percentage of predicted bounding boxes that correctly matched a ground truth box ($\frac{TP}{TP + FP}$). A high precision means the model rarely hallucinates fake objects.
* **Recall:** The percentage of actual ground truth boxes that the model successfully found and bounded ($\frac{TP}{TP + FN}$). A high recall means the model rarely misses real objects.
* **F1-Score (`localization_f1_at_{threshold}`):** The harmonic mean of Precision and Recall. This is the primary single-number metric for overall detection performance at the specified IoU threshold.
* **True Negatives (Document-Level):** Evaluates the model's ability to correctly stay silent. If a document contains *none* of the target objects, and the model predicts *zero* bounding boxes, it is counted as a True Negative.

### 2. Classification Metrics (Semantic Accuracy)

These metrics are calculated **strictly on the successfully localized boxes (True Positives)**. Once the model successfully draws a box around an object, these metrics evaluate if it assigned the correct text label (e.g., confusing a "stamp" for a "logo").

* **Classification Accuracy:** The percentage of successfully localized boxes that were assigned the correct ground-truth label.
* **Confusion Matrix:** A visual plot (`confusion_matrix.png`) showing the exact misclassifications between labels (e.g., how many times a 'logo' was predicted as a 'stamp').

### 3. Detailed Per-Class Metrics

Because object detection datasets are often heavily imbalanced (e.g., many more signatures than logos), the script breaks down performance individually for every label.

* **Per-Class Precision, Recall, & F1:** The standard metrics calculated specifically for a single class (e.g., the F1 score strictly for identifying 'signatures').
* **Class TP (True Positives):** The model correctly localized *and* correctly labeled the specific class.
* **Class FP (False Positives):** The model either hallucinated a box for this class where none existed, *or* it found an object but assigned it this incorrect label.
* **Class FN (False Negatives):** The model failed to localize an object of this class, *or* it localized it but gave it the wrong label.

### 4. Distance & Center Metrics (Secondary)

While IoU is the primary evaluation metric, the script also calculates center-point distances to help diagnose consistent offset or "sliding box" errors.

* **IoP (Intersection over Prediction):** Similar to IoU, but the denominator is only the area of the predicted box. Useful for identifying if predictions are consistently too large or too small.
* **Center Distance (px):** The absolute distance in pixels between the exact center of the predicted box and the center of the ground truth box.
* **Normalized Center Distance:** The center distance divided by the total diagonal length of the image, providing a resolution-agnostic measure of how far off the prediction's center point drifted.

---
