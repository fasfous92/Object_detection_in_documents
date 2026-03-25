
import os
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "OpenGVLab/InternVL3_5-4B-HF"

# Dataset root (expected to already exist locally)
# Example structure:
# data/Stamp-detection-1/
#   train/images
#   train/labels
#   valid/images
#   valid/labels
#   test/images
#   test/labels
DATA_ROOT = Path("data/Stamp-detection-1")

# Converted InternVL JSONL output
OUT_DIR = Path("data/internvl_stamp_data")

# Pretrained adapter zip stored in repo
PRETRAINED_ADAPTER_ZIP = Path("weights/internvl_vision_LORA_stamp_ft.zip")

# Temporary unzip directory
PRETRAINED_ADAPTER_UNZIP_DIR = Path("weights/internvl_vision_LORA_stamp_ft")

# Final saved adapter after continued fine-tuning
OUTPUT_DIR = Path("outputs/internvl_stamp_lora_ft/final")

PROMPT = "Detect and label all stamps in the image. Output each stamp with its bounding boxes."
TARGET_LABEL = "stamp"

# Training hyperparameters
EPOCHS = 4
BATCH = 1
ACCUM = 16
LR = 2e-4
MAX_DYNAMIC_PATCH = 12
REPEAT_TIME = 1

# Optional quick debug subset
USE_SUBSET = False
TRAIN_SUBSET_DIVISOR = 5
VALID_SUBSET_DIVISOR = 2

# ============================================================
# Safety / image settings
# ============================================================

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# Helpers
# ============================================================

def find_adapter_dir(root: Path) -> Path:
    """
    Find the actual adapter directory after unzipping.
    Prefer a nested 'final' folder if it exists and contains adapter files.
    """
    if not root.exists():
        raise FileNotFoundError(f"Unzip directory does not exist: {root}")

    candidate_dirs = []

    if root.is_dir():
        candidate_dirs.append(root)
        candidate_dirs.extend([p for p in root.rglob("*") if p.is_dir()])

    def looks_like_adapter_dir(p: Path) -> bool:
        files = {x.name for x in p.iterdir()} if p.exists() and p.is_dir() else set()
        return (
            "adapter_config.json" in files
            or "adapter_model.safetensors" in files
            or ("config.json" in files and "adapter_model.safetensors" in files)
        )

    final_dirs = [p for p in candidate_dirs if p.name == "final" and looks_like_adapter_dir(p)]
    if final_dirs:
        return final_dirs[0]

    adapter_dirs = [p for p in candidate_dirs if looks_like_adapter_dir(p)]
    if adapter_dirs:
        return adapter_dirs[0]

    raise FileNotFoundError(
        f"Could not find a valid adapter directory inside: {root}"
    )


def unzip_adapter(zip_path: Path, extract_dir: Path) -> Path:
    """
    Unzip the pretrained adapter and return the real adapter directory.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing adapter zip: {zip_path}")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    adapter_dir = find_adapter_dir(extract_dir)
    return adapter_dir


def yolo_to_xyxy(line: str, width: int, height: int) -> Optional[List[float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    _, xc, yc, w, h = parts
    xc, yc, w, h = map(float, [xc, yc, w, h])

    x1 = (xc - w / 2.0) * width
    y1 = (yc - h / 2.0) * height
    x2 = (xc + w / 2.0) * width
    y2 = (yc + h / 2.0) * height

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return [x1, y1, x2, y2]


def norm_to_1000(xyxy: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = xyxy
    return [
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
        int(round(x2 / width * 1000)),
        int(round(y2 / height * 1000)),
    ]


def build_answer(grouped: Dict[str, List[List[int]]]) -> str:
    lines = []
    for label, boxes in grouped.items():
        lines.append(f"<ref>{label}</ref><box>{boxes}</box>")
    return "\n".join(lines) + ("\n" if lines else "")


def collect_image_paths(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(images_dir.glob(ext))
    return sorted(image_paths)


def convert_split(split_name: str) -> tuple[Path, int]:
    images_dir = DATA_ROOT / split_name / "images"
    labels_dir = DATA_ROOT / split_name / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    image_paths = collect_image_paths(images_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_DIR / f"{split_name}.jsonl"
    n_written = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(tqdm(image_paths, desc=f"Convert {split_name}")):
            with Image.open(img_path) as im:
                width, height = im.size

            label_path = labels_dir / f"{img_path.stem}.txt"
            grouped: Dict[str, List[List[int]]] = {}

            if label_path.exists():
                lines = label_path.read_text(encoding="utf-8").strip().splitlines()
                boxes = []

                for line in lines:
                    parsed = yolo_to_xyxy(line, width, height)
                    if parsed is None:
                        continue
                    boxes.append(norm_to_1000(parsed, width, height))

                if boxes:
                    grouped[TARGET_LABEL] = boxes

            rel_image = f"{split_name}/images/{img_path.name}"
            answer = build_answer(grouped)

            sample = {
                "id": f"{split_name}_{idx}",
                "image": rel_image,
                "width": int(width),
                "height": int(height),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": answer if answer.strip() else "No stamps found.\n"},
                ],
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1

    return out_jsonl, n_written


def write_meta(train_jsonl: Path, valid_jsonl: Path, test_jsonl: Path,
               n_train: int, n_valid: int, n_test: int) -> Path:
    meta = {
        "stamp_train": {
            "root": str(DATA_ROOT),
            "annotation": str(train_jsonl),
            "data_augment": False,
            "max_dynamic_patch": MAX_DYNAMIC_PATCH,
            "repeat_time": REPEAT_TIME,
            "length": n_train,
        },
        "stamp_valid": {
            "root": str(DATA_ROOT),
            "annotation": str(valid_jsonl),
            "data_augment": False,
            "max_dynamic_patch": MAX_DYNAMIC_PATCH,
            "repeat_time": REPEAT_TIME,
            "length": n_valid,
        },
        "stamp_test": {
            "root": str(DATA_ROOT),
            "annotation": str(test_jsonl),
            "data_augment": False,
            "max_dynamic_patch": MAX_DYNAMIC_PATCH,
            "repeat_time": REPEAT_TIME,
            "length": n_test,
        },
    }

    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path

# ============================================================
# Dataset
# ============================================================

class InternVLDataset(Dataset):
    def __init__(self, jsonl_path: Path, data_root: Path, image_token: str):
        self.rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.data_root = data_root
        self.image_token = image_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = self.data_root / row["image"]

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        prompt = row["conversations"][0]["value"]
        answer = row["conversations"][1]["value"]

        prompt = prompt.replace("<image>", self.image_token)

        if self.image_token not in prompt:
            prompt = f"{self.image_token}\n{prompt}"

        while prompt.count(self.image_token) > 1:
            first = prompt.find(self.image_token)
            second = prompt.find(self.image_token, first + len(self.image_token))
            prompt = prompt[:second] + prompt[second + len(self.image_token):]

        return {"image": image, "prompt": prompt, "answer": answer}


def find_image_token(processor) -> str:
    image_token = None

    for attr in ["image_token", "image_placeholder", "image_tag"]:
        if hasattr(processor, attr):
            image_token = getattr(processor, attr)
            break

    if image_token is None and hasattr(processor, "tokenizer"):
        for attr in ["image_token", "img_token", "image_placeholder"]:
            if hasattr(processor.tokenizer, attr):
                image_token = getattr(processor.tokenizer, attr)
                break

    return image_token if image_token is not None else "<image>"


def collate_fn(batch: List[Dict[str, Any]], processor) -> Dict[str, torch.Tensor]:
    images = [b["image"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    answers = [b["answer"] for b in batch]

    enc_prompt = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    full_text = [p + a for p, a in zip(prompts, answers)]
    enc_full = processor(
        text=full_text,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    labels = input_ids.clone()

    prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if "pixel_values" in enc_full:
        out["pixel_values"] = enc_full["pixel_values"]

    return out

# ============================================================
# Main
# ============================================================

def main() -> None:
    print("========== CONFIG ==========")
    print("MODEL_ID:", MODEL_ID)
    print("DATA_ROOT:", DATA_ROOT)
    print("OUT_DIR:", OUT_DIR)
    print("PRETRAINED_ADAPTER_ZIP:", PRETRAINED_ADAPTER_ZIP)
    print("PRETRAINED_ADAPTER_UNZIP_DIR:", PRETRAINED_ADAPTER_UNZIP_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("============================")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {DATA_ROOT}\n"
            "Place the exported YOLO dataset there, or update DATA_ROOT."
        )

    print("\n[1/7] Unzipping pretrained adapter...")
    adapter_dir = unzip_adapter(PRETRAINED_ADAPTER_ZIP, PRETRAINED_ADAPTER_UNZIP_DIR)
    print("Adapter extracted and found at:", adapter_dir)

    print("\n[2/7] Converting YOLO dataset to InternVL JSONL...")
    train_jsonl, n_train = convert_split("train")
    valid_jsonl, n_valid = convert_split("valid")
    test_jsonl, n_test = convert_split("test")

    print("JSONL written:")
    print(train_jsonl, n_train)
    print(valid_jsonl, n_valid)
    print(test_jsonl, n_test)

    meta_path = write_meta(train_jsonl, valid_jsonl, test_jsonl, n_train, n_valid, n_test)
    print("meta.json saved to:", meta_path)

    print("\n[3/7] Loading processor and base model...")
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = cap_major >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        use_bf16 = False
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    print("\n[4/7] Loading existing LoRA adapter for continued fine-tuning...")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        is_trainable=True,
    )

    model.config.use_cache = False
    model.train()

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    print("Loaded base:", MODEL_ID)
    print("Loaded adapter:", adapter_dir)
    print("dtype:", dtype)

    print("\n[5/7] Building datasets...")
    image_token = find_image_token(processor)
    print("Using IMAGE_TOKEN =", repr(image_token))

    train_ds = InternVLDataset(train_jsonl, DATA_ROOT, image_token)
    valid_ds = InternVLDataset(valid_jsonl, DATA_ROOT, image_token) if valid_jsonl.exists() else None
    test_ds = InternVLDataset(test_jsonl, DATA_ROOT, image_token) if test_jsonl.exists() else None

    if USE_SUBSET:
        train_len = max(1, len(train_ds) // TRAIN_SUBSET_DIVISOR)
        train_ds = torch.utils.data.Subset(train_ds, range(train_len))

        if valid_ds is not None:
            valid_len = max(1, len(valid_ds) // VALID_SUBSET_DIVISOR)
            valid_ds = torch.utils.data.Subset(valid_ds, range(valid_len))

    print("Train samples:", len(train_ds))
    print("Valid samples:", len(valid_ds) if valid_ds is not None else 0)
    print("Test samples:", len(test_ds) if test_ds is not None else 0)

    print("\n[6/7] Setting up trainer...")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = cap_major >= 8
        use_fp16 = not use_bf16
    else:
        use_bf16 = False
        use_fp16 = False

    steps_per_epoch = math.ceil(len(train_ds) / (BATCH * ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(10, int(0.03 * total_steps))

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR.parent),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        num_train_epochs=EPOCHS,
        logging_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        bf16_full_eval=use_bf16,
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds if valid_ds is not None else None,
        data_collator=lambda batch: collate_fn(batch, processor),
    )

    print("\n[7/7] Training...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("\nTraining complete.")
    print("Saved fine-tuned adapter to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()