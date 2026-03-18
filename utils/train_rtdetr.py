import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from torch.optim import AdamW
from tqdm import tqdm

# CONFIG

DATASET_ROOT = "dataset_split/logo"  
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "valid")

MODEL_NAME = "PekingU/rtdetr_r50vd"
OUTPUT_DIR = "checkpoints_logo"

BATCH_SIZE = 4
LR = 3e-5
EPOCHS = 10
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# PROCESSOR

processor = RTDetrImageProcessor.from_pretrained(
    MODEL_NAME,
    size={"height": 512, "width": 512}
)


# DATASETS


train_dataset = CocoDetection(
    root=TRAIN_DIR,
    annFile=os.path.join(TRAIN_DIR, "_annotations.coco.json")
)

val_dataset = CocoDetection(
    root=VAL_DIR,
    annFile=os.path.join(VAL_DIR, "_annotations.coco.json")
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

num_classes = 1


# COLLATE FUNCTION

def collate_fn(batch):

    images = []
    annotations = []

    for i, (image, target) in enumerate(batch):

        images.append(image)

        annotations.append({
            "image_id": i,
            "annotations": target
        })

    encoding = processor(
        images=images,
        annotations=annotations,
        return_tensors="pt"
    )

    return encoding


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# MODEL

model = RTDetrForObjectDetection.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)


optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# TRAINING LOOP

best_val_loss = float("inf")
no_improve = 0

for epoch in range(EPOCHS):

    print(f"\n EPOCH {epoch+1}/{EPOCHS} ")

    # TRAIN 
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc="Training"):

        batch["pixel_values"] = batch["pixel_values"].to(device)

        batch["labels"] = [
            {k: v.to(device) for k, v in t.items()}
            for t in batch["labels"]
        ]

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)
    print("Train Loss:", train_loss)

    # VALIDATION 
    model.eval()
    total_val_loss = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Validation"):

            batch["pixel_values"] = batch["pixel_values"].to(device)

            batch["labels"] = [
                {k: v.to(device) for k, v in t.items()}
                for t in batch["labels"]
            ]

            outputs = model(**batch)
            total_val_loss += outputs.loss.item()

    val_loss = total_val_loss / len(val_loader)
    print("Val Loss:", val_loss)

    # SAVE EPOCH 
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    print("Model saved:", epoch_dir)

    # BEST MODEL 
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        no_improve = 0

        best_dir = os.path.join(OUTPUT_DIR, "best_model")
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)

        print("Best model updated")

    else:
        no_improve += 1
        print("No improvement:", no_improve)

    # EARLY STOPPING 
    if no_improve >= PATIENCE:
        print(" Early stopping")
        break

print("\nTraining finished.")