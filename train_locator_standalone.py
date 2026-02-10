import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou
from torch.cuda.amp import autocast, GradScaler

# --- CONFIG ---
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 768

# Paths
TRAIN_JSON = "data/signatures_augmented/rod_train.json"
VAL_JSON = "data/signatures_augmented/rod_valid.json"
IMG_DIR = "data/signatures_augmented/images"
SAVE_PATH = "output_model/clean_locator_owlv2"

# --- 1. ROBUST PRE-PROCESSING ---
def resize_and_pad(image, target_size):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_image.paste(image, (0, 0))
    return new_image, scale

# --- 2. DATASET ---
class CleanSignatureDataset(Dataset):
    def __init__(self, json_path, img_dir, processor, is_train=True):
        with open(json_path) as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.processor = processor
        self.is_train = is_train

    def __len__(self):
        return len(self.data) * 2 if self.is_train else len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            real_idx = idx // 2
            is_negative = (idx % 2 == 1)
        else:
            real_idx = idx
            is_negative = False
            
        item = self.data[real_idx]
        img_path = os.path.join(self.img_dir, item['file_name'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            image_processed, scale = resize_and_pad(image, TARGET_SIZE)
        except:
            return self.__getitem__((idx + 1) % self.__len__())

        if is_negative:
            query = "handwriting"
            boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            query = "signature"
            raw_boxes = item['boxes']
            fixed_boxes = []
            for box in raw_boxes:
                x, y, w, h = box
                x1 = (x * scale) / TARGET_SIZE
                y1 = (y * scale) / TARGET_SIZE
                x2 = ((x + w) * scale) / TARGET_SIZE
                y2 = ((y + h) * scale) / TARGET_SIZE
                fixed_boxes.append([max(0, min(1, x1)), max(0, min(1, y1)), max(0, min(1, x2)), max(0, min(1, y2))])
            boxes = torch.tensor(fixed_boxes, dtype=torch.float32)
        
        inputs = self.processor(images=image_processed, query_images=None, text=[[query]], return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "input_ids": inputs.input_ids.squeeze(),
            "boxes": boxes
        }

def collate_fn(batch):
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    input_ids = torch.stack([x['input_ids'] for x in batch])
    labels = [{"boxes": x['boxes']} for x in batch]
    return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

# --- 3. LOSS ---
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        indices = []
        for i in range(bs):
            tgt_bbox = targets[i]["boxes"]
            if len(tgt_bbox) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            p_prob = out_prob[i * num_queries : (i + 1) * num_queries]
            p_bbox = out_bbox[i * num_queries : (i + 1) * num_queries]
            cost_class = -p_prob[:, 0]
            cost_bbox = torch.cdist(p_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_convert(p_bbox, 'cxcywh', 'xyxy'), box_convert(tgt_bbox, 'xyxy', 'xyxy'))
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class.unsqueeze(-1)
            indices.append(linear_sum_assignment(C.cpu()))
        return [(torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = max(1.0, sum(len(t["boxes"]) for t in targets))
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=outputs["logits"].device)
        
        losses = {}
        src_logits = outputs["logits"]
        target_classes = torch.zeros_like(src_logits)
        for batch_idx, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0: target_classes[batch_idx, src_idx, 0] = 1.0
        losses['loss_ce'] = F.binary_cross_entropy_with_logits(src_logits, target_classes)

        src_boxes, target_boxes = [], []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(outputs["pred_boxes"][batch_idx][src_idx])
                target_boxes.append(targets[batch_idx]["boxes"][tgt_idx])
        
        if len(src_boxes) > 0:
            src_boxes = torch.cat(src_boxes)
            target_boxes = torch.cat(target_boxes)
            loss_bbox = F.l1_loss(src_boxes, box_convert(target_boxes, 'xyxy', 'cxcywh'), reduction='sum') / num_boxes
            loss_giou = (1 - torch.diag(generalized_box_iou(box_convert(src_boxes, 'cxcywh', 'xyxy'), target_boxes))).sum() / num_boxes
            losses['loss_bbox'] = loss_bbox
            losses['loss_giou'] = loss_giou
        else:
            losses['loss_bbox'] = torch.tensor(0.0, device=DEVICE)
            losses['loss_giou'] = torch.tensor(0.0, device=DEVICE)
        return losses

# --- 4. CHECKPOINTING ---
def save_checkpoint(model, processor, optimizer, scheduler, epoch, val_loss, path, is_best=False):
    os.makedirs(path, exist_ok=True)
    
    # Save Hugging Face Model
    if is_best:
        save_dir = os.path.join(path, "best_model")
        print(f"   [★] Saving New Best Model (Val Loss: {val_loss:.4f})")
    else:
        save_dir = os.path.join(path, "latest_checkpoint")
        print(f"   [Checkpoint] Saving Epoch {epoch} state...")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # Save Training State (Optimizer, Epoch)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }, os.path.join(save_dir, "training_state.pt"))

def load_checkpoint(path, model, optimizer, scheduler):
    """
    Robustly loads a checkpoint.
    1. If 'training_state.pt' exists, loads Epoch + Optimizer (Full Resume).
    2. If only 'pytorch_model.bin' exists, loads Weights Only (Fresh Start).
    """
    # 1. Try to Load Weights (Model)
    try:
        # We use standard Hugging Face loading
        print(f">> Attempting to load model weights from {path}...")
        model.load_state_dict(OwlViTForObjectDetection.from_pretrained(path).state_dict())
        print("   [✓] Model weights loaded successfully.")
    except Exception as e:
        print(f"   [!] Failed to load model weights: {e}")
        return 0, float('inf')

    # 2. Try to Load Optimizer State (Resume Training)
    state_path = os.path.join(path, "training_state.pt")
    
    if os.path.exists(state_path):
        print(f">> Loading optimizer state from {state_path}...")
        checkpoint = torch.load(state_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        print(f"   [✓] Full Resume successful! Continuing from Epoch {epoch}")
        return epoch, val_loss
    else:
        # Fallback: Weights loaded, but optimizer is fresh
        print("   [i] No optimizer state found (likely from 'Clean' script).")
        print("   [>] Starting fresh training using loaded weights.")
        return 0, float('inf')
# --- 5. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to 'latest_checkpoint' folder to resume from")
    args = parser.parse_args()

    print(f">> Starting Training. Device: {DEVICE}")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(DEVICE)
    
    train_ds = CleanSignatureDataset(TRAIN_JSON, IMG_DIR, processor, is_train=True)
    val_ds = CleanSignatureDataset(VAL_JSON, IMG_DIR, processor, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    criterion = SetCriterion(HungarianMatcher(), {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 3.0}, ['labels', 'boxes']).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * EPOCHS)
    
    start_epoch = 0
    best_val_loss = float('inf')

    # RESUME LOGIC
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"   Resuming from Epoch {start_epoch} (Best Loss: {best_val_loss:.4f})")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Train
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc="Training")
        for batch in loop:
            optimizer.zero_grad()
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
            
            with autocast():
                outputs = model(pixel_values=pixel_values, input_ids=input_ids)
                loss_dict = criterion({"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, targets)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += losses.item()
            loop.set_postfix(loss=losses.item())
        
        print(f"   Train Loss: {total_loss / len(train_loader):.4f}")

        # Validate
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
                outputs = model(pixel_values=pixel_values, input_ids=input_ids)
                loss_dict = criterion({"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, targets)
                val_loss_sum += sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys()).item()
        
        val_loss = val_loss_sum / len(val_loader)
        print(f"   Valid Loss: {val_loss:.4f}")

        # Save Latest Checkpoint (Overwrite every epoch for Resume)
        save_checkpoint(model, processor, optimizer, scheduler, epoch, val_loss, SAVE_PATH, is_best=False)

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, processor, optimizer, scheduler, epoch, val_loss, SAVE_PATH, is_best=True)

if __name__ == "__main__":
    main()
