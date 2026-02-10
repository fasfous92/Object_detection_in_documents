import torch
import argparse
import os
import math
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler # <--- 1. IMPORT AMP

from src.model import RODMLLM
from utils.ingestion import RealDataCollator 

# --- DATASET ---
class SignatureDataset(Dataset):
    def __init__(self, annotation_file, image_folder):
        self.image_folder = image_folder
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item["file_name"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))

        return {
            "image": image, 
            "boxes": torch.tensor(item["boxes"]), 
            "prompt": item["prompt"],
            "file_name": item["file_name"]
        }

def get_config():
    parser = argparse.ArgumentParser()
    
    # --- EXISTING ARGS ---
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--target_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    
    # --- ADD THIS LINE ---
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers for data loading")

    # --- REST OF EXISTING ARGS ---
    parser.add_argument("--train_annotation", type=str, default="data/signatures_augmented/rod_train.json")
    parser.add_argument("--val_annotation", type=str, default="data/signatures_augmented/rod_valid.json")
    parser.add_argument("--image_dir", type=str, default="data/signatures_augmented/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit_data", type=int, default=0) 
    parser.add_argument("--resume", type=str, default="output_model/stage1_epoch5.pt")
    
    return parser.parse_args()

def setup_model(stage):
    model = RODMLLM(device="cuda")
    # This prepares 4-bit/8-bit models for training (casts non-int8 layers to fp32)
    model.llm = prepare_model_for_kbit_training(model.llm) 
    
    # 1. Freeze Base
    model.visual_encoder.requires_grad_(False)
    model.llm.requires_grad_(False)

    # 2. Locator Strategy (Stage 2)
    if stage == 2:
        print("[INFO] Unfreezing Locator (OWLv2) for fine-tuning...")
        # Unfreeze the specific HF model inside your wrapper
        model.locator_module.model.requires_grad_(True) 
    else:
        model.locator_module.model.requires_grad_(False)
    
    # 3. Unfreeze Projectors
    for p in model.global_projector.parameters(): p.requires_grad = True
    for p in model.regional_projector.parameters(): p.requires_grad = True
    
    # 4. Add LoRA
    if stage == 2:
        print(f"\n[INFO] Adding LoRA Adapters...")
        lora_config = LoraConfig(
            r=128,            
            lora_alpha=256, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
    
    return model

def train(args):
    print(f">> Starting Accelerated Fine-Tuning (Stage {args.stage})")
    os.makedirs("output_model", exist_ok=True)
    device = "cuda"
    
    # Accumulation calculation
    grad_accum_steps = args.target_batch_size // args.per_device_batch_size
    print(f"[Config] Batch: {args.per_device_batch_size} | Accum: {grad_accum_steps} | Workers: {args.num_workers}")
    
    model = setup_model(args.stage)
    
    # Resume / Load Base Weights
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Loading weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("   Weights loaded successfully.")
    else:
        print(f"[WARNING] No checkpoint found at {args.resume}. Starting fresh.")

    # Datasets
    train_dataset = SignatureDataset(args.train_annotation, args.image_dir)
    val_dataset = SignatureDataset(args.val_annotation, args.image_dir)

    if args.limit_data > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.limit_data))
    
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    collator = RealDataCollator(model.tokenizer)
    
    # <--- 3. OPTIMIZED DATALOADERS
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.per_device_batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    # <--- 4. INIT GRAD SCALER
    scaler = GradScaler() 
    
    num_update_steps = (len(dataloader) // grad_accum_steps) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * num_update_steps), num_update_steps)
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # <--- 5. MIXED PRECISION FORWARD
            with autocast():
                outputs = model(input_ids, pixel_values, labels)
                loss = outputs.loss / grad_accum_steps
            
            # <--- 6. SCALED BACKWARD
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * grad_accum_steps
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        # Validation Loop (Also accelerated)
        model.eval()
        val_loss = 0
        print("Running Validation...")
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                with autocast():
                    outputs = model(input_ids, pixel_values, labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss:   {avg_val_loss:.4f}")
        model.train()

        # Saving Logic (FIXED to include locator)
        save_path = f"output_model/signature.pt"
        print(f"Saving to {save_path}...")
        state_dict = model.state_dict()
        
        trainable_weights = {}
        for k, v in state_dict.items():
            # Save Projectors, LoRA, AND Locator
            if "projector" in k or "lora" in k or "locator_module" in k:
                trainable_weights[k] = v
                
        torch.save(trainable_weights, save_path)
        print(f"   Saved {len(trainable_weights)} keys.")

if __name__ == "__main__":
    args = get_config()
    train(args)
