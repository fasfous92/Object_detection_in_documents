import torch
import argparse
import os
import math
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW 
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from torchvision.ops import roi_align

from src.model import RODMLLM
from utils.ingestion import RODDataset 

# --- COLLATOR ---
class RealDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        # Handle variable number of boxes per image
        boxes = [item["boxes"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        # Labels: -100 for padding so loss ignores them
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values, "input_ids": input_ids,
            "labels": labels, "prompts": prompts, "boxes": boxes 
        }

def get_config():
    parser = argparse.ArgumentParser()
    
    # --- PAPER SETTINGS ---
    parser.add_argument("--stage", type=int, default=1, help="1=Alignment, 2=Instruction Tuning")
    parser.add_argument("--target_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Hardware constraints
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    
    # Data Paths (ADAPTED FOR COCO)
    parser.add_argument("--dataset_json", type=str, default="data/ROD/rod_dataset_cached.json")
    parser.add_argument("--image_dir", type=str, default="data/images") 
    
    # Training Loop
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--limit_data", type=int, default=0) 
    parser.add_argument("--resume", type=str, default="")
    
    return parser.parse_args()

def setup_model(stage):
    model = RODMLLM(device="cuda")
    model.llm = prepare_model_for_kbit_training(model.llm) 
    
    # 1. Freeze Base Models
    model.visual_encoder.requires_grad_(False)
    model.locator_module.model.requires_grad_(False)
    model.llm.requires_grad_(False)
    
    # 2. Unfreeze Projectors (The main goal of Stage 1)
    for p in model.global_projector.parameters(): p.requires_grad = True
    for p in model.regional_projector.parameters(): p.requires_grad = True
    
    # 3. Stage 2: LoRA
    if stage == 2:
        print(f"\n[INFO] Stage 2: Adding LoRA Adapters...")
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
    print(f">> Starting Training STAGE {args.stage}")
    os.makedirs("output_model", exist_ok=True)
    device = "cuda"
    
    grad_accum_steps = args.target_batch_size // args.per_device_batch_size
    print(f"[Config] Target Batch: {args.target_batch_size} | GPU Batch: {args.per_device_batch_size}")
    print(f"[Config] Gradient Accumulation Steps: {grad_accum_steps}")
    
    model = setup_model(args.stage)
    
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Loading weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    # --- DATASET LOADING ---
    print(f"[Data] Loading JSON from {args.dataset_json}")
    if not os.path.exists(args.dataset_json):
        print("(!) JSON not found. Please run 'python src/prepare_stage1_data.py' first.")
        return

    # Note: image_dir usually needs to point to the folder containing the .jpg files
    # For COCO downloaded via the previous script, this is data/images/train2017
    full_dataset = RODDataset(args.dataset_json, args.image_dir)
    
    if args.limit_data > 0:
        print(f"[Data] Limiting to first {args.limit_data} samples.")
        full_dataset = Subset(full_dataset, range(args.limit_data))
        
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    collator = RealDataCollator(model.tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collator)
    
    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    num_update_steps = (len(dataloader) // grad_accum_steps) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.03 * num_update_steps), num_update_steps)
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            
            # Move to Device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # The model handles vision encoding, locator (if active), and alignment internally
            # We pass the boxes (from COCO GT) so the regional projector can learn
            # Note: The 'boxes' argument support depends on your model.forward() implementation.
            # If your model extracts boxes automatically using the Locator, you might not need to pass these.
            # However, for Stage 1 Alignment, passing GT boxes is standard.
            
            # Forward
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * grad_accum_steps
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Result: Avg Loss {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = f"output_model/stage{args.stage}_epoch{epoch+1}.pt"
        print(f"Saving to {save_path}...")
        state_dict = model.state_dict()
        trainable = {k: v for k, v in state_dict.items() if "projector" in k or "lora" in k}
        torch.save(trainable, save_path)

if __name__ == "__main__":
    args = get_config()
    train(args)
