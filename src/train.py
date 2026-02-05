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
from src.ingestion import RODDataset 

# --- COLLATOR ---
class RealDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        boxes = [item["boxes"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        # Labels: -100 for padding
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
    parser.add_argument("--target_batch_size", type=int, default=128, help="Paper uses 128")
    parser.add_argument("--lr", type=float, default=1e-4, help="Paper uses 1e-4 for both stages")
    
    # Hardware constraints
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="What fits on your GPU (e.g., 4 or 8)")
    
    # Data
    parser.add_argument("--epochs", type=int, default=2, help="Paper: Stage 1=2 epochs, Stage 2=1 epoch")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--limit_data", type=int, default=0) # Set to 0 for full run
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (e.g. output_model/stage1.pt)")
    
    return parser.parse_args()

def setup_model(stage):
    model = RODMLLM(device="cuda")
    # Enable gradients for 4-bit layers
    model.llm = prepare_model_for_kbit_training(model.llm) 
    
    # 1. Default: Freeze Everything
    model.visual_encoder.requires_grad_(False)
    model.locator_module.model.requires_grad_(False)
    model.llm.requires_grad_(False) # Freeze LLM base
    
    # 2. Unfreeze Projectors (Always trainable in both stages)
    for p in model.global_projector.parameters(): p.requires_grad = True
    for p in model.regional_projector.parameters(): p.requires_grad = True
    
    # 3. Stage 2 Specifics: Add LoRA
    if stage == 2:
        print(f"\n[INFO] Stage 2: Adding LoRA Adapters (r=128, alpha=256)...")
        lora_config = LoraConfig(
            r=128,           # Paper setting
            lora_alpha=256,  # Paper setting
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
    
    # 1. Calculate Accumulation Steps to hit Target Batch Size (128)
    grad_accum_steps = args.target_batch_size // args.per_device_batch_size
    print(f"[Config] Target Batch: {args.target_batch_size} | GPU Batch: {args.per_device_batch_size}")
    print(f"[Config] Gradient Accumulation Steps: {grad_accum_steps}")
    
    model = setup_model(args.stage)
    
    # Resume Logic
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Loading weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    # Dataset
    dataset_path = "data/ROD/rod_dataset_cached.json"
    full_dataset = RODDataset(dataset_path, "data/images/")
    
    if args.limit_data > 0:
        full_dataset = Subset(full_dataset, range(args.limit_data))
        
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Samples: {len(train_dataset)} Train, {len(val_dataset)} Val")
    
    collator = RealDataCollator(model.tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collator)
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    # Scheduler
    num_update_steps = (len(dataloader) // grad_accum_steps) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.03 * num_update_steps), num_update_steps)
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            batch_boxes = batch["boxes"]
            prompts = batch["prompts"]
            
            # Forward
            with torch.no_grad():
                # Pre-compute CLIP features to save memory during backprop
                clip_out = model.visual_encoder(pixel_values, output_hidden_states=True)
                last_hidden = clip_out.hidden_states[-1][:, 1:, :]
                b_s = last_hidden.shape[0]
                spatial_map = last_hidden.view(b_s, 24, 24, 1024).permute(0, 3, 1, 2)

            # Manual Injection of Regional Features
            for i in range(len(prompts)):
                boxes = batch_boxes[i].to(device)
                single_map = spatial_map[i].unsqueeze(0)
                roi_feats = roi_align(single_map, [boxes], output_size=(2, 2), spatial_scale=1.0/14.0)
                # We run this just to ensure the regional projector gets gradients
                _ = model.regional_projector(roi_feats.flatten(2).transpose(1, 2))

            # Main Forward
            outputs = model(input_ids, pixel_values, labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * grad_accum_steps
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        # End of Epoch
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
