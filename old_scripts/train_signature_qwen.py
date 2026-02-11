import torch
import argparse
import os
import json
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from peft import PeftModel

# Custom Imports
from src.qwen_model import RODMLLM
from qwen_vl_utils import process_vision_info

# --- 1. GROUNDING DATASET (With Masking Logic) ---
class GroundingDataset(Dataset):
    def __init__(self, jsonl_file, base_image_dir, processor):
        self.base_image_dir = base_image_dir
        self.processor = processor
        self.data = []
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"File not found: {jsonl_file}")
            
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"[Dataset] Loaded {len(self.data)} grounding samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.base_image_dir, item["image"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))

        # 1. Prepare Text
        user_text = item["conversations"][0]["value"].replace("<img></img>", "").strip()
        # The Assistant response contains the Critical Grounding Tokens: <box>...</box>
        assistant_text = item["conversations"][1]["value"]

        # 2. Build Conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        ]

        # 3. Process Vision Inputs
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 4. TOKENIZATION WITH MASKING (The Secret Sauce)
        # First, format the prompt (User part)
        prompt_text = self.processor.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        # Then, format the completion (Assistant part)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize Prompt (User) - We want to IGNORE this in loss
        prompt_ids = self.processor.tokenizer(
            prompt_text, return_tensors='pt', add_special_tokens=False
        ).input_ids[0]

        # Tokenize Full (User + Assistant)
        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        labels = input_ids.clone()

        # --- MASKING LOGIC ---
        # Set the labels corresponding to the User Prompt to -100 (Ignore Index)
        # This forces the model to ONLY learn from the coordinates (Assistant part)
        prompt_len = len(prompt_ids)
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
            
        # Also mask padding
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0), 
            "image_grid_thw": inputs.image_grid_thw, 
            "labels": labels
        }

# --- 2. COLLATOR ---
def qwen_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0),
        "image_grid_thw": torch.cat([x["image_grid_thw"] for x in batch], dim=0) if batch[0]["image_grid_thw"] is not None else None,
        "labels": torch.stack([x["labels"] for x in batch])
    }

# --- 3. CONFIG ---
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for precise grounding
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1) # 1 Epoch is enough with masking
    parser.add_argument("--limit_data", type=int, default=2000) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_data_path", type=str, default="data/qwen_signatures_ready")
    parser.add_argument("--output_dir", type=str, default="output_model/qwen_grounding_specialist")
    parser.add_argument("--resume_from", type=str, default="", help="Path to checkpoint")
    return parser.parse_args()

# --- 4. TRAIN LOOP ---
def train(args):
    print(f">> Starting GROUNDING SPECIALIST Training...")
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_dir = os.path.join(args.output_dir, "best_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model_wrapper = RODMLLM(device=device, train_mode=True)
    model = model_wrapper.model
    processor = model_wrapper.processor
    
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)

    # Enable Gradients for LoRA
    model.enable_input_require_grads()  
    model.gradient_checkpointing_enable()
    
    # 2. Dataset
    train_jsonl = os.path.join(args.base_data_path, "train.jsonl")
    val_jsonl = os.path.join(args.base_data_path, "valid.jsonl")
    
    full_train_dataset = GroundingDataset(train_jsonl, args.base_data_path, processor)
    val_dataset = GroundingDataset(val_jsonl, args.base_data_path, processor)
    
    if args.limit_data > 0:
        indices = torch.randperm(len(full_train_dataset))[:args.limit_data]
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=qwen_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qwen_collate_fn, num_workers=args.num_workers)
    
    # 3. Optimizer
    try:
        from bitsandbytes.optim import PagedAdamW8bit
        optimizer = PagedAdamW8bit(model.parameters(), lr=args.lr)
    except:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    
    num_update_steps = len(train_loader) * args.epochs // args.grad_accum
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * num_update_steps), num_update_steps)
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)
            labels = batch["labels"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device) if batch["image_grid_thw"] is not None else None
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                loss = outputs.loss / args.grad_accum
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum
            progress.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)
                labels = batch["labels"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device) if batch["image_grid_thw"] is not None else None
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels
                    )
                val_loss += outputs.loss.item()
        
        avg_val = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)
            print(f"Saved Best Model to {best_model_dir}")

if __name__ == "__main__":
    args = get_config()
    train(args)
