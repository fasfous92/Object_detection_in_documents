import os
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
import json
import torch
import random # <-- Added for shuffling
from datasets import Dataset 
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info
from utils.Qwen_25_LLM import Qwen_llm

# --- 1. CONFIGURATION ---
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 
OUTPUT_DIR = "output/qwen2.5-vl-final-detector-all-linear"
TRAIN_DATA = "data_final/train.jsonl"
VALID_DATA = "data_final/valid.jsonl"

RESUME_ADAPTER_PATH ="output/qwen2.5-vl-final-detector-all-linear"

# 🚀 NEW CONFIG: How many times to duplicate images containing a logo
LOGO_OVERSAMPLE_FACTOR = 10 

# --- 2. LOAD MODEL VIA WRAPPER ---
qwen_wrapper = Qwen_llm(base_model_path=LOCAL_MODEL_PATH, adapter_path=RESUME_ADAPTER_PATH)
qwen_wrapper.load(is_trainable=True)

processor = qwen_wrapper.processor
processor.image_processor.max_pixels = 602112 
processor.image_processor.min_pixels = 313600 

# --- 3. LORA CONFIGURATION ---
if not (RESUME_ADAPTER_PATH and os.path.exists(RESUME_ADAPTER_PATH)):
    print("⚙️ Configuring NEW LoRA adapter from scratch...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules='all-linear',
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none"
    )
    
    qwen_wrapper.model = get_peft_model(qwen_wrapper.model, lora_config)
    qwen_wrapper.model.print_trainable_parameters()
else:
    qwen_wrapper.model.print_trainable_parameters()
    
# --- 4. DATASET LOGIC WITH OVERSAMPLING ---
print("📂 Loading datasets...")

def load_and_oversample(file_path, is_train=False, oversample_factor=1):
    """
    Loads JSONL data and conditionally duplicates images containing minority classes.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    # Never oversample validation/test data!
    if not is_train or oversample_factor <= 1:
        return lines

    balanced_lines = []
    logo_count = 0
    
    for item in lines:
        balanced_lines.append(item) # Always append the original image at least once
        
        # Look for the assistant's response to see what labels are in this image
        try:
            assistant_msg = next(msg for msg in item["messages"] if msg["role"] == "assistant")
            assistant_text = assistant_msg["content"][0]["text"]
            
            # Fast string check to see if Qwen is supposed to output a 'logo' here
            if '"label": "logo"' in assistant_text:
                # Add it (oversample_factor - 1) extra times
                for _ in range(oversample_factor - 1):
                    balanced_lines.append(item)
                logo_count += 1
        except Exception:
            pass # Failsafe in case JSON is malformed
            
    print(f"🔄 Oversampled {logo_count} distinct logo images by a factor of {oversample_factor}x.")
    print(f"📈 Original train size: {len(lines)} | New balanced train size: {len(balanced_lines)}")
    
    # Shuffle the dataset so the duplicated logos aren't fed to the model in immediate succession
    random.shuffle(balanced_lines)
    return balanced_lines

# Load datasets using the new logic
train_list = load_and_oversample(TRAIN_DATA, is_train=True, oversample_factor=LOGO_OVERSAMPLE_FACTOR)
valid_list = load_and_oversample(VALID_DATA, is_train=False) # is_train=False guarantees no oversampling

train_dataset = Dataset.from_list(train_list)
valid_dataset = Dataset.from_list(valid_list)

# --- 5. COLLATOR ---
def qwen_collate_fn(examples):
    cleaned_messages_list = []
    
    for example in examples:
        clean_messages = []
        for msg in example["messages"]:
            clean_content = []
            for item in msg["content"]:
                clean_item = {k: v for k, v in item.items() if v is not None}
                clean_content.append(clean_item)
            clean_messages.append({"role": msg["role"], "content": clean_content})
        cleaned_messages_list.append(clean_messages)
        
    image_inputs, video_inputs = process_vision_info(cleaned_messages_list)
    
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
        for msg in cleaned_messages_list
    ]
    
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    vision_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for token in vision_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            labels[labels == token_id] = -100
            
    assistant_prefix = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    prefix_len = len(assistant_prefix)
    prefix_tensor = torch.tensor(assistant_prefix, device=labels.device)

    for i in range(labels.shape[0]):
        seq = labels[i]
        windows = seq.unfold(0, prefix_len, 1)
        matches = (windows == prefix_tensor).all(dim=1)
        match_indices = matches.nonzero(as_tuple=True)[0]
        
        if len(match_indices) > 0:
            match_idx = match_indices[0].item() + prefix_len
            labels[i, :match_idx] = -100 
            
    batch["labels"] = labels
    return batch

# --- 6. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, 
    num_train_epochs=6,
    learning_rate=5e-5,
    bf16=True, 
    logging_steps=4, 
    dataloader_num_workers=4, 
    dataloader_prefetch_factor=2, 
    gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    eval_strategy="epoch",          
    save_strategy="epoch",          
    save_total_limit=3,             
    load_best_model_at_end=True,    
    metric_for_best_model="eval_loss", 
    greater_is_better=False,        
    
    remove_unused_columns=False, 
    report_to="none" 
)

# --- 7. SFT TRAINER ---
print("🚀 Initializing SFTTrainer with Early Stopping...")
trainer = SFTTrainer(
    model=qwen_wrapper.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=qwen_collate_fn,
    dataset_kwargs={"skip_prepare_dataset": True}, 
    max_seq_length=4096, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
)

# --- 8. TRAIN & SAVE ---
print("🔥 Starting Training...")
trainer.train()

print(f"💾 Saving LoRA adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete!")
