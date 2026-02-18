import os
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
import json
import torch
from datasets import Dataset 
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel # <-- Added PeftModel
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info

# --- 1. CONFIGURATION ---
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 
OUTPUT_DIR = "qwen2.5-vl-signature-detector"
TRAIN_DATA = "data/train.jsonl"
VALID_DATA = "data/valid.jsonl"

# 🔄 NEW: Set this to the path of your saved adapter to continue training it. 
# Set to None (or an empty string) if you want to train from scratch.
RESUME_ADAPTER_PATH = "./qwen2.5-vl-signature-detector" # e.g., your previously saved output dir

# --- 2. LOAD PROCESSOR & BASE MODEL ---
print(f"⏳ Loading Processor from {LOCAL_MODEL_PATH}...")
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    max_pixels=448 * 28 * 28, 
    # Removed local_files_only=True so it can successfully pull from the HF cache/hub
)

print(f"⏳ Loading Base Model (Qwen2.5-VL) from {LOCAL_MODEL_PATH}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2", 
    # Removed local_files_only=True 
)

# --- 3. LORA CONFIGURATION (NEW LOGIC) ---
# If an adapter path is provided AND the folder exists, load it for further training
if RESUME_ADAPTER_PATH and os.path.exists(RESUME_ADAPTER_PATH):
    print(f"🔄 Loading EXISTING LoRA adapter from {RESUME_ADAPTER_PATH} for continued training...")
    # is_trainable=True is CRITICAL. Without it, the weights will be frozen for inference.
    model = PeftModel.from_pretrained(model, RESUME_ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()
else:
    print("⚙️ Configuring NEW LoRA adapter from scratch...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# --- 4. DATASET & COLLATOR ---
print("📂 Loading datasets...")

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_list = load_jsonl(TRAIN_DATA)
valid_list = load_jsonl(VALID_DATA)

train_dataset = Dataset.from_list(train_list)
valid_dataset = Dataset.from_list(valid_list)

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
    
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
        
    batch["labels"] = labels
    return batch

# --- 5. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, 
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True, 
    logging_steps=10, 
    dataloader_num_workers=4, 
    dataloader_prefetch_factor=2, 
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False, 
    report_to="none" 
)

# --- 6. SFT TRAINER ---
print("🚀 Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=qwen_collate_fn,
)

# --- 7. TRAIN & SAVE ---
print("🔥 Starting Training...")

# 🚀 PRO TIP: If you want to resume exactly from a crashed epoch (keeping optimizer states),
# change trainer.train() to trainer.train(resume_from_checkpoint=True)
# assuming your OUTPUT_DIR contains standard Hugging Face checkpoint folders.
trainer.train()

print(f"💾 Saving LoRA adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete!")
