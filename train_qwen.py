import os
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
import json
import torch
from datasets import Dataset 
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel # <-- Added PeftModel
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info
from wrapper.Qwen_25_LLM import Qwen_llm


# --- 1. CONFIGURATION ---
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 
OUTPUT_DIR = "./qwen2.5-vl-signature-detector"
TRAIN_DATA = "data/train.jsonl"
VALID_DATA = "data/valid.jsonl"

RESUME_ADAPTER_PATH =None # "./qwen2.5-vl-signature-detector" # Leave empty string "" if starting from scratch

# --- 2. LOAD MODEL VIA WRAPPER ---
# Initialize the wrapper
qwen_wrapper = Qwen_llm(base_model_path=LOCAL_MODEL_PATH, adapter_path=RESUME_ADAPTER_PATH)

# Load it in training mode! 
# This automatically handles model.train() and PeftModel's is_trainable flag.
qwen_wrapper.load(is_trainable=True)

# Extract the processor for the collator
processor = qwen_wrapper.processor

#limite pixel size
processor.image_processor.max_pixels = 602112 #776*776
processor.image_processor.min_pixels = 313600 #560*560

# --- 3. LORA CONFIGURATION ---
if not (RESUME_ADAPTER_PATH and os.path.exists(RESUME_ADAPTER_PATH)):
    print("⚙️ Configuring NEW LoRA adapter from scratch...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            
            "visual.merger.mlp.0", "visual.merger.mlp.2"
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none"
    )
    
    # YOUR FIX: Reassign the wrapped PEFT model directly back to the wrapper's property
    qwen_wrapper.model = get_peft_model(qwen_wrapper.model, lora_config)
    qwen_wrapper.model.print_trainable_parameters()
else:
    # It was already loaded as a trainable PeftModel by your wrapper!
    qwen_wrapper.model.print_trainable_parameters()
    
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
    
    # 1. Clean the incoming JSON dictionaries to remove None values
    for example in examples:
        clean_messages = []
        for msg in example["messages"]:
            clean_content = []
            for item in msg["content"]:
                clean_item = {k: v for k, v in item.items() if v is not None}
                clean_content.append(clean_item)
            clean_messages.append({"role": msg["role"], "content": clean_content})
        cleaned_messages_list.append(clean_messages)
        
    # 2. Extract image and video paths/arrays using Qwen's utility
    image_inputs, video_inputs = process_vision_info(cleaned_messages_list)
    
    # 3. Apply the Chat Template (injects <|im_start|> user ... <|im_start|> assistant ...)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
        for msg in cleaned_messages_list
    ]
    
    # 4. Process everything into padded PyTorch tensors
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # 5. Initialize labels as a clone of input_ids
    labels = batch["input_ids"].clone()
    
    # 6. Mask padding tokens so they don't contribute to the loss gradient
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # 7. 🚀 FIX: Mask ALL vision-related tokens so the model doesn't try to predict the image sequence
    vision_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for token in vision_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            labels[labels == token_id] = -100
            
    # 8. FAST PROMPT MASKING (Tensor sliding window)
    # We locate the exact start of the assistant's response to mask the user's prompt
    assistant_prefix = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    prefix_len = len(assistant_prefix)
    prefix_tensor = torch.tensor(assistant_prefix, device=labels.device)

    for i in range(labels.shape[0]):
        seq = labels[i]
        
        # Unfold creates a sliding window view of the tensor
        windows = seq.unfold(0, prefix_len, 1)
        
        # Check where the sliding window exactly matches the assistant_prefix sequence
        matches = (windows == prefix_tensor).all(dim=1)
        match_indices = matches.nonzero(as_tuple=True)[0]
        
        if len(match_indices) > 0:
            # The actual target tokens start AFTER the prefix
            match_idx = match_indices[0].item() + prefix_len
            
            # Mask everything from index 0 up to the start of the assistant's response
            labels[i, :match_idx] = -100 
            
    batch["labels"] = labels
    return batch

# --- 5. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, 
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True, 
    logging_steps=10, 
    dataloader_num_workers=4, 
    dataloader_prefetch_factor=2, 
    gradient_checkpointing=True, # 🚀 ADD THIS: Drastically reduces VRAM
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # --- BEST MODEL SELECTION LOGIC ---
    eval_strategy="epoch",          # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save a checkpoint at the end of each epoch (Must match eval_strategy)
    save_total_limit=3,             # Keep the last 3 checkpoints to save disk space
    load_best_model_at_end=True,    # 🚀 Load the best model at the end of training
    metric_for_best_model="eval_loss", # 🚀 Use validation loss to judge "best"
    greater_is_better=False,        # 🚀 Lower validation loss is better
    
    remove_unused_columns=False, 
    report_to="none" 
)
# --- 6. SFT TRAINER ---
print("🚀 Initializing SFTTrainer with Early Stopping...")
trainer = SFTTrainer(
    model=qwen_wrapper.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=qwen_collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # 🚀 ADD THIS LINE
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
