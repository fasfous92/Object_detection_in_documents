import os
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
import json
import re
import torch
from datasets import Dataset 
from peft import LoraConfig, get_peft_model
from trl.experimental.papo import PAPOTrainer, PAPOConfig 
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- 1. CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" 
OUTPUT_DIR = "./qwen2-vl-papo-official"
TRAIN_DATA = "data_signature_only/train.jsonl"

# --- 2. LOAD MODEL & PROCESSOR ---
print(f"⚙️ Loading {MODEL_ID}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- 3. LORA CONFIGURATION (Targeting Visual + Language) ---
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "visual.merger.mlp.0", "visual.merger.mlp.2" # Visual grounding layers
    ],
    task_type="CAUSAL_LM", lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# --- 4. DATASET PREPARATION ---
def load_and_transform_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]
    formatted_data = []
    for item in raw_data:
        try:
            user_msg = next(msg for msg in item["messages"] if msg["role"] == "user")
            assistant_msg = next(msg for msg in item["messages"] if msg["role"] == "assistant")
            gt_json = json.loads(assistant_msg["content"][0]["text"])
            gt_objects = [{"box": obj["bbox"], "label": obj["label"].lower()} for obj in gt_json]
            
            # Store the raw prompt list for the processor
            formatted_data.append({
                "prompt": user_msg["content"], 
                "ground_truth_objects": gt_objects
            })
        except: continue
    return Dataset.from_list(formatted_data)

train_dataset = load_and_transform_dataset(TRAIN_DATA)

# --- 5. REWARD FUNCTIONS ---
def extract_boxes(text):
    if not isinstance(text, str): return []
    text = re.sub(r"```json|```", "", text).strip()
    try:
        data = json.loads(text)
        return [{"box": obj["bbox"], "label": obj["label"].lower()} for obj in data if "bbox" in obj]
    except:
        pattern = r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*,\s*"label"\s*:\s*"([^"]+)"'
        return [{"box": [int(m[0]), int(m[1]), int(m[2]), int(m[3])], "label": m[4].lower()} for m in re.findall(pattern, text)]

def format_reward_func(completions, **kwargs):
    return [1.0 if len(extract_boxes(c)) > 0 else -1.0 for c in completions]

def iou_reward_func(completions, ground_truth_objects, **kwargs):
    rewards = []
    for content, gt_list in zip(completions, ground_truth_objects):
        preds = extract_boxes(content)
        if not preds or not gt_list:
            rewards.append(0.0); continue
        
        # Bipartite matching logic
        iou_matrix = []
        for g_idx, gt in enumerate(gt_list):
            for p_idx, pred in enumerate(preds):
                if gt["label"] == pred["label"]:
                    y1, x1 = max(pred["box"][0], gt["box"][0]), max(pred["box"][1], gt["box"][1])
                    y2, x2 = min(pred["box"][2], gt["box"][2]), min(pred["box"][3], gt["box"][3])
                    inter = max(0, y2-y1) * max(0, x2-x1)
                    union = (pred["box"][2]-pred["box"][0])*(pred["box"][3]-pred["box"][1]) + \
                            (gt["box"][2]-gt["box"][0])*(gt["box"][3]-gt["box"][1]) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > 0: iou_matrix.append((iou, g_idx, p_idx))
        
        iou_matrix.sort(key=lambda x: x[0], reverse=True)
        matched_gt, matched_pred, total_iou = set(), set(), 0.0
        for iou, g, p in iou_matrix:
            if g not in matched_gt and p not in matched_pred:
                matched_gt.add(g); matched_pred.add(p); total_iou += iou
        
        avg_iou = total_iou / len(gt_list)
        penalty = (len(preds) - len(matched_pred)) * 0.2
        rewards.append(max(-1.0, (avg_iou * 2.0) - penalty))
    return rewards

# --- 6. 🚀 CUSTOM TRAINER & COLLATOR (The Real Fix) ---

class QwenPAPOTrainer(PAPOTrainer):
    def _prepare_inputs(self, inputs):
        # Qwen-specific key mapping for PAPO's native masking logic
        if "pixel_values_vit" in inputs:
            inputs["pixel_values"] = inputs["pixel_values_vit"]
        return super()._prepare_inputs(inputs)

def collator(data):
    """Bypasses the tokenizer-only logic to support Qwen2-VL prompts."""
    prompts = [d["prompt"] for d in data]
    # Processor handles the image/text mixing that a simple tokenizer cannot
    inputs = processor(text=None, images=None, prompts=prompts, return_tensors="pt", padding=True)
    inputs["ground_truth_objects"] = [d["ground_truth_objects"] for d in data]
    return inputs

# --- 7. PAPO CONFIG ---
config = PAPOConfig(
    output_dir=OUTPUT_DIR,
    loss_type="grpo",
    perception_loss_weight=0.1,
    mask_ratio=0.3, # Official PAPO dynamic masking ratio
    mask_type="patch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=2, # VRAM safety for 20GB
    max_completion_length=128,
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=False,
    report_to="none"
)

# Initialize using our mapped subclass
trainer = QwenPAPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    reward_funcs=[format_reward_func, iou_reward_func],
    data_collator=collator, 
)

# --- 8. TRAIN ---
print("🔥 Launching REAL PAPO Training Loop...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
