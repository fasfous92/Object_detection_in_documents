import torch
from torch.utils.data import DataLoader
from wrapper.local_llm import LocalLLM
from wrapper.dataset_handler import GroundingDataset
import os
os.environ["HF_HOME"] = "/tmp/huggingface_cache" # Ensure this is

# 1. Setup
llm = LocalLLM(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
llm.load(use_adapters=False) 

# 2. Load ONE item
print("🕵️  Checking Input Sequence...")
dataset = GroundingDataset(jsonl_file="./data/debug_train.jsonl", base_image_dir="./data")
collator = llm._get_collator()
loader = DataLoader(dataset, batch_size=1, collate_fn=collator)
batch = next(iter(loader))

# 3. Decode the INPUT (What the model sees)
input_ids = batch["input_ids"][0]
decoded_text = llm.processor.tokenizer.decode(input_ids, skip_special_tokens=False)

print("\n📝 FULL INPUT SEQUENCE SEEN BY MODEL:")
print("=" * 60)
print(decoded_text)
print("=" * 60)

# 4. The Blindness Check
if "<|image_pad|>" in decoded_text or "<|vision_start|>" in decoded_text:
    print("✅  Image tokens found. The model SHOULD see the image.")
else:
    print("❌  CRITICAL FAILURE: No image tokens found!")
    print("    The model is receiving the text prompt but NO placeholder for the image.")
    print("    It is effectively trying to detect objects in a pure text chat.")
