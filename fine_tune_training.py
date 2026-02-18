import os
from wrapper.local_llm import LocalLLM
os.environ["HF_HOME"] = "/tmp/huggingface_cache"


def fine_tune_model():
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    model_id="Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_id=None #"output/signature_model/best_model" #model already done 5 epochs (but needs to fix)
    system_prompt = "Detect all signatures and return their locations and labels in the form of coordinates. "
    model=LocalLLM(model_id=model_id, system_prompt=system_prompt,adapter_id=adapter_id)
    model.load()
    model.train(
    train_data="data/train.jsonl",
    val_data="data/valid.jsonl",
    image_dir="data",
    output_dir="output/signature_model",
    
    # --- Performance & Memory Settings ---
    epochs=1,
    batch_size=1,                    # Keep this at 1 to avoid OOM
    gradient_accumulation_steps=8,   # Simulates a batch of 8
    num_workers=4,                   # Keeps the GPU fed with data
    lr=2e-4, #1e-3,                         # Standard for LoRA changed after 4 epochs from 2e-4 to 1e-3
    lora_rank=8,                          # Low-rank approximation for LoRA
    )
    # model.train(
    #     train_data="./data/train.jsonl", # <--- USE TINY DATA
    #     val_data="./data/valid.jsonl",   # Validate on same data (we want overfitting!)
        
    #     epochs=1,               # Run many epochs (it takes seconds)
    #     batch_size=1,
    #     gradient_accumulation_steps=1, # <--- UPDATE EVERY STEP (Instant feedback)
        
    #     # Use the stable settings we discussed
    #     lr=2e-4,                 # Stable LR
    #     lora_rank=16,
        
    #  # (Make sure you added the clipping line in your loop)
    # )
    
if __name__ == "__main__":
    fine_tune_model()
