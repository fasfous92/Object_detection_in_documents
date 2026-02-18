import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig ,
    get_cosine_schedule_with_warmup
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
import numpy as np
from tqdm import tqdm

# Adjust this import based on your actual file structure
from wrapper.model_llm import LLM 
from wrapper.dataset_handler import GroundingDataset


try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False
    




class LocalLLM(LLM):
    def __init__(self, model_id: str, adapter_id: str = None, device="cuda", system_prompt: str = None, load_in_4bit: bool = False):
        """
        Args:
            load_in_4bit (bool): If True, loads the model in 4-bit precision (requires less VRAM).
        """
        super().__init__(model_id, system_prompt)
        self.adapter_id = adapter_id
        self.load_in_4bit = load_in_4bit # <--- Store the flag
        
        # BitsAndBytes requires CUDA; fail gracefully if user tries 4bit on CPU
        if self.load_in_4bit and not torch.cuda.is_available():
            print("⚠️ Warning: 4-bit quantization requires a GPU. Falling back to non-quantized loading.")
            self.load_in_4bit = False
            
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load(self, use_adapters: bool = True):
        print(f"⚙️ Loading {self.model_id} (4-bit: {self.load_in_4bit})...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        # 1. Define Quantization Config
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, # Compute in fp16 for speed
                bnb_4bit_use_double_quant=True,       # Save even more memory
                bnb_4bit_quant_type="nf4"             # Standard 4-bit type
            )
        
        # 2. Determine Dtype (Use float16 if not using 4bit, otherwise let config handle it)
        # Note: When using quantization_config, we don't pass torch_dtype usually, 
        # or we pass the compute dtype.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 3. Load Model with Config
        # We assume 'auto' device map for quantization (it handles offloading better)
        # but self.device ("cuda") works if it fits on one GPU.
        load_kwargs = {
            "device_map": self.device,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
        }
        
        # Only add torch_dtype if NOT quantizing (BnB handles its own storage type)
        if not self.load_in_4bit:
            load_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **load_kwargs)
        except:
            # Fallback for models classified as CausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        # 4. Load Adapters (LoRA)
        if use_adapters and self.adapter_id:
            print(f"🔗 Merging Adapter: {self.adapter_id}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_id)
            
        print("✅ Model Ready.")

    def predict(self, image_path: str, prompt: str):
        if not self.model: self.load()
        
        # 1. Prepare Standard Message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"{self.system_prompt}\n{prompt}"},
                ],
            }
        ]

        input_h, input_w = 0, 0

        # --- STRATEGY A: QWEN-VL ---
        if "qwen" in self.model_id.lower() and HAS_QWEN_UTILS:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            if 'image_grid_thw' in inputs:
                grid = inputs['image_grid_thw'][0] 
                input_h = int(grid[1] * 14)
                input_w = int(grid[2] * 14)
            else:
                img = Image.open(image_path)
                input_w, input_h = img.size

        # --- STRATEGY B: GENERIC ---
        else:
            image = Image.open(image_path).convert("RGB")
            input_w, input_h = image.size
            
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if "<image>" not in formatted_prompt and "llava" in self.model_id.lower():
                formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

        # 3. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False
            )

        # 4. Trim
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, input_h, input_w

    def _get_collator(self):
        def collator(batch):
            # 1. Prepare Data
            # We must merge the User Prompt and the Assistant Answer into one conversation
            full_conversations = []
            user_prompts_only = [] # We keep this to calculate where to mask later
            
            for item in batch:
                # Get the user part (contains the image path)
                user_msg = item["messages"]
                
                # Get the assistant part (the label)
                assistant_text = item["assistant_response"]
                
                # Create the FULL conversation (User + Assistant)
                # Qwen expects the conversation to flow naturally
                full_conv = user_msg + [{"role": "assistant", "content": assistant_text}]
                
                full_conversations.append(full_conv)
                user_prompts_only.append(user_msg)

            # 2. Extract Visuals using the Official Utility
            # This automatically finds images in the 'full_conversations' list
            image_inputs, video_inputs = process_vision_info(full_conversations)

            # 3. Apply Chat Template to get Raw Text
            # We generate the text for the FULL conversation
            text_inputs = [
                self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for conv in full_conversations
            ]
            
            # 4. Tokenize Inputs (Text + Images)
            # This creates 'input_ids' containing: [User Tokens] + [Image Tokens] + [Assistant Tokens]
            batch_out = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                max_length=1500, # Enough space for image tokens (approx 600) + text
                truncation=True
            )

            # 5. Create Labels (The Masking Step)
            # Start by copying input_ids. We will mask the "User" part with -100.
            labels = batch_out["input_ids"].clone()
            
            # We need to find where the "Assistant" answer starts in every sequence
            # The robust way is to re-tokenize just the user prompt to get its length
            
            # Extract visuals for just the prompt (to get correct token length)
            prompt_imgs, prompt_vids = process_vision_info(user_prompts_only)
            
            prompt_texts = [
                self.processor.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in user_prompts_only
            ]
            
            # Tokenize just the prompt to find out how long it is
            prompt_inputs = self.processor(
                text=prompt_texts,
                images=prompt_imgs,
                videos=prompt_vids,
                padding=True, # Padding matches the batch_out strategy
                return_tensors="pt",
                max_length=1500,
                truncation=True
            )
            
            # Apply the mask
            for i in range(len(labels)):
                # Get the length of the prompt (User + Image)
                # We count valid tokens (not padding) in the prompt
                prompt_len = prompt_inputs.attention_mask[i].sum().item()
                
                # Mask everything up to the start of the assistant's answer
                # Note: Qwen's generation prompt adds a few tokens, so this aligns perfectly
                labels[i, :prompt_len] = -100
                
                # Also mask padding tokens in the full sequence (where inputs are 0 or pad_token)
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = -100

            batch_out["labels"] = labels
            
            return batch_out

        return collator
    
    def train(
        self, 
        train_data: str="./data/train.jsonl",
        val_data: str="./data/valid.jsonl",
        output_dir: str = "./output",
        image_dir: str = "./data",       # Default to root data folder
        epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8, 
        num_workers: int = 4,
        lr: float = 2e-4,
        lora_rank: int = 8
    ):
        print(f"🚀 Starting BF16 Training (No Scaler) for {self.model_id}...")
        os.makedirs(output_dir, exist_ok=True)

        if image_dir is None:
            image_dir = os.path.dirname(train_data)
        
        # 1. Prepare Model
        if self.model is None:
            self.load(use_adapters=True) # Ensure adapters are loaded if continuing training!

        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model.gradient_checkpointing_enable()

        self.model.enable_input_require_grads()

        # 2. Setup LoRA (Only if not already attached)
        if not isinstance(self.model, PeftModel):
            print(f"🛠️ Applying LoRA (Rank: {lora_rank})...")
            # peft_config = LoraConfig(
            #     r=lora_rank, 
            #     lora_alpha=lora_rank * 2,
            #     target_modules=["q_proj", "v_proj"], 
            #     task_type=TaskType.CAUSAL_LM,
            #     lora_dropout=0.05, 
            #     bias="none"
            # )
            peft_config = LoraConfig(
                r=lora_rank,      # ⬆️ Increase Rank slightly for spatial tasks
                lora_alpha=32,
                target_modules=[
                    # --- 1. The Brain (LLM Reasoning) ---
                    "q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj",
                    
                    # --- 2. The Connector (Merger) ---
                    # These are the specific layers you found in the log
                    "merger.mlp.0", 
                    "merger.mlp.2",

                    # --- 3. The Retina (Optional but powerful) ---
                    # Helps if signatures are very small or faint
                    "patch_embed.proj"
                ],
                # target_modules=[
                #     "q_proj", "v_proj", "k_proj", "o_proj", 
                #     "gate_proj", "up_proj", "down_proj"
                # ], 
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=0.05, 
                bias="none"
            )
            self.model = get_peft_model(self.model, peft_config)
        
        self.model.print_trainable_parameters()

        # 3. Data Loaders
        print("📦 Loading Datasets...")
        train_ds = GroundingDataset(jsonl_file=train_data, base_image_dir=image_dir)
        val_ds = GroundingDataset(jsonl_file=val_data, base_image_dir=image_dir)
        collator = self._get_collator()

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, 
            collate_fn=collator, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, 
            collate_fn=collator, num_workers=num_workers, pin_memory=True
        )

        # 4. Optimizer
        try:
            from bitsandbytes.optim import PagedAdamW8bit
            print("⚡ Using PagedAdamW8bit.")
            optimizer = PagedAdamW8bit(self.model.parameters(), lr=lr)
        except ImportError:
            print("⚠️ Falling back to standard AdamW.")
            optimizer = AdamW(self.model.parameters(), lr=lr)

        num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        max_train_steps = epochs * num_update_steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * max_train_steps), 
            num_training_steps=max_train_steps
        )
        
        # ❌ SCALER REMOVED (Not needed for bfloat16)

        # 5. Training Loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(pbar):
                if batch is None: continue
                
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                inputs["labels"] = batch["labels"].to(self.device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                # Forward Pass
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs)
                    loss = outputs.loss / gradient_accumulation_steps

                # Backward Pass (Standard)
                loss.backward()

                #clipping 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update Weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()       # <--- DIRECT STEP (No Scaler)
                    scheduler.step()
                    optimizer.zero_grad()
                    # torch.cuda.empty_cache() 
                
                current_loss = loss.item() * gradient_accumulation_steps
                train_loss += current_loss
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            # 6. Validation
            avg_train_loss = train_loss / len(train_loader)
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    if batch is None: continue
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    inputs["labels"] = batch["labels"].to(self.device)
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(**inputs)
                        val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"📉 Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            # 7. Save
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(output_dir, "best_model")
                print(f"💾 Saving Best Model: {save_path}")
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)

        print("✅ Training Complete.")

    # def train(
    #         self, 
    #         train_data: str="./data/train.jsonl",
    #         val_data: str="./data/valid.jsonl",
    #         output_dir: str = "./output",
    #         image_dir: str = "./data",       # Default to root data folder
    #         epochs: int = 3,
    #         batch_size: int = 1,             # Set to 1 for safety on 20GB VRAM
    #         gradient_accumulation_steps: int = 8, # Simulates batch_size=8
    #         num_workers: int = 4,
    #         lr: float = 2e-4,
    #         lora_rank: int = 16
    #     ):
    #         print(f"🚀 Starting Optimized Training for {self.model_id}...")
    #         os.makedirs(output_dir, exist_ok=True)

    #         # 1. Path Logic
    #         if image_dir is None:
    #             image_dir = os.path.dirname(train_data)
            
    #         # 2. Prepare Model
    #         if self.model is None:
    #             self.load(use_adapters=False)

    #         if self.load_in_4bit:
    #             self.model = prepare_model_for_kbit_training(self.model)
            
    #         # Enable Gradient Checkpointing (Saves VRAM, slightly slower)
    #         # If you have extra VRAM later, comment this out for faster speed.
    #         # self.model.gradient_checkpointing_enable()
    #         self.model.enable_input_require_grads()

    #         # 3. LoRA Configuration
    #         if not isinstance(self.model, PeftModel):
    #             print(f"🛠️ Applying LoRA (Rank: {lora_rank})...")
    #             peft_config = LoraConfig(
    #                 r=lora_rank, 
    #                 lora_alpha=lora_rank * 2,
    #                 target_modules=["q_proj", "v_proj"], #, "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    #                 task_type=TaskType.CAUSAL_LM,
    #                 lora_dropout=0.05, 
    #                 bias="none"
    #             )
    #             self.model = get_peft_model(self.model, peft_config)
            
    #         self.model.print_trainable_parameters()

    #         # 4. Data Loaders
    #         print("📦 Loading Datasets...")
    #         train_ds = GroundingDataset(jsonl_file=train_data, base_image_dir=image_dir)
    #         val_ds = GroundingDataset(jsonl_file=val_data, base_image_dir=image_dir)
    #         collator = self._get_collator()

    #         train_loader = DataLoader(
    #             train_ds, batch_size=batch_size, shuffle=True, 
    #             collate_fn=collator, num_workers=num_workers, pin_memory=True
    #         )
    #         val_loader = DataLoader(
    #             val_ds, batch_size=batch_size, shuffle=False, 
    #             collate_fn=collator, num_workers=num_workers, pin_memory=True
    #         )

    #         # 5. Optimized Optimizer (PagedAdamW8bit)
    #         # Saves massive memory compared to standard AdamW
    #         try:
    #             from bitsandbytes.optim import PagedAdamW8bit
    #             print("⚡ Using PagedAdamW8bit for memory efficiency.")
    #             optimizer = PagedAdamW8bit(self.model.parameters(), lr=lr)
    #         except ImportError:
    #             print("⚠️ bitsandbytes optimizer not found. Falling back to standard AdamW.")
    #             optimizer = AdamW(self.model.parameters(), lr=lr)

    #         # Scheduler
    #         # We calculate steps based on accumulation
    #         num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    #         max_train_steps = epochs * num_update_steps_per_epoch
    #         scheduler = get_cosine_schedule_with_warmup(
    #             optimizer, 
    #             num_warmup_steps=int(0.1 * max_train_steps), 
    #             num_training_steps=max_train_steps
    #         )
            
    #         scaler = torch.cuda.amp.GradScaler()

    #         # 6. Training Loop
    #         best_val_loss = float('inf')
            
    #         for epoch in range(epochs):
    #             self.model.train()
    #             train_loss = 0
    #             optimizer.zero_grad() # Initialize gradients
                
    #             pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    #             for step, batch in enumerate(pbar):
    #                 if batch is None: continue
                    
    #                 # Move inputs
    #                 inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
    #                 inputs["labels"] = batch["labels"].to(self.device)
    #                 if "pixel_values" in inputs:
    #                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    #                 # Forward Pass
    #                 with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #                     outputs = self.model(**inputs)
    #                     # Normalize loss by accumulation steps
    #                     loss = outputs.loss / gradient_accumulation_steps

    #                 # Backward Pass (Accumulate gradients)
    #                 scaler.scale(loss).backward()

    #                 # Update Weights (only every X steps)
    #                 if (step + 1) % gradient_accumulation_steps == 0:
    #                     scaler.step(optimizer)
    #                     scaler.update()
    #                     scheduler.step()
    #                     optimizer.zero_grad()
                    
    #                 # Logging (Scale loss back up for display)
    #                 current_loss = loss.item() * gradient_accumulation_steps
    #                 train_loss += current_loss
    #                 pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    #             # 7. Validation
    #             avg_train_loss = train_loss / len(train_loader)
    #             self.model.eval()
    #             val_loss = 0
                
    #             with torch.no_grad():
    #                 for batch in tqdm(val_loader, desc="Validating", leave=False):
    #                     if batch is None: continue
    #                     inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
    #                     inputs["labels"] = batch["labels"].to(self.device)
    #                     if "pixel_values" in inputs:
    #                         inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                        
    #                     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #                         outputs = self.model(**inputs)
    #                         val_loss += outputs.loss.item()
                
    #             avg_val_loss = val_loss / len(val_loader)
    #             print(f"📉 Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    #             # 8. Save
    #             if avg_val_loss < best_val_loss:
    #                 best_val_loss = avg_val_loss
    #                 save_path = os.path.join(output_dir, "best_model")
    #                 print(f"💾 Saving Best Model: {save_path}")
    #                 self.model.save_pretrained(save_path)
    #                 self.processor.save_pretrained(save_path)
