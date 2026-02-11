import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

class RODMLLM(nn.Module):
    
    # In src/model.py

    def __init__(self, llm_model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", train_mode=True):
        super().__init__()
        self.device = device
        
        print(f">> Loading Qwen2.5-VL (Speed Mode): {llm_model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            llm_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa" 
        )
        
        # --- AGGRESSIVE OPTIMIZATION ---
        # We limit the model to seeing roughly 512x512 pixels max.
        # 512 * 512 = 262,144 pixels
        # 256 * 28 * 28 = 200,704 pixels
        self.processor = AutoProcessor.from_pretrained(
            llm_model_id, 
            min_pixels=256*28*28, 
            max_pixels=768*768  # Cap at ~500px resolution
        )

        if train_mode:
            print(">> Applying LoRA Adapters...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,              # Reduce Rank from 16 -> 8 (Faster, less memory)
                lora_alpha=16,    # Scale alpha accordingly
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"] # Only train Attention layers (Fastest)
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels=None):
        """
        Forward pass.
        Qwen handles the embedding fusion internally.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, pixel_values, image_grid_thw, attention_mask=None, max_new_tokens=128):
        """
        Helper for inference/testing.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens
        )
    
    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
