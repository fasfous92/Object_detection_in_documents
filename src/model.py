import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, BitsAndBytesConfig
from src.locator import LowLevelLocator

class RODMLLM(nn.Module):
    def __init__(self, llm_model_id="lmsys/vicuna-7b-v1.5", device="cuda"):
        super().__init__()
        self.device = device
        
        print(f"Loading LLM in 4-bit: {llm_model_id}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id, use_fast=False)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # We don't rely on token replacement anymore, but we keep the token for compatibility
        self.image_token = "<image>"
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        print("Loading Visual Encoder (CLIP)...")
        self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)
        self.locator_module = LowLevelLocator(device=self.device)
        
        # Projectors
        self.global_projector = nn.Sequential(
            nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096)
        ).to(self.device)
        
        self.regional_projector = nn.Sequential(
            nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096)
        ).to(self.device)

    def encode_images(self, pixel_values):
        outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
        # Take feature map (Batch, 576, 1024)
        features = outputs.hidden_states[-1][:, 1:, :] 
        return self.global_projector(features)

    def forward(self, input_ids, pixel_values, labels=None):
        # 1. Get Text Embeddings
        # (Batch, Seq_Len, 4096)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. Get Visual Features
        # (Batch, 576, 4096)
        image_features = self.encode_images(pixel_values)
        
        # 3. FORCE MERGE (Prepend Logic)
        # We take the mean to get 1 visual token per image
        # (Batch, 1, 4096)
        image_mean = image_features.mean(dim=1, keepdim=True)
        
        # CRITICAL FIX: Ensure Dtypes Match
        # The LLM expects float16 (from bnb config), Projector outputs float32.
        image_mean = image_mean.to(inputs_embeds.dtype)
        
        # Concatenate: [Visual_Token, Text_Tokens]
        combined_embeds = torch.cat([image_mean, inputs_embeds], dim=1)
        
        # 4. Handle Labels
        if labels is not None:
            # We must prepend a "dummy" label (-100) for the visual token
            # so the model isn't trained to predict the text FROM the image token itself,
            # but predicts the next text token GIVEN the image.
            dummy_label = torch.full((labels.shape[0], 1), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([dummy_label, labels], dim=1)
        else:
            combined_labels = None

        # 5. Pass to LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds, 
            labels=combined_labels
        )
        return outputs
