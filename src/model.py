import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision.ops import roi_align

# Import the custom module we built previously
from src.locator import LowLevelLocator 

class RODMLLM(nn.Module):
    def __init__(self, llm_path="lmsys/vicuna-7b-v1.5", device="cuda"):
        super().__init__()
        self.device = device
        
        # --- 1. The Low-Level Locator (Composition) ---
        # We use the class from locator.py to handle OVD and N-gram extraction
        print("Loading Low-Level Locator (OWLv2)...")
        self.locator_module = LowLevelLocator(device=device) 
        
        # --- 2. The Visual Encoder (CLIP) ---
        print("Loading Visual Encoder (CLIP)...")
        self.visual_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.visual_encoder.to(self.device).eval() # Frozen
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        # --- 3. The High-Level LLM (Vicuna-7B) ---
        print("Loading LLM (Vicuna-7B) in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        # self.llm = AutoModelForCausalLM.from_pretrained(
        #     llm_path, 
        #     quantization_config=bnb_config, 
        #     device_map="auto"
        # )
        
        self.llm = RODVicunaForCausalLM.from_pretrained(
            llm_path, 
            quantization_config=bnb_config, 
            device_map="auto"
        )

        # After loading sub-modules, link them:
        self.llm.set_visual_modules(
            self.visual_encoder, 
            self.global_projector, 
            self.regional_projector
        )
        # Add special tokens
        self._add_special_tokens()

        # --- 4. Projectors ---
        self._build_projectors()

    def _add_special_tokens(self):
        special_tokens = ["<p>", "</p>", "<box>", "</box>"]
        anchor_tokens = [f"<a{i}>" for i in range(100)]
        self.tokenizer.add_tokens(special_tokens + anchor_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

    def _build_projectors(self):
        embed_dim = self.visual_encoder.config.hidden_size # 1024
        llm_dim = self.llm.config.hidden_size             # 4096
        
        # Global Projector (Section 3.1)
        self.global_projector = nn.Sequential(
            nn.Linear(embed_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        ).to(self.device)

        # Regional Projector (Section 3.2: 2x2 patches)
        # Input: 1024 (CLIP) -> Output: 4096 (LLM)
        self.regional_projector = nn.Sequential(
            nn.Linear(embed_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        ).to(self.device)

    def forward_inference(self, image, prompt):
        """
        Full End-to-End Inference Pipeline:
        1. Locator extracts candidates (N-grams -> OWLv2)
        2. Visual Encoder extracts feature map
        3. ROI Align extracts regional features
        4. LLM generates answer
        """
        # Step 1: Low-Level Localization
        # Note: We rely on the locator module to handle the N-gram logic internally
        boxes, _ = self.locator_module.get_candidate_boxes(image, prompt)
        
        if len(boxes) == 0:
            return "None"

        # Step 2: Global Features
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        clip_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
        
        # Last layer feature map for ROI Align (Shape: B x 1024 x 24 x 24 for 336px)
        feature_map = clip_outputs.hidden_states[-1] 
        # Second to last for global token (as per paper)
        global_embed = self.global_projector(clip_outputs.hidden_states[-2][:, 0, :])

        # Step 3: Regional Features (ROI Align)
        # Prepare boxes for torchvision roi_align (needs batch index)
        box_tensors = [boxes.to(self.device)] 
        
        # Output: [Num_Boxes, 1024, 2, 2] -> We use 2x2 output size as per Sec 3.2
        roi_features = roi_align(
            feature_map.permute(0, 3, 1, 2), # NCHW format
            box_tensors, 
            output_size=(2, 2),
            spatial_scale=1.0/14.0 # CLIP patch size is 14
        )
        
        # Flatten 2x2 -> 4 tokens per region
        # [Num_Boxes, 1024, 2, 2] -> [Num_Boxes, 4, 1024]
        roi_features = roi_features.flatten(2).transpose(1, 2) 
        
        # Project to LLM space -> [Num_Boxes, 4, 4096]
        region_tokens = self.regional_projector(roi_features)

        # Step 4: Construct LLM Input (Pseudo-code for brevity)
        # In a real run, you'd insert these embeddings into the embedding layer
        # alongside the text tokens.
        
        return "Model ready for generation loop..."
