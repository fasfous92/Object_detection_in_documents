import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Import our custom modules
from src.model import RODMLLM
from src.wrapper import RODVicunaForCausalLM
# Assuming you have the dataset from previous steps, or we use mock data
from src.ingestion import RODDataset 

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="1=Alignment, 2=Instruction Tuning")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2) # Small for Kaggle T4
    parser.add_argument("--grad_accum", type=int, default=64) # Simulate 128 batch size
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_mock_data", action="store_true", help="Run with random tensors for testing")
    return parser.parse_args()

def setup_model_for_training(model, stage):
    """
    Configures freezing/unfreezing based on Section 5.1 of the paper.
    """
    # 1. Always freeze the Vision Encoder and Locator
    model.visual_encoder.requires_grad_(False)
    model.locator_module.model.requires_grad_(False) # The OWLv2 detector is frozen
    
    # 2. Config based on Stage
    if stage == 1:
        print(">> STAGE 1 SETUP: Training Projectors Only")
        # Train Projectors
        model.global_projector.requires_grad_(True)
        model.regional_projector.requires_grad_(True)
        
        # Freeze LLM
        model.llm.requires_grad_(False)
        
    elif stage == 2:
        print(">> STAGE 2 SETUP: Training LoRA + Projectors")
        # Apply LoRA to LLM (Rank 128, Alpha 256 as per paper)
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Wrap the internal LLM with Peft
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
        
        # Ensure projectors are still trainable (usually fine-tuned alongside)
        model.global_projector.requires_grad_(True)
        model.regional_projector.requires_grad_(True)

    return model

def mock_data_collator(batch_size, tokenizer):
    """Generates dummy batch for sanity check."""
    # Dummy Image: [3, 336, 336]
    images = torch.randn(batch_size, 3, 336, 336)
    
    # Dummy Text
    prompts = ["Locate <p>cat</p>"] * batch_size
    
    # Dummy Labels (Input IDs masked with -100)
    # Just a simple sequence for testing gradient flow
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    labels = input_ids.clone()
    
    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "labels": labels,
        "prompts": prompts # Needed for the locator to extract queries
    }

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize Model
    # Note: We use the custom wrapper implicitly via the RODMLLM class logic
    # Make sure src/model.py imports RODVicunaForCausalLM as self.llm
    model = RODMLLM(device=device)
    
    # 2. Configure Layers (Freeze/Unfreeze)
    model = setup_model_for_training(model, args.stage)
    model.to(device)

    # 3. Optimizer
    # Filter only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    # 4. Data Loader
    if args.use_mock_data:
        print("(!) Using MOCK DATA - No real files loaded.")
        dataloader = range(10) # 10 dummy steps
    else:
        # Assuming you have run the ingestion script
        dataset = RODDataset(
            annotation_file="data/ROD/rod_dataset.json", 
            image_dir="data/images/"
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting Training Stage {args.stage}...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            
            # A. Prepare Batch
            if args.use_mock_data:
                batch = mock_data_collator(args.batch_size, model.tokenizer)
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            prompts = batch["prompts"] # Raw text for the locator
            
            # B. Forward Pass Strategy
            # The RODMLLM pipeline is complex:
            # 1. Locator -> Boxes
            # 2. Vision -> Global Feats
            # 3. Boxes + Vision -> Regional Feats
            # 4. Feats + Text -> LLM -> Loss
            
            # Step 1: Run Locator (Frozen) to get candidates
            # We do this inside the loop because boxes depend on the text prompts
            all_region_features = []
            
            # We need to process global features once
            with torch.no_grad():
                 clip_outputs = model.visual_encoder(pixel_values, output_hidden_states=True)
                 feature_map = clip_outputs.hidden_states[-1] # [B, 1024, 24, 24]
            
            batch_region_embeds = []
            
            for i, prompt in enumerate(prompts):
                # 1. Get Boxes for this sample
                # Note: model.locator_module is pre-loaded in RODMLLM
                boxes, _ = model.locator_module.get_candidate_boxes(
                    torch.from_numpy(pixel_values[i].cpu().numpy().transpose(1, 2, 0)).to(torch.uint8), # Rough conversion for API
                    prompt
                )
                
                # If no boxes, we might need a dummy box to prevent crash, or handle empty logic
                if len(boxes) == 0:
                    # Create a 0-size dummy for robustness
                    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
                
                # 2. Extract Regional Features (ROI Align)
                # We reuse the logic from model.py but need to call it manually here 
                # to get the tensor for the LLM wrapper
                from torchvision.ops import roi_align
                
                # feature_map[i] is [1024, 24, 24]
                # roi_align expects [N, C, H, W]
                single_map = feature_map[i].unsqueeze(0) 
                
                roi_feats = roi_align(
                    single_map,
                    [boxes.to(device)],
                    output_size=(2, 2),
                    spatial_scale=1.0/14.0
                ) # [Num_Boxes, 1024, 2, 2]
                
                roi_feats = roi_feats.flatten(2).transpose(1, 2) # [Num_Boxes, 4, 1024]
                
                # Project
                region_tokens = model.regional_projector(roi_feats) # [Num_Boxes, 4, 4096]
                batch_region_embeds.append(region_tokens)

            # Pad region embeds to stack them (since num_boxes varies)
            # For simplicity in this test script, we just take the first sample's logic
            # In production, you pass a list to the wrapper or mask it.
            # Here we pass the list to our custom wrapper if it supports it, 
            # OR we assume batch_size=1 for safety on Kaggle.
            
            # Step 3: LLM Forward
            # We call the wrapped LLM. It calculates Causal LM loss automatically if labels are passed.
            outputs = model.llm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                # region_features=batch_region_embeds, # Pass the list! Wrapper must handle it
                labels=labels
            )
            
            loss = outputs.loss
            
            # C. Backward (with Gradient Accumulation)
            loss = loss / args.grad_accum
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item() * args.grad_accum})

    print("✓ Training Test Complete.")

if __name__ == "__main__":
    args = get_config()
    train(args)
