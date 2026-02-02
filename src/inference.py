import torch
import re
from PIL import Image
from src.model import RODMLLM

class RODInference:
    def __init__(self, model_path="lmsys/vicuna-7b-v1.5", device="cuda"):
        self.device = device
        # Initialize the full composite model
        self.model = RODMLLM(llm_path=model_path, device=device)
        self.model.eval()

    def format_prompt(self, user_text, num_regions):
        """
        Constructs the multimodal prompt as shown in Figure 3.
        Format: <image> ... <image> User Instruction <region>...<region> <a0>...<an>
        """
        # Note: The specific prompt template depends on the Vicuna version, 
        # but here we follow the paper's logical structure.
        
        # 1. Visual Tokens (Global) - usually handled by the embedding layer insertion
        # distinct from text prompt, but we placeholders here for clarity
        
        # 2. Region Tokens & Anchors
        # Each region has 4 visual tokens + 1 anchor token (Section 3.2)
        region_part = ""
        for i in range(num_regions):
            # We don't actually type the visual tokens, they are inserted embeddings.
            # We only need to format the anchor tokens so the LLM knows 'who is who'.
            region_part += f"<a{i}>" 
        
        # Standard Vicuna conversation format
        # System prompt + User + Assistant
        prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: <image>\n{user_text}\n"
            f"Here are candidate regions: {region_part}\n"
            "ASSISTANT:"
        )
        return prompt

    def decode_response(self, response_text, candidate_boxes):
        """
        Parses the LLM output to map anchor tokens back to bounding boxes.
        Example Output: "<box>[<a4><a5>]</box>" -> returns boxes[4] and boxes[5]
        """
        # Check for explicit rejection (Reliability feature)
        if "None" in response_text:
            return [], "None"

        # Extract anchor indices using Regex
        # Matches <a12>, <a3>, etc.
        anchor_matches = re.findall(r'<a(\d+)>', response_text)
        
        selected_boxes = []
        for idx_str in anchor_matches:
            idx = int(idx_str)
            if idx < len(candidate_boxes):
                selected_boxes.append(candidate_boxes[idx].tolist())
        
        return selected_boxes, response_text

    @torch.no_grad()
    def predict(self, image_path, user_prompt):
        """
        End-to-End Inference
        """
        image = Image.open(image_path).convert("RGB")
        
        # 1. Low-Level Localization (Get Candidate Boxes)
        candidate_boxes, _ = self.model.locator_module.get_candidate_boxes(image, user_prompt)
        
        if len(candidate_boxes) == 0:
            return [], "None (No candidates found)"

        # 2. Forward Pass (Get Embeddings)
        # In a real Hugging Face 'generate' loop, we need to inject embeddings.
        # This part requires overriding the model's prepare_inputs_embeds.
        # For this script, we simulate the flow to show the logic.
        
        # Get Global & Regional Embeddings
        # (Using the methods we defined in model.py)
        # Note: Implementation of embedding injection into .generate() is complex
        # and requires a custom 'VicunaForCausalLM' wrapper. 
        # Below is the logic flow:
        
        # A. Format Text
        full_prompt = self.format_prompt(user_prompt, len(candidate_boxes))
        input_ids = self.model.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # B. Generate
        # To make this runnable without the full embedding-injection wrapper,
        # we assume standard generation for now. 
        output_ids = self.model.llm.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False, # Deterministic for evaluation
            temperature=0.0
        )
        
        response_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        
        # 3. Parse Result
        final_boxes, final_text = self.decode_response(response_text, candidate_boxes)
        
        return final_boxes, final_text

# --- Quick Test ---
if __name__ == "__main__":
    # Create the inference engine
    engine = RODInference()
    
    # Test on a dummy image
    img_path = "data/images/test.jpg" # Make sure this exists
    prompt = "Locate <p>the red car</p> in the image."
    
    print(f"Running inference on {img_path}...")
    # Note: This will likely fail without a real image file, just for structure demo
    try:
        boxes, response = engine.predict(img_path, prompt)
        print(f"Result: {response}")
        print(f"Boxes found: {len(boxes)}")
    except FileNotFoundError:
        print("(!) Please add a valid image to data/images/test.jpg to run.")
