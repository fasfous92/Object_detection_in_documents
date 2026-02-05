import torch
from transformers import LlamaForCausalLM
from typing import List, Optional

class RODVicunaForCausalLM(LlamaForCausalLM):
    """
    Wrapper for Vicuna to support multimodal inputs (Text + Image + Region).
    """
    def set_visual_modules(self, visual_encoder, global_projector, regional_projector):
        self.visual_encoder = visual_encoder
        self.global_projector = global_projector
        self.regional_projector = regional_projector

    def prepare_inputs_embeds(
        self, 
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        region_features: Optional[torch.FloatTensor] = None, 
        **kwargs
    ):
        # 1. Get standard text embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        target_dtype = inputs_embeds.dtype

        # 2. Inject Global Visual Features (if image provided)
        if pixel_values is not None:
            with torch.no_grad():
                clip_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
                global_feat = clip_outputs.hidden_states[-2][:, 0, :] 
                
                # Project and CAST to target_dtype
                global_visual_embeds = self.global_projector(global_feat).unsqueeze(1)
                global_visual_embeds = global_visual_embeds.to(target_dtype)

            # Prepend visual token
            inputs_embeds = torch.cat((global_visual_embeds, inputs_embeds), dim=1)

        # 3. Inject Regional Features (Placeholder)
        if region_features is not None:
             pass 

        return inputs_embeds.to(target_dtype)

    def forward(self, input_ids=None, pixel_values=None, region_features=None, labels=None, **kwargs):
        """
        Overridden forward to handle Label Padding + Embedding Injection
        """
        # If calling logic has already calculated embeddings, pass through (rare)
        if "inputs_embeds" in kwargs:
            return super().forward(labels=labels, **kwargs)
        
        # 1. Calculate the Multimodal Embeddings
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, region_features)
        
        # 2. Fix Labels (The Critical Fix)
        # If we added a visual token, we must add a dummy label so shapes match.
        if labels is not None and pixel_values is not None:
            # Create a column of -100 (Ignore Index)
            # Shape: [Batch_Size, 1]
            dummy_labels = torch.full(
                (labels.shape[0], 1), 
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            # Prepend to match the prepended visual embedding
            labels = torch.cat((dummy_labels, labels), dim=1)
        
        # 3. Pass to original Llama forward
        return super().forward(inputs_embeds=inputs_embeds, labels=labels, **kwargs)
