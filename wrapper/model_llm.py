
from wrapper.model import Model
import json
import regex as re
from typing import List, Dict, Any

class LLM(Model):
    """
    Shared logic for ALL Large Language Models (Local or API).
    Handles the common problem: "Parsing messy text into JSON".
    """
    def __init__(self, model_id: str, system_prompt: str = None):
        super().__init__(model_id)
        self.system_prompt = system_prompt or (
            ''' You are an object detector. Return a strictly valid JSON list. 
                Format: [{'box_2d': [xmin, ymin, xmax, ymax], 'label': 'signature'}]. 
                Detect the bounding box of the signature.
            '''
        )

    def postprocess(self, raw_results: List[Dict], img_h: int, img_w: int, normalize: bool = True) -> List[Dict]:
            """
            Converts 0-1000 model coordinates back to absolute pixels.
            
            Args:
                raw_results (List[Dict]): Output from the model (e.g., [{'box_2d': [200, 200, 500, 500], 'label': 'signature'}])
                img_h (int): Height of the original image.
                img_w (int): Width of the original image.
                normalize (bool): If True, converts 0-1000 range -> Absolute Pixels. 
                                (Note: 'normalize' usually means 0-1, but here it implies processing the norm-coords).
            
            Returns:
                List[Dict]: The same list structure with updated 'box_2d' values.
            """
            processed_results = []

            for el in raw_results:
                # Create a copy to avoid modifying the original list in place
                new_el = el.copy()
                
                # Extract the box (Assumes format [xmin, ymin, xmax, ymax])
                if "box_2d" in new_el:
                    bbox = new_el["box_2d"]
                    
                    if normalize and img_h is not None and img_w is not None:
                        # Logic: (Value / 1000) * Real_Dimension
                        x_min = (bbox[0] / 1000) * img_w
                        y_min = (bbox[1] / 1000) * img_h
                        x_max = (bbox[2] / 1000) * img_w
                        y_max = (bbox[3] / 1000) * img_h
                        
                        # Update the box with integers
                        new_el["box_2d"] = [int(x_min), int(y_min), int(x_max), int(y_max)]
                    else:
                        # Return as-is (already pixels or raw)
                        new_el["box_2d"] = bbox

                processed_results.append(new_el)
                
            return processed_results

