
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
            "You are an object detector. Return a strictly valid JSON list. "
            "Format: [{'label': 'name', 'box_2d': [ymin, xmin, ymax, xmax]}]. "
            "Coordinates must be normalized 0-1000."
        )

    def postprocess(self, raw_results: str) -> List[Dict]:
        # 1. Clean Markdown (```json ... ```)
        clean_text = raw_results.replace("```json", "").replace("```", "").strip()
        
        # 2. Extract JSON List using Regex (Robust fallback)
        try:
            # Look for the first '[' and the last ']'
            match = re.search(r'\[.*\]', clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass # Fallback to direct load

        # 3. Direct Load
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print(f"JSON Parse Error on: {raw_results[:50]}...")
            return []

