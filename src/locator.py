import torch
import nltk
import re
from nltk.util import ngrams
from torchvision.ops import nms
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LowLevelLocator:
    """
    Implements the Low-Level Localization module (Section 3.1).
    Responsible for generating candidate bounding boxes using OWLv2
    based on N-gram queries extracted from user prompts.
    """
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", device="cuda"):
        self.device = device
        print(f"Loading Locator: {model_name}...")
        
        # Load Processor and Model
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval() # Always frozen during training (Section 5.1)

        # Common objects for context-agnostic perception (Eq. 1 & Section 5.1)
        # Used to ensure valid candidates exist even if the query is obscure.
        self.common_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

    def extract_query_objects(self, text):
        """
        Parses user prompt to extract object queries using {1, 2, L}-grams.
        Logic based on Section 3.1 and 5.1.
        
        Example: "Locate <p>horseman's helmet</p>" 
        -> Extracts "horseman's helmet" -> Generates ["horseman", "helmet", "horseman's helmet"...]
        """
        # 1. Extract content between <p> tags
        # Regex finds all occurrences of <p>...</p>
        matches = re.findall(r'<p>(.*?)</p>', text)
        
        # If no tags found (e.g. during simple inference), use the whole text
        raw_queries = matches if matches else [text]
        
        all_ngrams = []
        
        for raw_query in raw_queries:
            # Clean and tokenize
            clean_text = raw_query.strip()
            if not clean_text: continue
            
            tokens = nltk.word_tokenize(clean_text.lower())
            L = len(tokens)
            
            # Generate {1, 2, L}-grams (Section 5.1)
            # We use a set to avoid duplicates (e.g., if L=1 or L=2)
            gram_sizes = {1, 2, L}
            
            for n in gram_sizes:
                if n <= L and n > 0:
                    generated_grams = ngrams(tokens, n)
                    all_ngrams.extend([" ".join(gram) for gram in generated_grams])
        
        # Remove duplicates and empty strings
        unique_queries = list(set([q for q in all_ngrams if q]))
        return unique_queries

    @torch.no_grad()
    def get_candidate_boxes(self, image, prompt):
        """
        Main pipeline for Stage 1 Localization.
        Returns:
            boxes (Tensor): [N, 4] Normalized coordinates (0-1)
            scores (Tensor): [N] Confidence scores
        """
        # 1. Prepare Queries: Combine extracted N-grams with Common Objects
        query_objs = self.extract_query_objects(prompt)
        
        # Fallback: If N-gram extraction yields nothing, rely solely on common objects
        # This prevents the locator from crashing on empty/malformed prompts
        if not query_objs:
            text_queries = self.common_objects
        else:
            text_queries = query_objs + self.common_objects
        
        # 2. Preprocess for OWLv2
        # Note: OWLv2 handles tokenization internally via the processor
        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt").to(self.device)
        
        # 3. Model Inference
        outputs = self.model(**inputs)
        
        # 4. Post-Process (Convert outputs to boxes)
        # Target size is needed to scale boxes to original image dimensions
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        
        # Threshold=0.12 (Section 5.1: "retaining up to 100 boxes with confidence scores above 0.12")
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=0.12, 
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"]  # [N, 4] (x1, y1, x2, y2)
        scores = results["scores"] # [N]

        # 5. Non-Maximum Suppression (NMS)
        # Threshold=0.6 (Section 5.1)
        if len(boxes) > 0:
            keep_idx = nms(boxes, scores, iou_threshold=0.6)
            
            # Limit to top 100 boxes (Section 5.1)
            keep_idx = keep_idx[:100]
            
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
        
        # 6. Normalize Boxes for ROI Align
        # ROI Align in model.py usually expects unnormalized coords if we pass spatial_scale,
        # BUT standard practice for MLLM coordinate tokens (like <box>) is often [0, 1000] or [0, 1].
        # For the internal ROI Align using torchvision, we keep them as absolute pixel coords
        # (matching the image size) because we calculated spatial_scale relative to that.
        
        return boxes, scores
