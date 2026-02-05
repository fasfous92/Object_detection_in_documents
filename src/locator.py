
import torch
import nltk
import re
from nltk.util import ngrams
from torchvision.ops import nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection

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
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval() # Always frozen during training (Section 5.1)

        # Common objects for context-agnostic perception (Eq. 1 & Section 5.1)
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
        """
        matches = re.findall(r'<p>(.*?)</p>', text)
        raw_queries = matches if matches else [text]
        
        all_ngrams = []
        for raw_query in raw_queries:
            clean_text = raw_query.strip()
            if not clean_text: continue
            
            tokens = nltk.word_tokenize(clean_text.lower())
            L = len(tokens)
            
            gram_sizes = {1, 2, L}
            for n in gram_sizes:
                if n <= L and n > 0:
                    generated_grams = ngrams(tokens, n)
                    all_ngrams.extend([" ".join(gram) for gram in generated_grams])
        
        unique_queries = list(set([q for q in all_ngrams if q]))
        return unique_queries

    @torch.no_grad()
    def get_candidate_boxes(self, image, prompt):
        """
        Main pipeline for Stage 1 Localization.
        """
        query_objs = self.extract_query_objects(prompt)
        
        if not query_objs:
            text_queries = self.common_objects
        else:
            text_queries = query_objs + self.common_objects
        
        # 1. Run Processor
        # Note: If image is a normalized Tensor, this might be suboptimal for OWLv2 
        # (which expects raw inputs), but for the purpose of the training loop flow 
        # it is acceptable.
        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs)
        
        # 2. Determine Target Size for Post-Processing
        # FIX: Handle both PyTorch Tensors (Training) and PIL Images (Inference)
        if isinstance(image, torch.Tensor):
            # Tensor shape is usually [C, H, W] -> We need [H, W]
            h, w = image.shape[-2:]
            target_sizes = torch.tensor([[h, w]]).to(self.device)
        else:
            # PIL Image size is (W, H) -> We need [H, W]
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=0.12, 
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]

        # 3. NMS Filtering
        if len(boxes) > 0:
            keep_idx = nms(boxes, scores, iou_threshold=0.6)
            keep_idx = keep_idx[:100]
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
        
        return boxes, scores
