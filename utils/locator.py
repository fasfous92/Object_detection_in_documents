import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class LowLevelLocator:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.model.eval()

    def get_candidate_boxes(self, image, text_queries, threshold=0.1):
        """
        Returns candidate bounding boxes for the given text queries.
        
        Args:
            image (PIL.Image): Input image
            text_queries (str or list): Text to search for (e.g., "signature")
            threshold (float): Score threshold to filter weak predictions
            
        Returns:
            boxes (Tensor): [N, 4] Normalized boxes (x1, y1, x2, y2)
            scores (Tensor): [N] Confidence scores
        """
        if isinstance(text_queries, str):
            text_queries = [text_queries]

        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Target image size (height, width)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        
        # Convert outputs (bounding boxes and class logits) to COCO-style (x1, y1, x2, y2)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=threshold
        )
        
        # We only processed one image, so take the first result
        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        
        # Normalize boxes to [0, 1] relative to image size
        w, h = image.size
        # x1, y1, x2, y2
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        
        return boxes, scores
