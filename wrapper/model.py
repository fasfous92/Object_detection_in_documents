from abc import ABC, abstractmethod
import json
import re
from typing import List, Dict, Any
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


class Model(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def load(self):
        """Load weights (Local) or Authenticate (API)"""
        pass

    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        """Returns RAW string output"""
        pass

    @abstractmethod
    def postprocess(self, raw_results, img_h: int = None, img_w: int = None) -> List[Dict]:
        """
        img_h, img_w: Optional dimensions used to denormalize coordinates smartly.
        """
        pass

    def plot_bounding_boxes(self, image_path, bbox_data, height, width, ground_truth=None,normalize=True):
        """
        Plots bounding boxes assuming Qwen-style format: [xmin, ymin, xmax, ymax] (0-1000).
        
        Args:
            image_path (str): Path to the image file.
            bbox_data (list): List of dicts [{'label': 'car', 'box_2d': [x1, y1, x2, y2]}].
            height (int): The height to resize the image to (e.g., model input height).
            width (int): The width to resize the image to.
            ground_truth (list): Optional list of GT boxes [x1, y1, x2, y2].
        """
        # 1. Load and Resize Image
        # We resize to match the dimensions the model actually "saw" or output
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((width, height))
        except Exception as e:
            print(f"Error processing image: {e}")
            return

        # 2. Setup Plot
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Detection Results: {self.model_id}")

        # 3. Plot Predictions (Red)
        # Assumes bbox_data is: [xmin, ymin, xmax, ymax] in 0-1000 scale
        for item in bbox_data:
            bbox = item.get('bbox_2d', [])
            label = item.get('label', 'object')
            
            if len(bbox) != 4: continue

            # SCALING LOGIC: 0-1000 -> Pixel Coordinate (width/height)
            # x_pixel = (x_1000 / 1000) * width
          
            if normalize:
                x_min = (bbox[0]/1000) * width
                y_min = (bbox[1]/1000) * height
                x_max = (bbox[2]/1000) * width
                y_max = (bbox[3]/1000) * height
            else: 
                x_min = bbox[0]
                y_min = bbox[1] 
                x_max = bbox[2]
                y_max = bbox[3]
                
            # Calculate width/height for Matplotlib Rectangle
            box_w = x_max - x_min
            box_h = y_max - y_min

            # Create Rectangle: (x, y), width, height
            # We strictly follow Qwen format where bbox[0] is X-min (Left)
            rect = patches.Rectangle(
                (x_min, y_min), box_w, box_h, 
                linewidth=2, edgecolor='red', facecolor='none', label='Prediction'
            )
            ax.add_patch(rect)
            
            # Add Label
            plt.text(
                x_min, y_min - 5, label, 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1)
            )

        # 4. Plot Ground Truth (Green)
        if ground_truth:
            for item in ground_truth:
                # Assuming GT is ALREADY in pixels [xmin, ymin, xmax, ymax]
                # If GT is also 0-1000, apply the same scaling math as above.
                gt_x1, gt_y1, gt_x2, gt_y2 = item[0], item[1], item[2], item[3]
                
                rect = patches.Rectangle(
                    (gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, 
                    linewidth=2, edgecolor='#00FF00', facecolor='none', linestyle='--', label='Ground Truth'
                )
                ax.add_patch(rect)

        # 5. Handle Legend (Deduplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.show()
