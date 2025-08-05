import os
import re
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from comfy.nodes import NODE_CLASS_MAPPINGS

class MedicalReferenceSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "./medical_bank"}),
                "keyword": ("STRING", {"default": "neuron"}),
                "edge_enhance": ("BOOLEAN", {"default": False}),
                "color_boost": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_reference"

    def load_reference(self, folder_path, keyword, edge_enhance, color_boost):
        # Ensure folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find image file that matches keyword
        pattern = re.compile(keyword, re.IGNORECASE)
        matched_file = None
        for f in os.listdir(folder_path):
            if pattern.search(f) and f.lower().endswith((".png", ".jpg", ".jpeg")):
                matched_file = os.path.join(folder_path, f)
                break

        if matched_file is None:
            raise FileNotFoundError(f"No image matching keyword '{keyword}' in {folder_path}")

        # Load image
        img = Image.open(matched_file).convert("RGB")
        img_np = np.array(img)

        # Optional preprocessing
        if edge_enhance:
            edges = cv2.Canny(img_np, 100, 200)
            img_np = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        if color_boost:
            enhancer = ImageEnhance.Color(Image.fromarray(img_np))
            img = enhancer.enhance(1.5)
        else:
            img = Image.fromarray(img_np)

        return (img,)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS.update({
    "MedicalReferenceSelectorNode": MedicalReferenceSelectorNode
})
