import os
import numpy as np
from PIL import Image

class MedicalReferenceNode:
    """Loads a folder of medical images and selects one or blends them."""
    CATEGORY = "KlaraMedical"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "/home/ania/ComfyUI/input/medical_reference_bank"}),
                "blend_all": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_reference"

    def load_reference(self, folder_path, blend_all):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            raise ValueError(f"No images found in {folder_path}")

        images = [np.array(Image.open(os.path.join(folder_path, f)).convert("RGB")) for f in files]

        if blend_all:
            stacked = np.stack(images, axis=0)
            avg_img = np.mean(stacked, axis=0).astype(np.uint8)
            return (avg_img,)
        else:
            return (images[0],)  # default: return the first image

# ✅ Register this node
NODE_CLASS_MAPPINGS = {
    "MedicalReferenceNode": MedicalReferenceNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MedicalReferenceNode": "Medical Reference Loader",
}
