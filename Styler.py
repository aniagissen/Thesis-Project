import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from comfy.nodes import NODE_CLASS_MAPPINGS

class MedicalStyleBankNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "./medical_style_bank"}),
                "mode": (["random", "cycle", "average"],),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 20}),
                "color_boost": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_styles"

    # Track last index for cycling
    last_index = 0

    def load_styles(self, folder_path, mode, max_images, color_boost):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Style bank folder not found: {folder_path}")

        # Get all images
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            raise ValueError("No images found in style bank folder.")

        # Select images based on mode
        if mode == "random":
            selected = random.sample(files, min(max_images, len(files)))
        elif mode == "cycle":
            start = MedicalStyleBankNode.last_index
            end = start + max_images
            selected = files[start:end] if end <= len(files) else files[start:] + files[:end - len(files)]
            MedicalStyleBankNode.last_index = (end) % len(files)
        else:  # average
            selected = files[:min(max_images, len(files))]

        # Load images
        imgs = [Image.open(p).convert("RGB") for p in selected]

        # Apply color boost if needed
        if color_boost:
            imgs = [ImageEnhance.Color(im).enhance(1.5) for im in imgs]

        # If average mode, blend all images into one
        if mode == "average":
            arrs = [np.array(im.resize(imgs[0].size), dtype=np.float32) for im in imgs]
            avg = np.mean(arrs, axis=0).astype(np.uint8)
            final_img = Image.fromarray(avg)
            return (final_img,)

        # Otherwise, return first selected image (ComfyUI IMAGE type expects a single output here)
        return (imgs[0],)

# Register node
NODE_CLASS_MAPPINGS.update({
    "MedicalStyleBankNode": MedicalStyleBankNode
})
