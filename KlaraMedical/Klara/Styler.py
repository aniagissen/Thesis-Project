import numpy as np
from PIL import Image

class MedicalStylerNode:
    """Applies optional style blending to an input image."""
    CATEGORY = "KlaraMedical"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "style_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_style"

    def apply_style(self, input_image, style_strength):
        img = np.array(input_image).astype(np.float32)
        styled = img * (1 - style_strength) + 255 * style_strength
        styled = np.clip(styled, 0, 255).astype(np.uint8)
        return (styled,)

# ✅ Register this node
NODE_CLASS_MAPPINGS = {
    "MedicalStylerNode": MedicalStylerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MedicalStylerNode": "Medical Styler",
}
