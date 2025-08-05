import numpy as np
import cv2
from PIL import Image
from comfy.nodes import NODE_CLASS_MAPPINGS

class MedicalParamOptimizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "animation_length": ("INT", {"default": 72, "min": 10, "max": 200}),
                "style_mode": (["diagram", "microscopy", "mixed"],),
            }
        }

    # Outputs: CFG (FLOAT), Steps (INT), Flow Shift (INT)
    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("cfg", "steps", "flow_shift")
    FUNCTION = "optimize"

    def optimize(self, image, animation_length, style_mode):
        img = np.array(image.convert("RGB"))
        
        # --- 1. Analyze Edge Density ---
        edges = cv2.Canny(img, 100, 200)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        # --- 2. Analyze Color Variance ---
        color_variance = np.var(img) / 255.0

        # --- 3. Heuristics for CFG, Steps, Flow Shift ---
        # Base defaults
        cfg = 9.0
        steps = 10
        flow_shift = 12

        # Adjust based on image complexity
        if edge_density > 0.05:     # lots of fine details
            steps += 2              # more sampling for detail
            cfg += 1.0              # stronger adherence
        if color_variance > 0.1:    # highly colorful/complex
            cfg -= 0.5              # avoid oversaturation artifacts

        # Adjust based on style_mode
        if style_mode == "microscopy":
            cfg += 0.5              # push realism
            flow_shift -= 2         # smoother motion
        elif style_mode == "diagram":
            steps += 1              # ensure clear edges
            flow_shift = 12         # keep motion crisp

        # Adjust based on animation length
        if animation_length > 100:
            flow_shift -= 2         # smoother motion for long sequences

        # Clamp values to safe ranges
        cfg = float(np.clip(cfg, 7.0, 12.0))
        steps = int(np.clip(steps, 8, 20))
        flow_shift = int(np.clip(flow_shift, 6, 15))

        return (cfg, steps, flow_shift)

# Register with ComfyUI
NODE_CLASS_MAPPINGS.update({
    "MedicalParamOptimizerNode": MedicalParamOptimizerNode
})
