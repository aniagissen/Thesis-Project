import numpy as np
import cv2

class MedicalParamOptimizerNode:
    """Optimizes parameters based on input image complexity."""
    CATEGORY = "KlaraMedical"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "animation_length_sec": ("INT", {"default": 5, "min": 1, "max": 30}),
                "resolution": (["420p", "720p", "1080p"], {"default": "720p"}),
                "style_mode": (["diagram", "microscopy", "mixed"], {"default": "diagram"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("cfg", "steps", "flow_shift")
    FUNCTION = "optimize"

    def optimize(self, image, animation_length_sec, resolution, style_mode):
        img = np.array(image).astype(np.uint8)

        edges = cv2.Canny(img, 100, 200)
        edge_density = edges.sum() / (edges.size * 255)
        color_variance = np.var(img) / 255.0

        cfg, steps, flow_shift = 9.0, 10, 12
        if edge_density > 0.05:
            steps += 2; cfg += 1.0
        if color_variance > 0.1:
            cfg -= 0.5

        if style_mode == "microscopy":
            cfg += 0.5; flow_shift -= 2
        elif style_mode == "diagram":
            steps += 1; flow_shift = 12

        if animation_length_sec > 10:
            flow_shift -= 2

        cfg = float(np.clip(cfg, 7.0, 12.0))
        steps = int(np.clip(steps, 8, 20))
        flow_shift = int(np.clip(flow_shift, 6, 15))

        return (cfg, steps, flow_shift)

# ✅ Register this node
NODE_CLASS_MAPPINGS = {
    "MedicalParamOptimizerNode": MedicalParamOptimizerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MedicalParamOptimizerNode": "Medical Parameter Optimizer",
}
