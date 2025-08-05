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
                "duration_seconds": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0}),
                "resolution": (["420p", "720p", "1080p"],),
                "style_mode": (["diagram", "microscopy", "mixed"],),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "manual_steps": ("INT", {"default": 0, "min": 0, "max": 40}),  # NEW OVERRIDE
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cfg", "steps", "flow_shift", "width", "height", "frame_count")
    FUNCTION = "optimize"

    def optimize(self, image, duration_seconds, resolution, style_mode, fps, manual_steps):
        img = np.array(image.convert("RGB"))

        # --- Analyze the input image ---
        edges = cv2.Canny(img, 100, 200)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)
        color_variance = np.var(img) / 255.0

        # --- Base defaults ---
        cfg = 9.0
        steps = 10
        flow_shift = 12

        # --- Adjust automatically if manual override is not used ---
        if manual_steps == 0:
            if edge_density > 0.05:
                steps += 2
                cfg += 1.0
            if color_variance > 0.1:
                cfg -= 0.5

            if style_mode == "microscopy":
                cfg += 0.5
                flow_shift -= 2
            elif style_mode == "diagram":
                steps += 1
                flow_shift = 12
        else:
            steps = manual_steps  # Use manual override

        # --- Resolution mapping ---
        res_map = {
            "420p": (640, 420),
            "720p": (1280, 720),
            "1080p": (1920, 1080)
        }
        width, height = res_map.get(resolution, (1280, 720))

        # --- Frame count ---
        frame_count = int(duration_seconds * fps)
        if frame_count > 100:
            flow_shift -= 2

        # --- Clamp values ---
        cfg = float(np.clip(cfg, 7.0, 12.0))
        steps = int(np.clip(steps, 1, 40))
        flow_shift = int(np.clip(flow_shift, 6, 15))

        return (cfg, steps, flow_shift, width, height, frame_count)

NODE_CLASS_MAPPINGS.update({
    "MedicalParamOptimizerNode": MedicalParamOptimizerNode
})
