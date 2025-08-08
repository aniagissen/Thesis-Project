"""
@title: KlaraMedical Pack
@description: Custom nodes for medical animation generation.
"""

import importlib
import logging

logging.info("### Loading: KlaraMedical Custom Nodes (relative import - final)")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in ["MedicalReference", "Parameters", "Styler"]:
    try:
        mod = importlib.import_module(f".{module_name}", package=__name__)
        if hasattr(mod, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        logging.error(f"KlaraMedical failed to import {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
