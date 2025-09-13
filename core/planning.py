import os, requests
from pydantic import BaseModel, Field, ValidationError
from typing import List
from .models import VisualPlan

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

class VisualPlanSchema(BaseModel):
    shot_type: str                  # e.g., "diagram" | "macro" | "explainer"
    primary_subject: str            # e.g., "insulin receptor"
    action: str                     # e.g., "binding and signaling"
    visual_level: str               # "schematic" | "realistic" | "iconographic"
    color_style: str                # "clinical" | "cool" | "warm"
    avoid: List[str] = Field(default_factory=list)
    duration_s: float             
    keywords: List[str] = Field(default_factory=list)
    sensitivity: str               

def plan_visual_for_sentence(sentence: str, sensitivity: str = "medium") -> VisualPlan:
    prompt = (
        "Return ONLY JSON with fields: shot_type, primary_subject, action, visual_level, color_style, "
        "avoid (array), duration_s (float 4-8), keywords (array), sensitivity.\n"
        "Keep values short and unambiguous. duration_s between 4 and 8.\n"
        f"Sentence: {sentence}\n"
        f"Target sensitivity: {sensitivity}\n"
    )
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json().get("response", "{}")
        parsed = VisualPlanSchema.model_validate_json(data)
    except Exception:
        parsed = VisualPlanSchema(
            shot_type="diagram" if sensitivity == "low" else "explainer",
            primary_subject="medical subject",
            action="explain mechanism",
            visual_level="schematic",
            color_style="clinical",
            avoid=["clutter"],
            duration_s=6.0,
            keywords=["mechanism"],
            sensitivity=sensitivity,
        )
    d = max(4.0, min(8.0, float(parsed.duration_s)))
    parsed.duration_s = d
    return VisualPlan(**parsed.model_dump())
