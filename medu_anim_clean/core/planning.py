"""Visual planning: map a sentence to a structured visual plan.
Replace with an LLM call that returns this schema.
"""
from __future__ import annotations

from typing import List

from .constants import MAX_SCENE_DURATION_S, MIN_SCENE_DURATION_S
from .models import VisualPlan


def plan_visual_for_sentence(sentence: str, sensitivity: str) -> VisualPlan:
    """Heuristic stub. Swap with an LLM-generated plan validated by schema."""
    s = sentence.lower()
    subject = "pancreatic beta cell" if "insulin" in s else "glucose uptake"
    keywords: List[str] = [w for w in ["insulin", "vesicle", "snare", "receptor", "glut4"] if w in s]

    # Approx target duration by words; clamp to sensible range
    words = max(1, len(sentence.split()))
    approx = words / 2.5
    duration = max(MIN_SCENE_DURATION_S, min(MAX_SCENE_DURATION_S, approx))

    return VisualPlan(
        shot_type="diagram" if sensitivity == "low" else "macro",
        primary_subject=subject,
        action="explain mechanism",
        visual_level="schematic" if sensitivity != "high" else "realistic",
        color_style="clinical",
        avoid=["gore"],
        duration_s=duration,
        keywords=keywords or ["mechanism"],
        sensitivity=sensitivity,
    )
