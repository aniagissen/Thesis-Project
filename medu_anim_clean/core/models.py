"""Dataclasses for core domain objects."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class VisualPlan:
    """Structured description of visuals for a narrated sentence."""
    shot_type: str
    primary_subject: str
    action: str
    visual_level: str
    color_style: str
    avoid: List[str]
    duration_s: float
    keywords: List[str]
    sensitivity: str


@dataclass
class Take:
    """A concrete video option for a sentence, typically from the library or generator."""
    source: str  # "library" | "comfy"
    clip_id: str
    clip_uri: str
    duration: float
    similarity: float
    metadata: Dict[str, Any]


@dataclass
class SentenceItem:
    """Narration sentence plus derived assets and decisions."""
    id: str
    text: str
    tts_path: Optional[str] = None
    duration_s: Optional[float] = None
    visual_plan: Optional[Dict[str, Any]] = None
    accepted_take: Optional[Dict[str, Any]] = None
