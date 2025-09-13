
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class VisualPlan:
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
    source: str 
    clip_id: str
    clip_uri: str
    duration: float
    similarity: float
    metadata: Dict[str, Any]


@dataclass
class SentenceItem:
    id: str
    text: str
    tts_path: Optional[str] = None
    duration_s: Optional[float] = None
    visual_plan: Optional[Dict[str, Any]] = None
    accepted_take: Optional[Dict[str, Any]] = None
