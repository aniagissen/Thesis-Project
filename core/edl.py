from __future__ import annotations

from typing import Dict, List, Tuple

from .models import SentenceItem


def build_edl(sentences_by_scene: Dict[str, List[SentenceItem]]) -> Dict[str, List[Dict]]:
    edl_v: List[Dict] = []
    edl_a: List[Dict] = []
    t_cursor = 0.0

    ordered_keys = sorted(sentences_by_scene.keys(), key=lambda k: int(k.split("-")[-1]))
    for key in ordered_keys:
        for item in sentences_by_scene[key]:
            if not item.accepted_take or not item.duration_s:
                continue
            v = item.accepted_take
            edl_v.append({
                "scene_key": key,
                "sentence_id": item.id,
                "clip_id": v["clip_id"],
                "uri": v["clip_uri"],
                "in": 0.0,
                "out": v["duration"],
                "start": t_cursor,
            })
            edl_a.append({
                "sentence_id": item.id,
                "uri": item.tts_path,
                "start": t_cursor,
            })
            t_cursor += max(float(v["duration"]), float(item.duration_s))
    return {"video": edl_v, "audio": edl_a}
