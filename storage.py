import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from config import ROOT_SCENES, ROOT_PROMPTS

def ensure_scene_dirs(scene_idx: int) -> Tuple[Path, Path]:
    sid = st.session_state.get('session_timestamp', 'session_default')
    scene_dir = ROOT_SCENES / sid / f"scene_{scene_idx}"
    prompt_dir = ROOT_PROMPTS / sid / f"scene_{scene_idx}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir, prompt_dir

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
