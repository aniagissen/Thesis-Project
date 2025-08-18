from datetime import datetime
from pathlib import Path
import streamlit as st

ROOT_SCENES = Path("Scenes")
ROOT_PROMPTS = Path("Prompts")

def ensure_roots() -> None:
    ROOT_SCENES.mkdir(parents=True, exist_ok=True)
    ROOT_PROMPTS.mkdir(parents=True, exist_ok=True)

def init_session_state() -> Path:
    if "session_timestamp" not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if "master_file" not in st.session_state:
        st.session_state.master_file = ROOT_PROMPTS / f"all_prompts_{st.session_state.session_timestamp}.jsonl"
    return st.session_state.master_file
