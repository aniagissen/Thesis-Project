from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import streamlit as st

def act_description_input(scene_idx: int, act_idx: int, *, key_prefix: str = "") -> str:
    key = key_prefix or f"desc_{scene_idx}_{act_idx}"
    return st.text_area(
        "Short description", key=key, height=90, placeholder="e.g., A doctor walks through a neon-lit corridor..."
    )

def show_existing_prompt(existing: Optional[Dict[str, Any]], scene_idx: int, act_idx: int) -> None:
    if existing and "generated_prompt" in existing:
        with st.expander("Existing generated prompt (read-only)", expanded=False):
            st.text_area(
                label="generated",
                value=existing.get("generated_prompt", ""),
                height=120,
                key=f"existing_{scene_idx}_{act_idx}",
            )

def act_action_buttons(scene_idx: int, act_idx: int) -> Tuple[bool, bool]:
    cols = st.columns([1, 1, 2])
    with cols[0]:
        gen_clicked = st.button("Generate", key=f"gen_{scene_idx}_{act_idx}", use_container_width=True)
    with cols[1]:
        show_raw = st.button("Show input", key=f"showraw_{scene_idx}_{act_idx}", use_container_width=True)
    return gen_clicked, show_raw

def show_generated_output(prompt_text: str, scene_idx: int, act_idx: int) -> None:
    st.text_area(label="Generated prompt", value=prompt_text, height=140, key=f"out_{scene_idx}_{act_idx}")
