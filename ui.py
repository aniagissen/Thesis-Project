from typing import Optional, Dict, Any, Tuple
import streamlit as st

def shot_reference_image_controls(scene_idx: int, shot_idx: int) -> Tuple[bool, Optional["st.runtime.uploaded_file_manager.UploadedFile"], str]:
    use_ref = st.checkbox(
        "Use reference image for this shot",
        key=f"use_ref_{scene_idx}_{shot_idx}",
        value=False
    )
    file = st.file_uploader(
        "Reference image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"refimg_{scene_idx}_{shot_idx}",
        disabled=not use_ref
    )
    entity_hint = ""
    if use_ref:
        entity_hint = st.text_input(
            "This image represents (e.g., “neuron”, “microtubule”, “antibody”):",
            key=f"refimg_entity_{scene_idx}_{shot_idx}",
            placeholder="e.g., neuron",
        )
    if file is not None:
        st.image(file, caption="Reference", use_container_width=True)
    return use_ref, file, entity_hint


def shot_description_input(scene_idx: int, shot_idx: int, *, key_prefix: str = "") -> str:
    key = key_prefix or f"desc_{scene_idx}_{shot_idx}"
    return st.text_area("Short description", key=key, height=90, placeholder="e.g., A doctor walks through a hosptial corridor...")

def show_existing_prompt(existing: Optional[Dict[str, Any]], scene_idx: int, shot_idx: int) -> None:
    if existing and "generated_prompt" in existing:
        with st.expander("Existing generated prompt (read-only)", expanded=False):
            st.text_area(label="generated", value=existing.get("generated_prompt", ""), height=120, key=f"existing_{scene_idx}_{shot_idx}")

def shot_action_buttons(scene_idx: int, shot_idx: int) -> Tuple[bool, bool]:
    cols = st.columns([1, 1, 2])
    with cols[0]:
        gen_clicked = st.button("Generate", key=f"gen_{scene_idx}_{shot_idx}", use_container_width=True)
    with cols[1]:
        show_raw = st.button("Show input", key=f"showraw_{scene_idx}_{shot_idx}", use_container_width=True)
    return gen_clicked, show_raw

def show_generated_output(prompt_text: str, scene_idx: int, shot_idx: int) -> None:
    st.text_area(label="Generated prompt", value=prompt_text, height=140, key=f"out_{scene_idx}_{shot_idx}")
