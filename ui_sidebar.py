from typing import Dict
import streamlit as st

# ---- Style presets ----
STYLE_PRESETS = {
    "2D Animation": (
        "Lighting is sterile and even. The animation style is flat vector illustration, clean geometric lines, "
        "minimal shading, stylised anatomy, soft gradients, glowing light effects, futuristic corporate science aesthetic, "
        "2D motion graphics style, medical explainer animation look, crisp edges, Adobe Illustrator and After Effects style."
    ),
    "3D Realistic": (
        "Physically based rendering under neutral surgical lighting, global illumination, realistic anatomy and materials, "
        "subsurface scattering, cinematic depth of field, subtle volumetrics, clinical environment."
    ),
    "3D Stylised": (
        "Stylised 3D medical aesthetic, simplified materials, smooth bevels, toon/NPR shading, gentle bloom, "
        "clean UI overlays, educational clarity, pastel palette."
    ),
    "Whiteboard / Line": (
        "Monochrome line-art on white background, hand-drawn strokes, schematic diagrams, sparse shading, "
        "clear labels and callouts, high contrast, clinical clarity."
    ),
    "Custom": ""
}


def render_sidebar(default_system: str) -> Dict:
    st.header("Settings")

    with st.expander("System prompt (advanced)", expanded=False):
        system_prompt = st.text_area(
            "System message sent to the model:",
            value=default_system,
            height=180
        )

    model_name = st.text_input(
        "Ollama model name",
        value="mistral",
        help="Any local Ollama model, e.g. 'mistral', 'llama3:instruct', 'qwen2.5'"
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    num_predict = st.slider(
        "Max tokens (num_predict)", 128, 1024, 512, 64,
        help="Upper bound for tokens predicted by the model."
    )
    overwrite_ok = st.checkbox("Overwrite existing prompts when regenerating", value=False)

    st.divider()
    scenes_count = st.number_input("How many scenes?", min_value=1, max_value=20, value=3, step=1)
    acts_per_scene = {}
    for i in range(1, int(scenes_count) + 1):
        acts_per_scene[i] = st.number_input(
            f"Acts in Scene {i}", min_value=1, max_value=20, value=3, step=1, key=f"acts_{i}"
        )

    st.divider()
    reset_clicked = st.button("Reset session (new master file)")

    # ---- Style suffix controls ----
    st.divider()
    st.subheader("Style suffix")
    enable_suffix = st.checkbox("Append style suffix to every prompt", value=True)

    style_choice = st.selectbox(
        "Style preset",
        options=list(STYLE_PRESETS.keys()),
        index=0,
        help="Pick a preset; you can still edit the text below."
    )

    # Persist user's edits unless the preset changes
    if "style_choice" not in st.session_state:
        st.session_state.style_choice = style_choice
    if "suffix_text" not in st.session_state:
        st.session_state.suffix_text = STYLE_PRESETS.get(style_choice, "")

    if style_choice != st.session_state.style_choice:
        st.session_state.suffix_text = STYLE_PRESETS.get(style_choice, st.session_state.get("suffix_text", ""))
        st.session_state.style_choice = style_choice

    suffix_text = st.text_area(
        "Suffix text",
        key="suffix_text",
        height=130,
        disabled=not enable_suffix
    )

    # ---- Length control ----
    st.divider()
    st.subheader("Length control")
    max_words = st.slider(
        "Max words (after suffix)",
        min_value=20, max_value=120, value=80, step=5,
        help="Final prompt will be trimmed to this budget while keeping the style suffix intact."
    )

    return {
        "system_prompt": system_prompt,
        "model_name": model_name,
        "temperature": temperature,
        "num_predict": num_predict,
        "overwrite_ok": overwrite_ok,
        "scenes_count": int(scenes_count),
        "acts_per_scene": {int(k): int(v) for k, v in acts_per_scene.items()},
        "reset_clicked": reset_clicked,
        "enable_suffix": enable_suffix,
        "suffix_text": suffix_text,
        "style_choice": style_choice,
        "max_words": max_words,
    }
