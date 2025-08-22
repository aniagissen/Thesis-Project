from typing import Dict
import streamlit as st

# Build sidebar with set and editable settings
def render_sidebar(default_system: str) -> Dict:
    st.header("Settings")

    with st.expander("System prompt (advanced)", expanded=False):
        system_prompt = st.text_area("System message sent to the model:", value=default_system, height=180)

    # choice of ollama model - set to mistral
    model_name = st.text_input("Ollama model name", value="mistral", help="Any local Ollama model, e.g. 'mistral', 'llama3:instruct', 'qwen2.5'")
    # Editable fetures to control creativity and length
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    num_predict = st.slider("Max tokens (num_predict)", 128, 1024, 512, 64, help="Upper bound for tokens predicted by the model.")
    # Allows overwriting previous prompts
    overwrite_ok = st.checkbox("Overwrite existing prompts when regenerating", value=False)

    st.divider()

    # Counts inputted scenes and acts
    scenes_count = st.number_input("How many scenes?", min_value=1, max_value=20, value=3, step=1)
    acts_per_scene = {}
    for i in range(1, int(scenes_count) + 1):
        acts_per_scene[i] = st.number_input(f"Acts in Scene {i}", min_value=1, max_value=20, value=3, step=1, key=f"acts_{i}")

    st.divider()

    # Clears state and creates new master
    reset_clicked = st.button("Reset session (new master file)")

    st.divider()

    # Dropdown menu of sytle suffixes for the user to choose
    st.subheader("Style")
    style_presets = {
        "2D Animation": ("Lighting is sterile and even. The animation style is flat vector illustration, clean geometric lines, minimal shading, "
                         "stylised anatomy, soft gradients, glowing light effects, futuristic corporate science aesthetic, 2D motion graphics style, "
                         "medical explainer animation look, crisp edges, Adobe Illustrator and After Effects style"),
        "3D Realistic": ("Photorealistic CGI render, physically based materials, naturalistic global illumination, shallow depth of field, "
                         "cinematic color grade, ray-traced reflections, detailed textures"),
        "Documentary Realism": ("Natural handheld camerawork, practical lighting, neutral color grade, true-to-life textures, ambient room tone, "
                                "unobtrusive visual style"),
        "Sketch/Storyboard": ("Monochrome pencil sketch style, rough crosshatching, loose gesture lines, storyboard frame aesthetic, minimal shading"),
    }
    style_choice = st.selectbox("Style preset", list(style_presets.keys()), index=0)

    # keep suffix in sync with the preset unless user changes it after
    if "style_choice" not in st.session_state:
        st.session_state["style_choice"] = style_choice
    if "suffix_text" not in st.session_state:
        st.session_state["suffix_text"] = style_presets[style_choice]
    if st.session_state["style_choice"] != style_choice:
        st.session_state["style_choice"] = style_choice
        st.session_state["suffix_text"] = style_presets[style_choice]
        st.rerun()

    st.subheader("Style suffix")
    enable_suffix = st.checkbox("Append style suffix to every prompt", value=True)
    # Default of the settings I used
    default_suffix = (
        "The animation style is flat vector illustration, clean geometric shapes, minimal shading, "
        "soft gradients, cyan–turquoise palette, 2D motion graphics, medical explainer look, crisp edges."
    )
    suffix_text = st.text_area("Suffix text", key="suffix_text", height=130, disabled=not enable_suffix)

    st.divider()

    # Allows user to overwrite the workflow parameters, if off its uses the original, if on it overwrites
    st.subheader("Render parameters (ComfyUI)")
    override_params = st.checkbox("Override workflow parameters for this batch", value=False, help="If off, your workflow JSON values are used.")
    col1, col2 = st.columns(2)
    with col1:
        steps = st.number_input("Steps", min_value=1, max_value=128, value=12, step=1, disabled=not override_params)
        guidance = st.number_input("Guidance", min_value=0.0, max_value=50.0, value=8.0, step=0.5, disabled=not override_params)
        width = st.number_input("Width", min_value=64, max_value=2048, value=448, step=16, disabled=not override_params)
        height = st.number_input("Height", min_value=64, max_value=2048, value=240, step=16, disabled=not override_params)
    with col2:
        length = st.number_input("Length (frames)", min_value=1, max_value=4096, value=73, step=1, disabled=not override_params)
        fps = st.number_input("Frame rate (fps)", min_value=1, max_value=120, value=24, step=1, disabled=not override_params)
        format_str = st.text_input("Video format", value="video/nvenc_h264-mp4", disabled=not override_params)
        seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2147483647, value=0, step=1, disabled=not override_params)

    st.subheader("Length control")
    max_words = st.slider("Max words (after suffix)", min_value=20, max_value=120, value=80, step=5,
                          help="Final prompt will be trimmed to this budget while keeping the style suffix intact.")

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
        "style_preset": style_choice,
        "max_words": max_words,
        "override_params": override_params,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "length": length,
        "fps": fps,
        "format": format_str,
        "seed": seed,
    }
