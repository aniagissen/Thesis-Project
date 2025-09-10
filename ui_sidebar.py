from typing import Dict
from pathlib import Path
import os
import streamlit as st

from config import ROOT_PROMPTS
from prompt_service import generate_style_suffix_from_image 

def render_sidebar(default_system: str) -> Dict:
    st.header("Settings")

    with st.expander("System prompt (advanced)", expanded=False):
        system_prompt = st.text_area("System message sent to the model:", value=default_system, height=180)

    # Primary text model
    model_name = st.text_input(
        "Ollama model name",
        value="llama3.2-vision:11b",
        help="Local model for prompt generation (text). Using a vision-capable model lets you use the same name for image style too."
    )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    num_predict = st.slider("Max tokens (num_predict)", 128, 1024, 512, 64, help="Upper bound for tokens predicted by the model.")
    overwrite_ok = st.checkbox("Overwrite existing prompts when regenerating", value=False)

    st.divider()

    scenes_count = st.number_input("How many scenes?", min_value=1, max_value=20, value=3, step=1)
    shots_per_scene = {}
    for i in range(1, int(scenes_count) + 1):
        shots_per_scene[i] = st.number_input(f"Shots in Scene {i}", min_value=1, max_value=20, value=3, step=1, key=f"shots_{i}")

    st.divider()
    reset_clicked = st.button("Reset session (new master file)")
    st.divider()

    st.subheader("Style")

    style_presets = {
        "2D Animation": ("Lighting is sterile and even. The animation style is flat vector illustration, clean geometric lines, minimal shading, "
                        "stylised anatomy, soft gradients, glowing light effects, futuristic corporate science aesthetic, 2D motion graphics style, "
                        "medical explainer animation look, crisp edges, Adobe Illustrator and After Effects style"),
        "3D Realistic": ("Photorealistic CGI render, physically based materials, naturalistic global illumination, shallow depth of field, "
                        "cinematic color grade, ray-traced reflections, detailed textures"),
        "None (start blank)": ""
    }
    style_choice = st.selectbox("Style preset", list(style_presets.keys()), index=0)

    if "suffix_text" not in st.session_state:
        st.session_state["suffix_text"] = style_presets[style_choice]

    apply_preset = st.button("Apply preset to suffix")
    if apply_preset:
        st.session_state["suffix_text"] = style_presets[style_choice]
        st.toast("Applied preset to suffix.")

    st.markdown("Or build from a reference image")
    vision_model = st.text_input(
        "Vision model (Ollama)",
        value=model_name,
        help="Must support images, e.g., 'llama3.2-vision:11b'."
    )
    use_img_style = st.checkbox("Use image to generate suffix", value=False)
    img_file = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg", "webp"], disabled=not use_img_style)

    c_img1, c_img2 = st.columns([1,1])
    with c_img1:
        analyse_clicked = st.button("Analyse image → create suffix", disabled=not use_img_style or (img_file is None))
    with c_img2:
        if img_file is not None:
            st.image(img_file, caption="Reference", use_container_width=True)

    if analyse_clicked and img_file is not None:
        try:
            sid = st.session_state.get("session_timestamp", "session_default")
            save_dir = ROOT_PROMPTS / sid / "style_refs"
            save_dir.mkdir(parents=True, exist_ok=True)
            ext = Path(img_file.name).suffix or ".png"
            img_path = save_dir / f"style_ref{ext}"
            with open(img_path, "wb") as f:
                f.write(img_file.getbuffer())

            suffix = generate_style_suffix_from_image(vision_model, str(img_path))
            st.session_state["suffix_text"] = suffix
            st.success("Generated style suffix from the image and applied below.")
        except Exception as e:
            st.error(f"Image analysis failed: {e}")

    st.subheader("Style suffix")
    enable_suffix = st.checkbox("Append style suffix to every prompt", value=True)
    suffix_text = st.text_area("Suffix text", key="suffix_text", height=130, disabled=not enable_suffix)


    st.divider()

    st.subheader("Render parameters in ComfyUI")
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
        "shots_per_scene": {int(k): int(v) for k, v in shots_per_scene.items()},
        "reset_clicked": reset_clicked,

        "enable_suffix": enable_suffix,
        "suffix_text": suffix_text,
        "style_preset": style_choice,

        "vision_model": vision_model,
        "use_img_style": use_img_style,

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
