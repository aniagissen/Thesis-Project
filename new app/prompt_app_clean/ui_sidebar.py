from typing import Dict
import streamlit as st

def render_sidebar(default_system: str) -> Dict:
    st.header("Settings")

    with st.expander("System prompt (advanced)", expanded=False):
        system_prompt = st.text_area("System message sent to the model:", value=default_system, height=180)

    model_name = st.text_input("Ollama model name", value="mistral", help="Any local Ollama model, e.g. 'mistral', 'llama3:instruct', 'qwen2.5'")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    num_predict = st.slider("Max tokens (num_predict)", 128, 1024, 512, 64, help="Upper bound for tokens predicted by the model.")
    overwrite_ok = st.checkbox("Overwrite existing prompts when regenerating", value=False)

    st.divider()
    scenes_count = st.number_input("How many scenes?", min_value=1, max_value=20, value=3, step=1)
    acts_per_scene = {}
    for i in range(1, int(scenes_count) + 1):
        acts_per_scene[i] = st.number_input(f"Acts in Scene {i}", min_value=1, max_value=20, value=3, step=1, key=f"acts_{i}")

    st.divider()
    reset_clicked = st.button("Reset session (new master file)")

    st.divider()
    st.subheader("Style suffix")
    enable_suffix = st.checkbox("Append style suffix to every prompt", value=True)
    default_suffix = (
        "The animation style is flat vector illustration, clean geometric shapes, minimal shading, "
        "soft gradients, cyan–turquoise palette, 2D motion graphics, medical explainer look, crisp edges."
    )
    suffix_text = st.text_area("Suffix text", value=default_suffix, height=130, disabled=not enable_suffix)

    st.divider()
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
        "max_words": max_words,
    }
