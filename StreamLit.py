import streamlit as st
import os
from Ollama import generate_prompt 

st.title("biomations.ai")

PROMPT_ROOT = "Prompts"
SCENE_ROOT = "Scenes"

# User input for scenes
num_scenes = st.number_input("How many scenes?", min_value=1, step=1)

for scene_num in range(1, num_scenes + 1):
    st.subheader(f"Scene {scene_num}")

    # Create folders
    scene_path = os.path.join(SCENE_ROOT, f"scene_{scene_num}")
    prompt_scene_path = os.path.join(PROMPT_ROOT, f"scene_{scene_num}")
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(prompt_scene_path, exist_ok=True)

    # Input number of acts
    num_acts = st.number_input(
        f"How many acts in Scene {scene_num}?", 
        min_value=1, step=1, key=f"acts_scene_{scene_num}")

    for act_num in range(1, num_acts + 1):
        description = st.text_area(
            f"Act {act_num} Description",
            key=f"scene{scene_num}_act{act_num}"
        )

        if description:
            # Save the description
            act_path = os.path.join(scene_path, f"act_{act_num}.txt")
            with open(act_path, "w") as f:
                f.write(description)

            enhanced_prompt = generate_prompt(scene_description=description)

            # Save to prompts folder
            prompt_file = os.path.join(prompt_scene_path, f"prompt_act_{act_num}.json.txt")
            with open(prompt_file, "w") as f:
                f.write(enhanced_prompt)

            st.success(f"Saved Scene {scene_num} - Act {act_num} to:")
            st.text(f"{prompt_file}")
            st.caption("Generated prompt preview:")
            st.code(enhanced_prompt[:500] + "...", language="text")
