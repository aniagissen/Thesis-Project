import streamlit as st
import os
import datetime
import ollama

st.title("biomations.ai")

PROMPT_ROOT = "Prompts"
SCENE_ROOT = "Scenes"

def generate_prompt(model="mistral", scene_description=""):
    prompt = f"""
    You are an expert prompt engineer for ComfyUI using the Hunyuan video model specialised in generating cartoon style videos with a strong medical or scientific theme.

    Produce a single, highly detailed paragraph, approximately 100 to 300 words that integrates all of the following elements seamlessly:

    The subject - Clearly define the main focus of the scene ensuring it is directly tied to a medical or scientific concept
    The scene or Environment - Describe the setting vividly with illustrative cartoon details that convey a scientific or clinical context such as labs anatomical environments cellular landscapes or futuristic medical facilities
    The motion or Action - Detail how the subject moves or interacts in a dynamic way appropriate for a short animated clip
    The camera Movement - Specify how the camera captures this action with pans tilts or tracking shots to enhance the educational focus
    The atmosphere and Mood - Set an emotionally engaging tone that underscores the importance of medical discovery or scientific wonder
    The lighting - Explain how light shapes the mood and highlights key anatomical or clinical features
    The Shot Composition - Ensure visual balance and interest describing angles and framing that emphasize the scientific theme

    Here is an example prompt: “A lone camel walks across vast, golden sand dunes under a clear blue sky. The camera begins with a wide shot, capturing the endless dunes, then slowly zooms in on the camel as it moves gracefully. The soft wind blows sand into gentle waves, adding motion to the scene. The atmosphere is serene and timeless, emphasizing the beauty of the desert. Bright sunlight casts sharp shadows on the dunes, creating striking contrasts. The shot transitions from a wide view of the desert to a medium shot of the camel, framed against the rolling sands.”

    The output must be in one continuous cinematic style paragraph with no line breaks, subtitles, or lists. It must use evocative descriptive language and maintain a consistent cartoon style with a clear medical or scientific context regardless of the input.

    Use these instructions to craft the perfect video prompt based on this brief {scene_description}
    """

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


# Generate filename based on timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
MASTER_PROMPT_FILE = f"all_prompts_{timestamp}.txt"

st.info(f"Master file for this session: `{MASTER_PROMPT_FILE}`")

num_scenes = st.number_input("How many scenes?", min_value=1, step=1)

for scene_num in range(1, num_scenes + 1):
    st.subheader(f"Scene {scene_num}")

    # Create folders
    scene_path = os.path.join(SCENE_ROOT, f"scene_{scene_num}")
    prompt_scene_path = os.path.join(PROMPT_ROOT, f"scene_{scene_num}")
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(prompt_scene_path, exist_ok=True)

    num_acts = st.number_input(
        f"How many acts in Scene {scene_num}?",
        min_value=1, step=1, key=f"acts_scene_{scene_num}")

    for act_num in range(1, num_acts + 1):
        description = st.text_area(
            f"Act {act_num} Description",
            key=f"scene{scene_num}_act{act_num}"
        )

        if description:
            # Save short description
            act_path = os.path.join(scene_path, f"act_{act_num}.txt")
            with open(act_path, "w") as f:
                f.write(description)

            # Generate prompt
            enhanced_prompt = generate_prompt(scene_description=description)

            #Save to scene-specific prompt file
            prompt_file = os.path.join(prompt_scene_path, f"prompt_act_{act_num}.json.txt")
            with open(prompt_file, "w") as f:
                f.write(enhanced_prompt)

            #append to unique master file for Inspire
            with open(MASTER_PROMPT_FILE, "a") as f:
                f.write(enhanced_prompt.replace("\n", " ") + "\n")

# download button
if os.path.exists(MASTER_PROMPT_FILE):
    with open(MASTER_PROMPT_FILE, "rb") as f:
        st.download_button(
            label="Download prompts file",
            data=f,
            file_name=MASTER_PROMPT_FILE,
            mime="text/plain"
        )
