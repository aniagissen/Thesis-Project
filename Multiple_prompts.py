import streamlit as st
import ollama

st.title("biomations.ai")

initial_prompt = st.text_input("Write your vision of this act")

def generate_prompt(model="mistral", initial_prompt=initial_prompt):
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

    The output must be in one continuous cinematic style paragraph with no line breaks, subtitles, or lists. It must use evocative descriptive language and maintain a consistent cartoon style with a clear medical or scientific context regardless of the input.

    Use these instructions to craft the perfect video prompt based on this brief: {initial_prompt}.
    """

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
