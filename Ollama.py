import ollama

def generate_prompt(model="llama2:7b-chat", scene_description=""):
    prompt = f"""
You write the most specific and highly detailed prompt for Hunyuan model in ComfyUI. These videos are for scientific educational purposes.
They include: 
Subject - Define Your Star.
Must clearly specify the main focus of your video.
Examples:
"A young woman with flowing red hair"
"A sleek electric sports car"
"A majestic eagle in flight"

Scene - Set Your Stage.
Describe the environment where your action takes place.
Examples:
"In a neon-lit cyberpunk cityscape"
"Amidst a snow-covered forest at dawn"
"Inside a minimalist modern apartment"

Motion - Bring Life to Your Video.
Detail how your subject moves or interacts.
Examples:
"Gracefully dancing through falling autumn leaves"
"Rapidly accelerating along a coastal highway"
"Smoothly transitioning between yoga poses"

Camera Movement - Direct Your Shot.
Specify how the camera should capture the action.
Examples:
"Slow upward tilt revealing the cityscape"
"Smooth tracking shot following the subject"
"Dramatic circular pan around the scene"

Atmosphere - Create the Right Mood.
Set the emotional tone of your video.
Examples:
"Mysterious and ethereal atmosphere"
"Energetic and vibrant mood"
"Calm and serene ambiance"

Lighting - Perfect Your Illumination.
Define how light shapes your scene.
Examples:
"Soft, warm sunlight filtering through trees"
"Sharp, contrasting shadows from street lights"
"Diffused, ethereal glow from above"

Shot Composition - Frame Your Vision.
Describe how elements should be arranged.
Examples:
"Close-up shot focusing on emotional expression"
"Wide landscape shot emphasizing scale"
"Low-angle shot creating dramatic perspective"

Remember: The video is only 5 seconds, so create a well-structured prompt of about 100-300 words, no subtitles or newlines. Describe subject, scene, action, camera, mood and lighting in one flowing paragraph. Keep it visually consistent and use strong evocative language. 

Use these instructions to craft the perfect prompt for this brief: {scene_description}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


# Only runs if you execute `python Ollama.py` directly in the terminal
if __name__ == "__main__":
    print("Please briefly and clearly describe what you imagine from act 1 of this scene")
    act_1 = input("\nScene Description: ")
    final_prompt = generate_prompt(scene_description=act_1)
    print(final_prompt)
