import ollama

print("Please briefly and clearly describe what you imagine from act 1 of this scene")
act_1 = input("\nScene Description: ")

def generate_prompt(model="mistral", scene_description=""):
    prompt = f"""
You write the most specific and highly detailed prompt for Hunyuan model in ComfyUI. These videos are for scientific educational purposes.
They include: 
Subject - Define Your Star.
Must clearly specify the main focus of your video.
Examples:
"A young woman with flowing red hair"
"A sleek electric sports car"
"A majestic eagle in flight"
Pro tip: Be specific about details that matter, such as size, color, or distinctive features.

Scene - Set Your Stage.
Describe the environment where your action takes place.
Examples:
"In a neon-lit cyberpunk cityscape"
"Amidst a snow-covered forest at dawn"
"Inside a minimalist modern apartment"
Pro tip: Include background elements that enhance your story's context.

Motion - Bring Life to Your Video.
Detail how your subject moves or interacts.
Examples:
"Gracefully dancing through falling autumn leaves"
"Rapidly accelerating along a coastal highway"
"Smoothly transitioning between yoga poses"
Pro tip: Use vivid verbs to describe precise movements.

Camera Movement - Direct Your Shot.
Specify how the camera should capture the action.
Examples:
"Slow upward tilt revealing the cityscape"
"Smooth tracking shot following the subject"
"Dramatic circular pan around the scene"
Pro tip: Think like a cinematographer to create dynamic visual interest.

Atmosphere - Create the Right Mood.
Set the emotional tone of your video.
Examples:
"Mysterious and ethereal atmosphere"
"Energetic and vibrant mood"
"Calm and serene ambiance"
Pro tip: Consider how lighting and color contribute to the desired atmosphere.

Lighting - Perfect Your Illumination.
Define how light shapes your scene.
Examples:
"Soft, warm sunlight filtering through trees"
"Sharp, contrasting shadows from street lights"
"Diffused, ethereal glow from above"
Pro tip: Light can dramatically affect the mood and visual impact.

Shot Composition - Frame Your Vision.
Describe how elements should be arranged.
Examples:
"Close-up shot focusing on emotional expression"
"Wide landscape shot emphasizing scale"
"Low-angle shot creating dramatic perspective"
Pro tip: Consider the rule of thirds and visual balance in your composition.

Remember: Combining these elements effectively in your prompt will help HunyuanVideo generate exactly the video you envision. Start with these fundamentals and experiment to find what works best for your specific project.

The video is only 5-second, so creating with HunyuanVideo requires a well-structured prompt that paints a complete picture. Let's break down the essential requirements for crafting prompts that deliver exceptional results:

Optimal Prompt Structure 100-300 word paragraph, no subtitiles or new lines.

Your prompt should flow naturally while incorporating all vital elements in a coherent sequence:

Lead with Your Subject.
Begin with a clear, detailed description of your main focus.
Example: "A graceful ballet dancer in a flowing white dress"
Keep it specific but concise.

Establish Your Scene.
Immediately follow with the setting details.
Example: "Inside a grand, historic theater with ornate golden decorations"
Set the stage for your action to unfold.

Describe the Action.
Detail the motion sequence that will occur.
Example: "The dancer performs a fluid pirouette, her dress creating circular patterns in the air"
Focus on smooth, continuous movements that work well in a 5-second timeframe.

Specify Camera Work.
Include clear camera movement instructions.
Example: "Camera slowly circles around the dancer, starting at eye level and gradually ascending"
Ensure camera movements complement your subject's action.

Set the Mood and Lighting.
Close with atmospheric and lighting details.
Example: "Dramatic spotlight from above creates a soft glow, while shadows dance across the stage floor"
Use lighting to enhance the emotional impact.

Remember:
Keep your description focused on what can realistically happen in 5 seconds
Use clear, descriptive language without being overly complex
Maintain a logical flow from subject to scene to action
Ensure all elements work together to create a cohesive vision
By following this structure, you'll create prompts that HunyuanVideo can interpret accurately, resulting in professionally polished video outputs that match your creative vision.

Tips for Visualizing the Scene
Use Evocative Language
Choose Impact Words
Weak: "A bright day"
Strong: "Sun-drenched"
Weak: "Moving quickly"
Strong: "Surging," "Gliding"
Paint with Details
Weak: "A forest"
Strong: "Ancient redwoods towering into misty heights"
Weak: "A nice house"
Strong: "A Victorian mansion with intricate wrought-iron details"

Maintain Visual Consistency
Keep lighting conditions unified throughout the scene
Ensure color themes work together
Match mood elements across all scene components
Maintain realistic physics and movement patterns

Write Clear, Direct Descriptions
Focus on concrete visual elements
Describe exactly what should be seen
Structure descriptions from main subject to supporting details
Use precise spatial and directional terms

Use these tips and instructions to craft the perfect prompt for a 5 second video based on this brief: {scene_description}


"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Run and print result
if __name__ == "__main__":
    prompt = generate_prompt(model="mistral", scene_description=act_1)
    print(prompt)