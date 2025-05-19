import ollama

def generate_prompt(model="mistral"):
    prompt = """
You are a highly successful prompt engineer who works specifically with ComfyUI’s Hunyuan model to generate high-quality animated scientific explainer videos.

SCENE SCRIPT:

Scene 1 – What is Parkinson’s?  
- Animation of a healthy brain slowly showing signs of deterioration.  
- Zoom into basal ganglia as it dims.  
- Dopamine pathways flicker and shrink.  
- Neurons slow and disconnect.  
- Cut to subtle hand tremors beginning.

Scene 2 – Introducing the Test  
- Blood is drawn and enters a futuristic AI lab.  
- Zoom into the plasma, revealing glowing 8 protein biomarkers:  
  GRN, MASP2, HSPA5, PTGDS, ICAM1, C3, DKK3, SERPING1  
- AI scans the proteins with pulses of light, detecting subtle shifts.  
- Show a bar chart comparison: “At-risk” profile vs healthy baseline.

Scene 3 – Understanding the Biomarkers  
- C3: Red alert flares, highlighting inflammation inside the body.  
- GRN: A protective barrier breaking or fading — representing loss of neuroprotection.  
- DKK3: A Wnt-signaling bridge collapsing — vital brain repair mechanisms disrupted.  
- SERPING1: Alarms sounding, linked to protein build-up and stress responses.

Scene 4 – Prediction  
- Timeline animation:  
  “Test Taken” → “7 Years Later” → Early motor symptoms  
- High-risk detected result.  
- Intervention: patients join a clinical trial, begin therapy earlier.  
- Show individuals benefiting from early intervention.

TASK:  
Using the provided scene descriptions, generate a **visually focused prompt** for each scene for use in ComfyUI’s Hunyuan model.  
Each scene should:  
- Be 1–2 sentences  
- Describe clearly what should be seen in the animation  
- Focus on movement, color, elements, and transitions  
- Be designed to last approximately 7 seconds  

Please return your response in the format:  
Scene 1: [Prompt text]  
Scene 2: [Prompt text]  
Scene 3: [Prompt text]  
Scene 4: [Prompt text]  
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Run and print result
if __name__ == "__main__":
    prompts = generate_prompt(model="mistral")
    print(prompts)