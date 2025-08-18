def generate_prompt(model: str, system_msg: str, act_desc: str, *, temperature: float, num_predict: int) -> str:
    try:
        import ollama
    except Exception as e:
        raise RuntimeError("Unable to import 'ollama'. Ensure Ollama is installed and the Python package is available.") from e

    messages = [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": (
            "Based on the following act description, write one prompt for a text-to-video model.\n\n"
            f"Act description: \"{act_desc.strip()}\"\n\n"
            "Constraints:\n"
            "- Output exactly one sentence (max ~80 words).\n"
            "- Only describe the visuals wanted, clearly and with no fluff.\n"
            "- Include 1–2 adjectives, a clear action, and specific environment cues.\n"
            "- Prefer concrete nouns over abstractions.\n"
            "- No scene numbers, bullet points, or quotes."
        )},
    ]
    try:
        resp = ollama.chat(model=model, messages=messages, options={"temperature": float(temperature), "num_predict": int(num_predict)})
    except Exception as e:
        raise RuntimeError("Failed to contact Ollama. Is the Ollama server running and is the model pulled?") from e

    try:
        content = resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError("Unexpected response shape from Ollama.") from e
    if not content:
        raise RuntimeError("Model returned empty content.")
    return content
