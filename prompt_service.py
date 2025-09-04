from text_utils import enforce_word_budget


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
            "- Include 1 adjective, a clear action, and specific environment cues.\n"
            "- Prefer concrete nouns over abstractions.\n"
            "- No scene numbers, bullet points, or quotes."
        )},
    ]
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": float(temperature), "num_predict": int(num_predict)}
        )
    except Exception as e:
        raise RuntimeError("Failed to contact Ollama. Is the Ollama server running and is the model pulled?") from e

    try:
        content = resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError("Unexpected response shape from Ollama.") from e
    if not content:
        raise RuntimeError("Model returned empty content.")
    return content


def _assert_vision_capable_ollama(model: str, img_path: str) -> None:
    """
    Minimal probe: call chat once with an image attached. Text-only models will error.
    """
    try:
        import ollama, os
    except Exception as e:
        raise RuntimeError("Unable to import 'ollama'. Ensure Ollama is installed and the Python package is available.") from e
    if not os.path.exists(img_path):
        raise RuntimeError(f"Image not found: {img_path}")

    try:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "Describe the image style.", "images": [img_path]}],
            options={"num_predict": 1}
        )
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("image", "vision", "unsupported")):
            raise RuntimeError(
                f"Model '{model}' appears to be text-only and cannot accept images. "
                "Choose a vision model like 'llama3.2-vision:11b'."
            ) from e
        raise


def generate_style_suffix_from_image(
    model: str,
    image_path: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 256
) -> str:
    """
        "You are a visual style analyst. Given an image, output a single, comma-separated sentence "
        "describing lighting, illustration/rendering style, line/shape characteristics, shading, anatomy stylization, "
        "textures/gradients, color treatment, and any relevant production tools. "
        "Write it as a style suffix to append to a video prompt. No preamble, no quotes."
        "(comma-separated descriptors, 1–2 short sentences max)."
    """
    try:
        import ollama, os, re
    except Exception as e:
        raise RuntimeError("Unable to import 'ollama'. Ensure Ollama is installed and the Python package is available.") from e

    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    _assert_vision_capable_ollama(model, image_path)

    system = (
        "You are a visual style analyst. Given an image, output a single, comma-separated sentence "
        "describing lighting, illustration/rendering style, line/shape characteristics, shading, anatomy stylization, "
        "textures/gradients, color treatment, and any relevant production tools. "
        "Write it as a style suffix to append to a video prompt. No preamble, no quotes."
        "(comma-separated descriptors, 1–2 short sentences max)."
    )

    user = (
        "Analyse this image to infer its visual *style*. Return a style suffix fit for appending to a prompt—"
        "crisp, comma-separated tokens describing lighting, materials, palette, rendering/animation style, camera/lens, post-processing. "
        "Avoid narration about content; focus on *style*. Keep it compact but useful."
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [image_path]},
            ],
            options={"temperature": float(temperature), "num_predict": int(num_predict)}
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed while analyzing the image: {e}") from e

    try:
        suffix = resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError("Unexpected response while generating style suffix.") from e

    suffix = re.sub(r"\s+", " ", suffix).strip()
    suffix, _ = enforce_word_budget(suffix, 60, protect_suffix=None)  # NEW: clamp
    return suffix


def generate_prompt_from_desc_and_image(
    model: str,
    system_msg: str,
    act_desc: str,
    image_path: str,
    *,
    temperature: float,
    num_predict: int,
    max_words: int = 80,
    entity_hint: str = ""
) -> str:
    """
    Uses a vision LLM to read a reference image and the user act description,
    then produces a SINGLE-SENTENCE prompt (~max_words words), focusing on visuals.
    The optional `entity_hint` disambiguates what the image represents (e.g., “neuron”).
    """
    try:
        import ollama, os, re
    except Exception as e:
        raise RuntimeError("Unable to import 'ollama'. Ensure Ollama is installed and the Python package is available.") from e

    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    _assert_vision_capable_ollama(model, image_path)

    entity_hint = (entity_hint or "").strip()
    hint_line = f"\nThe image represents: {entity_hint}." if entity_hint else ""

    system = (
        (system_msg or "").strip() + "\n\n"
        "CRITICAL: Output exactly one sentence (no lists), British English, focused ONLY on visible content. "
        "Integrate key visual details inferred from the attached image to disambiguate domain objects."
    )

    user = (
        "Task: Write a single-sentence text-to-video prompt for the following act.\n\n"
        f"Act description: \"{(act_desc or '').strip()}\"\n"
        f"{hint_line}\n"
        "Use the attached image to infer precise look/shape/material/texture/colour of the object(s). "
        "Be concrete and visual (camera angle, motion, setting). Avoid exposition and meta-instructions."
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [image_path]},
            ],
            options={"temperature": float(temperature), "num_predict": int(num_predict)}
        )
    except Exception as e:
        raise RuntimeError("Failed to contact Ollama for image+text prompting.") from e

    try:
        content = resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError("Unexpected response shape from Ollama (image+text).") from e

    if not content:
        raise RuntimeError("Model returned empty content.")

    content = re.sub(r"\s+", " ", content).strip()
    content, _ = enforce_word_budget(content, int(max_words), protect_suffix=None)
    return content