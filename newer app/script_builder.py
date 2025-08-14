from typing import Dict
def build_plain_script(paper_text: str, model: str = "mistral", target_words: int = 360) -> Dict:
    try:
        import ollama
    except Exception as e:
        raise RuntimeError("Ollama not available; install and run it.") from e
    system = (
        "You are an expert science communicator. Write a clear 2–3 minute script "
        "explaining the paper in plain language. Use sections (Hook, Problem, Method, Results, Implications). "
        "Keep it factual and cite the paper inline with [p.X] where appropriate."
    )
    user = (
        f"Write {int(target_words*0.9)}–{int(target_words*1.1)} words total. "
        "Use short sentences. Avoid parentheticals. British English.\n\n"
        "Paper text (extracts; summarize as needed):\n"
        "<<<PAPER>>>\n" + paper_text[:150000] + "\n<<<END>>>"
    )
    resp = ollama.chat(model=model, messages=[
        {"role":"system","content":system},
        {"role":"user","content":user},
    ], options={"temperature": 0.5, "num_predict": 900})
    text = resp["message"]["content"].strip()
    return {"script_text": text}
