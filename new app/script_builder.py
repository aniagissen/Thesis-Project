import json, re
from typing import Dict, Any

SYSTEM_PROMPT = (
    "You are an expert science communicator. Your task is to write a short, accurate narration "
    "explaining a research paper to a general audience. Use plain language and avoid jargon."
)

USER_TEMPLATE = (
    "You will be given the extracted text of a paper. Create a narration script lasting about {target_words} words "
    "(2–3 minutes at ~140–160 wpm). Structure it as BEATS (1–2 sentences per beat), each referencing source pages.\n\n"
    "Constraints:\n"
    "- Total words: {target_words_min}–{target_words_max}.\n"
    "- Each beat should be 3–5 seconds worth of speech (about 1–2 sentences).\n"
    "- Use inline source refs like [p.3] or [Fig 2, p.7].\n"
    "- Output STRICT JSON with this schema:\n"
    "{\n"
    "  \"script\": {\n"
    "    \"target_duration_s\": 150,\n"
    "    \"beats\": [\n"
    "      {\"id\": 1, \"text\": \"...\", \"source_refs\": [\"Intro p.1\"], \"topic\": \"Hook\"}\n"
    "    ]\n"
    "  }\n"
    "}\n\n"
    "Paper text (may be long, use summarization as needed):\n"
    "<<<PAPER>>>\n{paper}\n<<<END>>>"
)

def _call_ollama(model: str, system: str, user: str) -> str:
    try:
        import ollama
    except Exception as e:
        raise RuntimeError("Ollama not available: install and run the ollama server.") from e
    resp = ollama.chat(model=model, messages=[
        {"role":"system","content":system.strip()},
        {"role":"user","content":user.strip()},
    ], options={"temperature": 0.5, "num_predict": 800})
    return resp["message"]["content"]

def _json_from_text(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}\s*$", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Model did not return valid JSON.")

def build_script_from_paper(paper_text: str, model: str = "mistral", target_words: int = 360):
    user = USER_TEMPLATE.format(
        target_words=target_words,
        target_words_min=int(0.9*target_words),
        target_words_max=int(1.1*target_words),
        paper=paper_text[:150000]
    )
    out = _call_ollama(model, SYSTEM_PROMPT, user)
    data = _json_from_text(out)
    if "script" not in data or "beats" not in data["script"]:
        raise ValueError("Malformed script JSON (missing 'script.beats').")
    for i, b in enumerate(data["script"]["beats"], 1):
        b["id"] = b.get("id", i)
        b["text"] = (b.get("text") or "").strip()
        b["source_refs"] = b.get("source_refs") or []
        b["topic"] = b.get("topic") or ""
    return data
