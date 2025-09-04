import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def ensure_suffix(prompt: str, suffix: str) -> tuple[str, bool]:
    p = (prompt or "").strip()
    suf = (suffix or "").strip()
    if not suf:
        return p, False
    if normalize(suf).lower() in normalize(p).lower():
        return p, False
    if not p.endswith(('.', '!', '?')):
        p = p.rstrip() + "."
    return p + " " + suf, True

def enforce_word_budget(text: str, max_words: int, protect_suffix: str | None = None) -> tuple[str, bool]:
    try:
        mw = int(max_words)
    except Exception:
        mw = 0
    if mw <= 0:
        return (text or "").strip(), False
    text = (text or "").strip()
    words = re.findall(r"[\w'-]+", text)
    if len(words) <= mw:
        return text, False
    if protect_suffix:
        norm_tail = normalize(protect_suffix).lower()
        norm_text = normalize(text).lower()
        if norm_text.endswith(norm_tail):
            tail_words = re.findall(r"[\w'-]+", protect_suffix)
            head_budget = max(1, mw - len(tail_words))
            head_words = re.findall(r"[\w'-]+", text)[:head_budget]
            head = " ".join(head_words).rstrip(".")
            if not head.endswith(('.', '!', '?')):
                head = head + "."
            return (head + " " + protect_suffix.strip()).strip(), True
    trimmed_words = words[:mw]
    out = " ".join(trimmed_words).rstrip(".")
    if not out.endswith(('.', '!', '?')):
        out = out + "."
    return out, True
