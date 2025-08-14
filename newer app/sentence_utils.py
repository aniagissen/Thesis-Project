import re
from typing import List

_abbrev = r"(?:Mr|Mrs|Ms|Dr|Prof|Fig|Eq|Ref|et al)\.$"

def split_sentences(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    # Basic sentence split that tries not to split on common abbreviations
    # Split on ., !, ? followed by space+capital or end of string.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", s)
    out = []
    buf = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        buf.append(t)
        # Heuristic: if ends with terminal punctuation, flush
        if re.search(r"[.!?]['\"]?$", t):
            out.append(" ".join(buf).strip())
            buf = []
    if buf:
        out.append(" ".join(buf).strip())
    # remove tiny remnants
    out = [x for x in out if len(x) > 0]
    return out

def soft_split_long_sentence(s: str) -> List[str]:
    """Try to split a long sentence by semicolons, dashes, or commas into 2 chunks."""
    t = (s or "").strip()
    if not t:
        return []
    for pat in [r";\s+", r"\s+—\s+|\s+-\s+", r",\s+(?=[a-z])"]:
        parts = re.split(pat, t)
        if len(parts) >= 2:
            # join back into at most 2 chunks
            left = parts[0].strip()
            right = ", ".join(parts[1:]).strip()
            return [x for x in [left, right] if x]
    return [t]
