# XXX finish afer test
import hashlib
import re
from typing import List

from .constants import SENTENCE_REGEX


def hash_text(text: str) -> str:
    """Stable short hash for filenames and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def split_scenes(script: str) -> List[str]:
    """Split a script into scenes by blank-line paragraphs."""
    return [p.strip() for p in script.strip().split("\n\n") if p.strip()]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Try spaCy; fallback to regex.
    This is intentionally conservative for medical text.
    """
    try:
        import spacy  
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
        nlp.add_pipe("sentencizer")
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        return sentences
    except Exception:
        rx = re.compile(SENTENCE_REGEX)
        parts = [s.strip() for s in rx.split(text) if s.strip()]
        return [p if p[-1:] in ".?!" else p + "." for p in parts]
