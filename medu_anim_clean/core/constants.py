"""Project-wide constants and default values."""

DEFAULT_TTS_WORDS_PER_SECOND: float = 2.5  # ~150 wpm
SENTENCE_REGEX: str = r"(?<!\b[A-Z])[.?!]+(?=\s+[A-Z])"  # heuristic fallback
MAX_SCENE_DURATION_S: float = 8.0
MIN_SCENE_DURATION_S: float = 4.0
TOP_K_MATCHES: int = 3
