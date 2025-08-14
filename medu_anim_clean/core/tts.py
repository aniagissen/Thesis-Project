"""TTS services. Includes a stub implementation for demos/tests."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from .constants import DEFAULT_TTS_WORDS_PER_SECOND
from .text import hash_text


@dataclass
class TTSResult:
    path: str
    duration_s: float


def synth_tts_stub(base_dir: str, voice_id: str, text: str) -> TTSResult:
    """Create a placeholder 'audio' file and estimate duration by words/second.
    Note: This is NOT a real WAV. Replace with ElevenLabs integration for production.
    """
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, f"tts_{voice_id}_{hash_text(text)}.wav")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("TTS-STUB:" + text)

    words = max(1, len(text.split()))
    duration_s = round(words / DEFAULT_TTS_WORDS_PER_SECOND, 2)
    return TTSResult(path=filename, duration_s=duration_s)
