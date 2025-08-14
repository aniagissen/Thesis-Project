# core/tts.py
from __future__ import annotations
import os, hashlib, requests
from dataclasses import dataclass

DEFAULT_WPS = 2.5  # words per second, used for estimate if needed

@dataclass
class TTSResult:
    path: str
    duration_s: float

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]

def synth_tts(base_dir: str, voice_id: str, text: str) -> TTSResult:
    os.makedirs(base_dir, exist_ok=True)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        # fallback: no audio, just a duration estimate
        fn = os.path.join(base_dir, f"tts_{voice_id}_{_hash(text)}.txt")
        with open(fn, "w", encoding="utf-8") as f: f.write(text)
        return TTSResult(path=fn, duration_s=round(len(text.split())/DEFAULT_WPS, 2))

    fn = os.path.join(base_dir, f"tts_{voice_id}_{_hash(text)}.mp3")
    if os.path.exists(fn):
        # crude reuse; if you want precise duration, parse with pydub
        return TTSResult(path=fn, duration_s=round(len(text.split())/DEFAULT_WPS, 2))

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key}
    payload = {"text": text, "model_id": "eleven_multilingual_v2"}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    with open(fn, "wb") as f: f.write(r.content)
    return TTSResult(path=fn, duration_s=round(len(text.split())/DEFAULT_WPS, 2))
