import os, io, requests
from typing import Tuple, Dict, Any, Optional

API_KEY = os.getenv("ELEVENLABS_API_KEY")
BASE = "https://api.elevenlabs.io/v1"

def tts_bytes(text: str, voice_id: str, *, model_id: str = "eleven_multilingual_v2", output_format: str = "mp3_44100_128", timeout: int = 60) -> bytes:
    if not API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    url = f"{BASE}/text-to-speech/{voice_id}?output_format={output_format}"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.content

def mp3_duration_seconds(mp3_bytes: bytes) -> float:
    try:
        from mutagen.mp3 import MP3
    except Exception as e:
        raise RuntimeError("mutagen not installed; add it to requirements.txt") from e
    bio = io.BytesIO(mp3_bytes)
    audio = MP3(bio)
    return float(audio.info.length or 0.0)
