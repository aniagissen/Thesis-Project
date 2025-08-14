# Medical Animation Editor – Thesis MVP (Clean Code)

Single-process Streamlit app with organized modules. Stubs for TTS and visual planning.
Replace stubs with ElevenLabs and LLM calls; replace simple matcher with CLIP/FAISS.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Structure

```
core/
  constants.py        # shared constants
  edl.py              # EDL helpers
  library.py          # loads local clip library JSON
  match.py            # simple matcher (replace with CLIP retrieval)
  models.py           # dataclasses
  planning.py         # visual plan (stub -> LLM)
  text.py             # hashing and split utils
  tts.py              # TTS stub (replace with ElevenLabs)

app/
  streamlit_app.py    # UI

data/
  sample_library.json # example library entries

assets/
  ... your demo mp4s go here ...
```
