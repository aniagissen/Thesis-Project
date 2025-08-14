# Prompt Builder (Modular)

Modularised Streamlit app to turn short act descriptions into production-ready cinematic prompts for Hunyuan/ComfyUI.

## Files
- `app.py` – main entrypoint.
- `config.py` – directories + session initialisation.
- `storage.py` – file I/O helpers.
- `prompt_service.py` – Ollama call for prompt generation.
- `export_service.py` – export ZIP utility.
- `ui_sidebar.py` – sidebar renderer.
- `ui.py` – small UI helpers for acts.
- `requirements.txt` – dependencies.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Ensure the `ollama` daemon is running and the specified model is pulled.
