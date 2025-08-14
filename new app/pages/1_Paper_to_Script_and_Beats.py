import json
from pathlib import Path
import streamlit as st

from paper_ingest import extract_pdf
from script_builder import build_script_from_paper
from tts_service import tts_bytes, mp3_duration_seconds
from text_utils import ensure_suffix, enforce_word_budget
from prompt_service import generate_prompt

st.set_page_config(page_title="Paper → Script → Beats", page_icon="📄")
st.title("📄 Paper → Script → Beats")
st.caption("Upload a PDF → build a 2–3 min explainer → synthesize ElevenLabs VO per beat → generate visual prompts.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("LLM (Ollama) model", value="mistral")
    target_words = st.slider("Target words (2–3 min ≈ 300–450)", 280, 500, 360, 10)

    st.divider()
    st.subheader("ElevenLabs")
    voice_id = st.text_input("voice_id", value="JBFqnCBsd6RMkjVDRZzb", help="Paste your ElevenLabs voice_id")
    tts_model = st.text_input("TTS model_id", value="eleven_multilingual_v2")
    output_fmt = st.selectbox("Output format", ["mp3_44100_128","mp3_44100_64","mp3_22050_32"], index=0)

    st.divider()
    st.subheader("Visual style")
    enable_suffix = st.checkbox("Append style suffix", value=True)
    suffix_text = st.text_area("Suffix", value=(
        "The animation style is flat vector illustration, clean geometric shapes, minimal shading, "
        "soft gradients, cyan–turquoise palette, 2D motion graphics, medical explainer look, crisp edges."
    ), height=120, disabled=not enable_suffix)
    max_words = st.slider("Max words (after suffix)", 20, 120, 80, 5)

    st.divider()
    out_dir = Path(st.text_input("Output folder", value="Project"))
    st.caption("Audio in Project/audio/, prompts in Project/prompts/, plan JSON in Project/plan.json")

pdf_file = st.file_uploader("Upload a paper (PDF)", type=["pdf"])
paper = None
if pdf_file is not None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "paper.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())
    with st.spinner("Extracting text from PDF..."):
        paper = extract_pdf(pdf_path)
    st.success(f"Loaded: {paper['title']} ({paper['num_pages']} pages)")
    with st.expander("Preview extraction (first 2 pages)"):
        for pg in paper["pages"][:2]:
            st.markdown(f"**Page {pg['n']}**\n\n{pg['text'][:2000]}")

script_data = None
if paper:
    if st.button("Generate 2–3 minute script as beats"):
        with st.spinner("Generating script via Ollama..."):
            all_text = "\n\n".join([p["text"] for p in paper["pages"]])
            script_data = build_script_from_paper(all_text, model=model_name, target_words=target_words)
            st.session_state["_script_data"] = script_data
    if "_script_data" in st.session_state and not script_data:
        script_data = st.session_state["_script_data"]

if script_data:
    beats = script_data["script"]["beats"]
    st.subheader("Beats")
    for b in beats:
        st.markdown(f"**Beat {b['id']} ({b.get('topic','')})**: {b['text']}  ")
        if b.get("source_refs"):
            st.caption(f"Refs: {', '.join(b['source_refs'])}")

if script_data:
    if st.button("Synthesize voice per beat"):
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        beats = script_data["script"]["beats"]
        durations = []
        for b in beats:
            text = b["text"]
            try:
                audio = tts_bytes(text, voice_id=voice_id, model_id=tts_model, output_format=output_fmt)
            except Exception as e:
                st.error(f"Beat {b['id']} TTS failed: {e}")
                continue
            fname = audio_dir / f"beat_{int(b['id']):03d}.mp3"
            with open(fname, "wb") as f:
                f.write(audio)
            try:
                d = mp3_duration_seconds(audio)
            except Exception:
                d = 0.0
            b["audio_path"] = str(fname)
            b["duration_s"] = round(float(d), 2)
            durations.append(b["duration_s"])
        st.session_state["_script_data"] = script_data
        if durations:
            st.success(f"Generated {len(durations)} clips. Avg duration: {sum(durations)/len(durations):.2f}s")
    if any("audio_path" in b for b in script_data["script"]["beats"]):
        with st.expander("Audio preview"):
            for b in script_data["script"]["beats"]:
                if "audio_path" in b:
                    st.audio(b["audio_path"])

if script_data:
    if st.button("Generate visual prompts for beats"):
        prompts_dir = out_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        master_lines = []
        for b in script_data["script"]["beats"]:
            brief = b["text"]
            try:
                ptxt = generate_prompt(model=model_name, system_msg=(
                    "You write concise cinematic prompts for a text-to-video model based on a narration beat. "
                    "Return a single paragraph with subject, setting, camera, movement, lighting."
                ), act_desc=brief, temperature=0.5, num_predict=300)
            except Exception as e:
                st.error(f"Beat {b['id']} prompt failed: {e}")
                continue
            applied = False
            if enable_suffix:
                ptxt, applied = ensure_suffix(ptxt, suffix_text)
            if max_words:
                ptxt, _ = enforce_word_budget(ptxt, int(max_words), protect_suffix=(suffix_text if enable_suffix else None))
            b["prompt"] = ptxt
            rec = {"beat_id": b["id"], "text": brief, "prompt": ptxt, "duration_s": b.get("duration_s")}
            with open(prompts_dir / f"beat_{int(b['id']):03d}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            master_lines.append(json.dumps({"scene": None, "act": None, "generated_prompt": ptxt}))
        if master_lines:
            (out_dir / "prompts.jsonl").write_text("\n".join(master_lines) + "\n", encoding="utf-8")
            (out_dir / "prompts.txt").write_text("\n".join([json.loads(l)["generated_prompt"] for l in master_lines]) + "\n", encoding="utf-8")
        st.session_state["_script_data"] = script_data
        st.success("Visual prompts generated and saved.")

if script_data:
    if st.button("Export beat plan JSON"):
        plan = {
            "paper_title": "paper",
            "script": script_data["script"],
            "output_dir": str(out_dir),
            "notes": "Each beat may have audio_path, duration_s, and prompt fields once generated."
        }
        (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"Saved: {out_dir / 'plan.json'}")
