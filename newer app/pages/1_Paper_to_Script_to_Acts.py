import json, math
from pathlib import Path
import streamlit as st
from pypdf import PdfReader

from paper_ingest import extract_pdf
from script_builder import build_plain_script
from sentence_utils import split_sentences, soft_split_long_sentence
from tts_service import tts_bytes, mp3_duration_seconds
from prompt_service import generate_prompt
from text_utils import ensure_suffix, enforce_word_budget

st.set_page_config(page_title="Paper → Script → Acts", page_icon="🎞️")

st.title("🎞️ Paper → Script → Acts (sentence-first)")
st.caption("Upload PDF → LLM writes a 2–3 min script → split by sentences → normalize to 3–6s acts → TTS per act → prompts per act.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("LLM (Ollama) model", value="mistral")
    target_words = st.slider("Target words", 280, 500, 360, 10)

    st.divider()
    st.subheader("ElevenLabs")
    voice_id = st.text_input("voice_id", value="JBFqnCBsd6RMkjVDRZzb", help="Your ElevenLabs voice_id")
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
    target_min = st.number_input("Min act length (s)", 2.5, 5.0, 3.0, 0.5)
    target_max = st.number_input("Max act length (s)", 5.5, 8.0, 6.0, 0.5)

    st.divider()
    out_dir = Path(st.text_input("Output folder", value="ProjectActs"))
    fps = st.number_input("FPS for later video gen", 8, 30, 12, 1)

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

script_text = None
if paper:
    if st.button("Generate 2–3 minute script"):
        all_text = "\n\n".join([p["text"] for p in paper["pages"]])
        with st.spinner("Generating script via Ollama..."):
            data = build_plain_script(all_text, model=model_name, target_words=target_words)
            st.session_state["_script_text"] = data["script_text"]
    if "_script_text" in st.session_state and not script_text:
        script_text = st.session_state["_script_text"]

if script_text:
    st.subheader("Script (preview)")
    st.text_area("Full script", value=script_text, height=220)

# Sentence split
sentences = []
if script_text:
    sentences = split_sentences(script_text)
    st.caption(f"Split into {len(sentences)} sentences.")

# Normalize to acts 3–6s using TTS durations
acts = []
if sentences:
    if st.button("Make TTS per sentence and normalize to 3–6s acts"):
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        # First pass: TTS per sentence
        raw = []
        for i, s in enumerate(sentences, 1):
            try:
                audio = tts_bytes(s, voice_id=voice_id, model_id=tts_model, output_format=output_fmt)
                dur = mp3_duration_seconds(audio)
            except Exception as e:
                st.error(f"TTS failed on sentence {i}: {e}")
                continue
            raw.append({"sentence_idx": i, "text": s, "audio": audio, "dur": float(dur)})
        # Merge short (<target_min), split long (>target_max)
        i = 0
        tmp = []
        while i < len(raw):
            cur = raw[i]
            cur_text = cur["text"]
            cur_dur = cur["dur"]
            # too long -> attempt soft split
            if cur_dur > target_max:
                parts = soft_split_long_sentence(cur_text)
                if len(parts) >= 2:
                    # re-TTS both parts
                    merged = []
                    for part in parts[:2]:
                        a = tts_bytes(part, voice_id=voice_id, model_id=tts_model, output_format=output_fmt)
                        d = mp3_duration_seconds(a)
                        merged.append({"text": part, "audio": a, "dur": float(d)})
                    for m in merged:
                        tmp.append(m)
                else:
                    tmp.append({"text": cur_text, "audio": cur["audio"], "dur": cur_dur})
                i += 1
                continue
            # short: merge forward
            if cur_dur < target_min and i+1 < len(raw):
                nxt = raw[i+1]
                combined_text = (cur_text.rstrip() + " " + nxt["text"].lstrip()).strip()
                a = tts_bytes(combined_text, voice_id=voice_id, model_id=tts_model, output_format=output_fmt)
                d = mp3_duration_seconds(a)
                # if still below min and we can add one more, just accept and continue merging in next loop
                tmp.append({"text": combined_text, "audio": a, "dur": float(d)})
                i += 2
                continue
            tmp.append({"text": cur_text, "audio": cur["audio"], "dur": float(cur_dur)})
            i += 1
        # Save to disk and index as acts within a single scene per section (simplify: one scene)
        acts = []
        for j, a in enumerate(tmp, 1):
            audio_path = out_dir / "audio" / f"act_{j:03d}.mp3"
            with open(audio_path, "wb") as f:
                f.write(a["audio"])
            acts.append({"scene": 1, "act": j, "text": a["text"], "audio_path": str(audio_path), "duration_s": round(a["dur"], 2)})
        st.session_state["_acts"] = acts
        st.success(f"Built {len(acts)} acts. Avg duration: {sum(x['duration_s'] for x in acts)/len(acts):.2f}s")

if "_acts" in st.session_state:
    acts = st.session_state["_acts"]
    with st.expander("Acts preview"):
        for a in acts:
            st.markdown(f"**S{a['scene']:02d} A{a['act']:02d}** — {a['duration_s']}s")
            st.write(a["text"])
            st.audio(a["audio_path"])

# Generate visual prompts per act
if acts:
    if st.button("Generate visual prompts for acts"):
        prompts_dir = out_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        jsonl_lines = []
        for a in acts:
            try:
                ptxt = generate_prompt(model=model_name, system_msg=(
                    "You write concise cinematic prompts for a text-to-video model from a narration act. "
                    "Return one paragraph with subject, setting, camera, movement, lighting."
                ), act_desc=a["text"], temperature=0.5, num_predict=300)
            except Exception as e:
                st.error(f"S{a['scene']:02d} A{a['act']:02d} prompt failed: {e}")
                continue
            if enable_suffix:
                ptxt, _ = ensure_suffix(ptxt, suffix_text)
            if max_words:
                ptxt, _ = enforce_word_budget(ptxt, int(max_words), protect_suffix=(suffix_text if enable_suffix else None))
            a["prompt"] = ptxt
            # save per act
            rec = {"scene": a["scene"], "act": a["act"], "text": a["text"], "prompt": ptxt, "duration_s": a["duration_s"]}
            with open(prompts_dir / f"S{a['scene']:02d}_A{a['act']:02d}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            jsonl_lines.append(json.dumps({"scene": a["scene"], "act": a["act"], "generated_prompt": ptxt, "duration_s": a["duration_s"]}))
        # master files
        (out_dir / "prompts.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
        (out_dir / "prompts.txt").write_text("\n".join([json.loads(l)["generated_prompt"] for l in jsonl_lines]) + "\n", encoding="utf-8")
        st.success("Saved prompts.jsonl and prompts.txt")

# Export plan.json
if acts:
    if st.button("Export plan.json"):
        plan = {"fps": int(fps), "acts": acts}
        (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"Saved: {out_dir/'plan.json'}")
