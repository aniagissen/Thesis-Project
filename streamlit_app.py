import io
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st

from core.constants import TOP_K_MATCHES
from core.edl import build_edl
from core.library import load_index
from core.match import clip_match
from core.models import SentenceItem, VisualPlan
from core.planning import plan_visual_for_sentence
from core.text import split_scenes, split_sentences
from core.tts import synth_tts 

HAVE_COMFY = True
try:
    from core.comfy import submit_comfy_generation
    from core.ingest_one import ingest_one  
except Exception:
    HAVE_COMFY = False

st.set_page_config(page_title="Thesis MVP – Medical Animation Editor", layout="wide")
ROOT = Path(__file__).resolve().parent   
DATA_DIR = ROOT / "data"
df, VECTORS, ID_INDEX = load_index(str(DATA_DIR))

st.title("Medical Animation Editor – Thesis MVP (Clean Version)")
st.sidebar.caption(f"Index loaded: {len(df)} clips · vecs {getattr(VECTORS, 'shape', None)} · ids {len(ID_INDEX)}")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def extract_pdf_text(file_bytes: bytes) -> str:
    """Plain text from a PDF using pdfminer.six."""
    from pdfminer.high_level import extract_text
    with io.BytesIO(file_bytes) as f:
        text = extract_text(f) or ""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)

def generate_script_with_ollama(
    paper_text: str,
    demographics: str,
    tone: str,
    sensitivity: str,
    target_seconds: int = 150,
) -> str:
    """Ask Ollama to write 8–14 scene paragraphs. Each scene must be a separate paragraph."""
    if not paper_text.strip():
        return ""
    source = paper_text[:120_000] 

    prompt = f"""
You are an educational medical narrator.

Audience: {demographics}
Tone: {tone}
Sensitivity: {sensitivity}
Target total length: {target_seconds}-{target_seconds+30} seconds.

Write a narration as 8–14 SCENES.
Rules:
- Each SCENE is 1–3 short sentences.
- Separate each scene with a single blank line.
- Use only information implied by the SOURCE; avoid speculating.
- Minimize jargon; define any unavoidable term the first time.

SOURCE:
---
{source}
---

OUTPUT:
[Only the scene paragraphs. No titles, numbering, bullets, or extra text.]
""".strip()

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception:
        return ""

with st.sidebar:
    st.header("Inputs")
    demographics = st.text_input("Demographics", value="Adults, non-specialists")
    tone = st.selectbox("Tone", ["neutral", "reassuring", "clinical", "enthusiastic"], index=0)
    sensitivity = st.selectbox("Sensitivity", ["low", "medium", "high"], index=0)
    voice_id = st.text_input("Voice ID (placeholder)", value="emma")
    THRESH = st.slider("Min similarity (τ)", 0.00, 0.60, 0.28, 0.01)
    st.caption("This MVP uses stubs for TTS and visual planning. Replace with real engines as you iterate.")


if "scenes" not in st.session_state:
    st.session_state["scenes"] = []
if "sentences" not in st.session_state:
    st.session_state["sentences"] = {}
if "edl" not in st.session_state:
    st.session_state["edl"] = {"video": [], "audio": []}


st.subheader("Step 1–2: Create the script")
tab_manual, tab_pdf = st.tabs(["Manual", "From PDF"])

with tab_manual:
    script_input = st.text_area(
        "Paste a narration (each scene in its own paragraph)",
        value=(
            "Insulin is released from pancreatic beta cells in response to glucose.\n\n"
            "Insulin binds to receptors, triggering GLUT4 translocation to the cell membrane.\n\n"
            "Glucose uptake increases, lowering blood sugar."
        ),
        height=180,
        key="manual_script",
    )
    if st.button("Use this script", key="use_manual"):
        scenes = split_scenes(script_input)
        st.session_state["scenes"] = scenes
        st.session_state["sentences"] = {
            f"scene-{i}": [
                SentenceItem(id=f"scene-{i}-s{j}", text=s)
                for j, s in enumerate(split_sentences(scene_text))
            ]
            for i, scene_text in enumerate(scenes)
        }
        st.success(f"Parsed {len(scenes)} scenes.")

with tab_pdf:
    uploaded_pdf = st.file_uploader("Upload research paper (PDF)", type=["pdf"], key="paper_pdf")
    st.caption("The model will generate 8–14 short scene paragraphs based on the paper text.")
    if st.button("Generate script from PDF", key="gen_from_pdf", disabled=uploaded_pdf is None):
        if not uploaded_pdf:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Extracting text and calling Ollama..."):
                paper_text = extract_pdf_text(uploaded_pdf.getvalue())
                if len(paper_text.strip()) < 500:
                    st.error("The PDF seems to have little or no extractable text (scanned?). Try another PDF or add OCR.")
                else:
                    script = generate_script_with_ollama(
                        paper_text=paper_text,
                        demographics=demographics,
                        tone=tone,
                        sensitivity=sensitivity,
                        target_seconds=150,
                    )
                    if not script:
                        st.error("Generation failed. Is Ollama running? (`ollama serve`) Is the model pulled?")
                    else:
                        st.text_area(
                            "Generated script (you can edit it before using)",
                            value=script,
                            height=220,
                            key="pdf_script",
                        )
                        if st.button("Use this generated script", key="use_pdf_script"):
                            scenes = split_scenes(st.session_state["pdf_script"])
                            st.session_state["scenes"] = scenes
                            st.session_state["sentences"] = {
                                f"scene-{i}": [
                                    SentenceItem(id=f"scene-{i}-s{j}", text=s)
                                    for j, s in enumerate(split_sentences(scene_text))
                                ]
                                for i, scene_text in enumerate(scenes)
                            }
                            st.success(f"Parsed {len(scenes)} scenes.")

st.markdown("---")
st.subheader("Per-sentence planning, TTS, and matching")

STYLE_SUFFIXES = {
    "style_clinical_v1": "clean clinical palette, thin outlines, soft gradients, no clutter, legible labels",
    "style_icon_v2": "flat iconographic look, simplified shapes, high contrast, minimal shading",
}

for i, _ in enumerate(st.session_state.get("scenes", [])):
    st.markdown(f"### Scene {i+1}")
    sentence_items: List[SentenceItem] = st.session_state["sentences"][f"scene-{i}"]

    for item in sentence_items:
        with st.container(border=True):
            st.write(f"**Sentence**: {item.text}")

            col_tts, col_plan, col_match = st.columns([1, 1, 2])

            with col_tts:
                if st.button("Generate TTS", key=f"tts-{item.id}"):
                    path, dur = synth_tts(base_dir=".cache/tts", voice_id=voice_id, text=item.text)
                    item.tts_path = path
                    item.duration_s = dur
                if item.duration_s:
                    st.caption(f"~{item.duration_s:.1f}s (est)")
                if item.tts_path:
                    st.caption(os.path.basename(item.tts_path))

            with col_plan:
                if st.button("Plan Visual", key=f"plan-{item.id}"):
                    vp = plan_visual_for_sentence(item.text, sensitivity)
                    item.visual_plan = asdict(vp)
                if item.visual_plan:
                    st.json(item.visual_plan)

            with col_match:
                if item.visual_plan:
                    takes = clip_match(VisualPlan(**item.visual_plan), df, VECTORS, ID_INDEX, k=TOP_K_MATCHES)
                    takes = [t for t in takes if t.similarity >= THRESH]

                    if not takes:
                        st.info(f"No matches ≥ {THRESH:.2f}. Try lowering the threshold or adjust the visual plan.")
                    else:
                        for t in takes:
                            c1, c2, c3 = st.columns([3, 1, 1])
                            with c1:
                                st.write(f"{t.clip_id} · {t.metadata.get('title', t.clip_uri)}")
                                st.caption(
                                    f"{t.metadata.get('shot_type', '—')} · "
                                    f"{t.metadata.get('visual_level', '—')} · "
                                    f"{t.metadata.get('sensitivity', '—')}"
                                )
                            with c2:
                                st.metric("Sim", f"{t.similarity:.2f}")
                            with c3:
                                if st.button("Accept take", key=f"accept-{item.id}-{t.clip_id}"):
                                    item.accepted_take = asdict(t)
                                    st.success("Accepted")
                else:
                    st.info("Click 'Plan Visual' to see matches.")

            with st.expander("No good match? Generate with ComfyUI"):
                if not HAVE_COMFY:
                    st.info("ComfyUI bridge not available. Create core/comfy.py and core/ingest_one.py to enable this.")
                else:
                    style_id = st.selectbox("Style suffix", list(STYLE_SUFFIXES.keys()), index=0, key=f"style-{item.id}")
                    seed = st.number_input("Seed", min_value=0, max_value=2**31-1, value=1234, step=1, key=f"seed-{item.id}")
                    steps = st.slider("Steps", 10, 60, 30, 1, key=f"steps-{item.id}")
                    if st.button("Generate with ComfyUI", key=f"gen-{item.id}"):
                        try:
                            plan = item.visual_plan or asdict(plan_visual_for_sentence(item.text, sensitivity))
                            video_path = submit_comfy_generation(plan, STYLE_SUFFIXES[style_id], seed=seed, steps=steps, seconds=plan.get("duration_s", 6.0))
                            st.success(f"ComfyUI done: {Path(video_path).name}")
                            df, VECTORS, ID_INDEX = ingest_one(Path(video_path), Path("assets"), df, VECTORS, ID_INDEX)
                            t = {
                                "source": "comfy",
                                "clip_id": ID_INDEX[-1],
                                "clip_uri": Path(video_path).name,
                                "duration": float(plan.get("duration_s", 6.0)),
                                "similarity": 1.0,
                                "metadata": {"title": Path(video_path).stem, "shot_type": plan.get("shot_type"), "visual_level": plan.get("visual_level")}
                            }
                            if st.button("Accept generated take", key=f"accept-gen-{item.id}"):
                                item.accepted_take = t
                                st.success("Accepted generated clip")
                        except Exception as e:
                            st.error(f"ComfyUI generation failed: {e}")


st.markdown("---")
st.subheader("Build and download EDL")

if st.button("Build EDL from accepted takes"):
    st.session_state["edl"] = build_edl(st.session_state["sentences"])
    v_count = len(st.session_state["edl"]["video"])
    a_count = len(st.session_state["edl"]["audio"])
    st.success(f"EDL built with {v_count} video and {a_count} audio events.")

if st.session_state["edl"]["video"]:
    st.json(st.session_state["edl"])
    edl_path = os.path.join(".cache", f"edl_{int(time.time())}.json")
    os.makedirs(os.path.dirname(edl_path), exist_ok=True)
    with open(edl_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state["edl"], f, indent=2)
    st.download_button(
        "Download EDL JSON",
        data=json.dumps(st.session_state["edl"], indent=2),
        file_name=os.path.basename(edl_path),
        mime="application/json",
    )

def render_from_edl(edl: dict, assets_root: str, out_mp4: str, fps: int = 24) -> str:
    from pathlib import Path
    import tempfile
    video_events = edl.get("video", [])
    assets_root = Path(assets_root).resolve()
    out_mp4 = Path(out_mp4).resolve()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        segs = []
        for i, ev in enumerate(video_events):
            src = Path(ev["uri"])
            if not src.is_absolute():
                src = assets_root / src
            t_in = float(ev.get("in", 0.0)); t_out = float(ev.get("out", 0.0))
            dur = max(0.01, t_out - t_in)
            seg = td / f"seg_{i:03d}.mp4"
            subprocess.run([
                "ffmpeg","-y","-ss",f"{t_in}","-t",f"{dur}","-i",str(src),
                "-r",str(fps),"-an","-vf","scale=1920:-2:flags=bicubic",
                "-c:v","libx264","-preset","veryfast","-crf","20",str(seg)
            ], check=True)
            segs.append(seg)
        concat_list = td/"concat.txt"
        with open(concat_list,"w") as f:
            for p in segs: f.write(f"file '{p.as_posix()}'\n")
        subprocess.run([
            "ffmpeg","-y","-f","concat","-safe","0","-i",str(concat_list),
            "-c:v","libx264","-crf","20","-preset","veryfast","-pix_fmt","yuv420p",
            str(out_mp4)
        ], check=True)
    return str(out_mp4)

if st.session_state["edl"]["video"]:
    if st.button("Render MP4 (video only)"):
        out = render_from_edl(st.session_state["edl"], assets_root="assets", out_mp4="output.mp4")
        st.success(f"Rendered: {out}")
        with open(out, "rb") as f:
            st.download_button("Download MP4", f, file_name="output.mp4", mime="video/mp4")
