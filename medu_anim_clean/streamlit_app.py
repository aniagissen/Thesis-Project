"""Streamlit UI for the thesis MVP. Single-process, no external services required."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

import streamlit as st

from core.edl import build_edl
from core.library import load_index
from core.match import clip_match
from core.models import SentenceItem, VisualPlan
from core.planning import plan_visual_for_sentence
from core.text import split_scenes, split_sentences
from core.tts import synth_tts_stub
from core.constants import TOP_K_MATCHES
from pathlib import Path

ROOT = Path(__file__).resolve().parent        # medu_anim_clean/
DATA_DIR = ROOT / "data"
df, VECTORS, ID_INDEX = load_index(str(DATA_DIR))

st.set_page_config(page_title="Thesis MVP – Medical Animation Editor", layout="wide")
st.title("Medical Animation Editor – Thesis MVP (Clean Version)")

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    _ = st.file_uploader("Upload paper (PDF or text)", type=["pdf", "txt"], accept_multiple_files=False)
    demographics = st.text_input("Demographics", value="Adults, non-specialists")
    tone = st.selectbox("Tone", ["neutral", "reassuring", "clinical", "enthusiastic"], index=0)
    sensitivity = st.selectbox("Sensitivity", ["low", "medium", "high"], index=0)
    voice_id = st.text_input("Voice ID (placeholder)", value="emma")
    st.caption("This MVP uses stubs for TTS and visual planning. Replace with real engines as you iterate.")

# Session state
if "scenes" not in st.session_state:
    st.session_state.scenes: List[str] = []
if "sentences" not in st.session_state:
    st.session_state.sentences: Dict[str, List[SentenceItem]] = {}
if "edl" not in st.session_state:
    st.session_state.edl: Dict[str, List[Dict]] = {"video": [], "audio": []}

# Script input
st.subheader("Step 1–2: Enter or generate a script (each scene as its own paragraph)")
script_input = st.text_area(
    "Script input",
    value=(
        "Insulin is released from pancreatic beta cells in response to glucose.\n\n"
        "Insulin binds to receptors, triggering GLUT4 translocation to the cell membrane.\n\n"
        "Glucose uptake increases, lowering blood sugar."
    ),
    height=160,
)
if st.button("Use this script"):
    st.session_state.scenes = split_scenes(script_input)
    st.session_state.sentences = {
        f"scene-{i}": [
            SentenceItem(id=f"scene-{i}-s{j}", text=sent)
            for j, sent in enumerate(split_sentences(scene_text))
        ]
        for i, scene_text in enumerate(st.session_state.scenes)
    }
    st.success(f"Parsed {len(st.session_state.scenes)} scenes.")

st.markdown("---")
st.subheader("Per-sentence planning, TTS, and matching")

for i, _ in enumerate(st.session_state.get("scenes", [])):
    st.markdown(f"### Scene {i+1}")
    sentence_items: List[SentenceItem] = st.session_state.sentences[f"scene-{i}"]

    for item in sentence_items:
        with st.container(border=True):
            st.write(f"**Sentence**: {item.text}")

            col_tts, col_plan, col_match = st.columns([1, 1, 2])
            with col_tts:
                if st.button("Generate TTS", key=f"tts-{item.id}"):
                    result = synth_tts_stub(base_dir=".cache/tts", voice_id=voice_id, text=item.text)
                    item.tts_path = result.path
                    item.duration_s = result.duration_s
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
                    for t in takes:
                        c1, c2, c3 = st.columns([3, 1, 1])
                        with c1:
                            st.write(f"{t.clip_id} · {t.metadata['title']}")
                            st.caption(f"{t.metadata['shot_type']} · {t.metadata['visual_level']} · {t.metadata['sensitivity']}")
                        with c2:
                            st.metric("Sim", f"{t.similarity:.2f}")
                        with c3:
                            if st.button("Accept take", key=f"accept-{item.id}-{t.clip_id}"):
                                item.accepted_take = asdict(t)
                                st.success("Accepted")
                else:
                    st.info("Click 'Plan Visual' to see matches.")

st.markdown("---")
st.subheader("Build and download EDL")


if st.button("Build EDL from accepted takes"):
    st.session_state.edl = build_edl(st.session_state.sentences)
    v_count = len(st.session_state.edl["video"])
    a_count = len(st.session_state.edl["audio"])
    st.success(f"EDL built with {v_count} video and {a_count} audio events.")

if st.session_state.edl["video"]:
    st.json(st.session_state.edl)
    edl_path = os.path.join(".cache", f"edl_{int(time.time())}.json")
    os.makedirs(os.path.dirname(edl_path), exist_ok=True)
    with open(edl_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.edl, f, indent=2)
    st.download_button(
        "Download EDL JSON",
        data=json.dumps(st.session_state.edl, indent=2),
        file_name=os.path.basename(edl_path),
        mime="application/json",
    )
