from datetime import datetime
from pathlib import Path
import json
import streamlit as st

from config import ensure_roots, init_session_state, ROOT_SCENES, ROOT_PROMPTS
from storage import ensure_scene_dirs, save_json, append_jsonl, load_json
from prompt_service import generate_prompt
from export_service import create_export_zip
from ui_sidebar import render_sidebar
from ui import act_description_input, show_existing_prompt, act_action_buttons, show_generated_output
from text_utils import ensure_suffix, enforce_word_budget

DEFAULT_SYSTEM = (
    "You are a prompt engineer generating cinematic, production-ready video prompts "
    "for the Hunyuan text-to-video model in ComfyUI. Each output must be a single, "
    "concise paragraph (2–4 sentences) describing the shot with strong visual nouns, "
    "active verbs, camera movement, lens/composition, time of day, lighting, environment, "
    "mood, and color palette. Avoid exposition and meta-notes. No numbering, no lists, "
    "no quotes. Use present tense. British English."
)

def main() -> None:
    ensure_roots()
    master_file = init_session_state()

    st.title("🎬 Prompt Builder – Streamlit + Ollama (modular)")
    st.caption(f"Master file for this session: **{master_file.name}** → saved in `{master_file.parent}`")

    with st.sidebar:
        settings = render_sidebar(DEFAULT_SYSTEM)

    if settings["reset_clicked"]:
        st.session_state.clear()
        st.experimental_rerun()

    scenes_count = settings["scenes_count"]
    acts_per_scene = settings["acts_per_scene"]

    for scene_idx in range(1, int(scenes_count) + 1):
        st.subheader(f"Scene {scene_idx}")
        scene_dir, prompt_dir = ensure_scene_dirs(scene_idx)

        for act_idx in range(1, int(acts_per_scene[scene_idx]) + 1):
            st.markdown(f"**Act {act_idx}**")
            act_desc = act_description_input(scene_idx, act_idx)

            raw_txt_path = scene_dir / f"act_{act_idx}.txt"
            json_path = prompt_dir / f"act_{act_idx}.json"

            existing = load_json(json_path)
            show_existing_prompt(existing, scene_idx, act_idx)

            gen_clicked, show_raw = act_action_buttons(scene_idx, act_idx)
            if show_raw:
                st.code(act_desc or "(empty)")

            if gen_clicked:
                if not act_desc or not act_desc.strip():
                    st.warning("Please enter a description before generating.")
                    st.stop()

                if json_path.exists() and not settings["overwrite_ok"]:
                    st.info("Prompt already exists for this act. Enable 'Overwrite existing prompts' in the sidebar to regenerate.")
                else:
                    try:
                        raw_txt_path.write_text(act_desc.strip() + "\n", encoding="utf-8")
                    except Exception as e:
                        st.error(f"Failed to write raw description: {e}")
                        st.stop()

                    try:
                        prompt_text = generate_prompt(
                            model=settings["model_name"],
                            system_msg=settings["system_prompt"],
                            act_desc=act_desc,
                            temperature=settings["temperature"],
                            num_predict=settings["num_predict"],
                        )
                    except RuntimeError as e:
                        st.error(str(e))
                        st.stop()

                    applied = False
                    if settings.get("enable_suffix"):
                        prompt_text, applied = ensure_suffix(prompt_text, settings.get("suffix_text", ""))
                    trimmed = False
                    if settings.get("max_words"):
                        prompt_text, trimmed = enforce_word_budget(
                            prompt_text, int(settings.get("max_words")), protect_suffix=(settings.get("suffix_text") if settings.get("enable_suffix") else None)
                        )

                    record = {
                        "scene": scene_idx,
                        "act": act_idx,
                        "input_description": act_desc.strip(),
                        "generated_prompt": prompt_text,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "model": settings["model_name"],
                        "options": {"temperature": settings["temperature"], "num_predict": settings["num_predict"]},
                        "style_suffix": {"enabled": bool(settings.get("enable_suffix")), "applied": bool(applied)},
                        "length_control": {"max_words": int(settings.get("max_words", 0)), "trimmed": bool(trimmed)},
                        "paths": {"raw": str(raw_txt_path), "json": str(json_path), "master": str(master_file)},
                    }

                    try:
                        save_json(json_path, record)
                        append_jsonl(master_file, record)
                    except Exception as e:
                        st.error(f"Failed to save outputs: {e}")
                        st.stop()

                    st.success("Prompt generated and saved.")
                    show_generated_output(prompt_text, scene_idx, act_idx)

        st.divider()

    # JSONL download
    if master_file.exists():
        try:
            master_text = master_file.read_text(encoding="utf-8")
            st.download_button(
                label="⬇️ Download master prompts (JSONL)",
                data=master_text,
                file_name=master_file.name,
                mime="application/jsonl",
            )
        except Exception as e:
            st.error(f"Couldn't read master file: {e}")

    # TXT export (one prompt per line)
    if master_file.exists():
        with st.expander("Export prompts.txt (one per line)"):
            dedupe = st.checkbox("Dedupe by scene+act (keep latest)", value=True)
            if st.button("Build prompts.txt"):
                try:
                    lines = []
                    with master_file.open("r", encoding="utf-8") as f:
                        for ln, raw in enumerate(f, 1):
                            raw = raw.strip()
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                continue
                            text = obj.get("generated_prompt", "")
                            if not text:
                                continue
                            # Re-apply suffix and length constraints to be safe
                            if settings.get("enable_suffix"):
                                text, _ = ensure_suffix(text, settings.get("suffix_text", ""))
                                if settings.get("max_words"):
                                    text, _ = enforce_word_budget(text, int(settings.get("max_words", 0)), protect_suffix=settings.get("suffix_text", ""))
                            else:
                                if settings.get("max_words"):
                                    text, _ = enforce_word_budget(text, int(settings.get("max_words", 0)))
                            scene = obj.get("scene")
                            act = obj.get("act")
                            lines.append((ln, scene, act, text))

                    if dedupe:
                        last = {}
                        for ln, sc, ac, txt in lines:
                            key = (sc, ac)
                            last[key] = (ln, txt)
                        ordered = [txt for (_k, (ln, txt)) in sorted(last.items(), key=lambda x: x[1][0])]
                        out_text = "\n".join(ordered) + ("\n" if ordered else "")
                    else:
                        out_text = "\n".join(txt for (_ln, _sc, _ac, txt) in lines) + ("\n" if lines else "")

                    st.download_button(
                        label="⬇️ Download prompts.txt (one per line)",
                        data=out_text,
                        file_name="prompts.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Couldn't build prompts.txt: {e}")

    # ZIP export for Scenes/ and Prompts/
    with st.expander("Export Scenes/ and Prompts/ as a ZIP"):
        if st.button("Create ZIP"):
            data = create_export_zip(ROOT_SCENES, ROOT_PROMPTS)
            st.download_button(label="⬇️ Download export.zip", data=data, file_name="export.zip", mime="application/zip")

    st.caption("Tip: Each act only generates on button click. Existing prompts are preserved unless you enable overwrite.")

if __name__ == "__main__":
    main()
