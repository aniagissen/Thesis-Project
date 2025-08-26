from datetime import datetime
from pathlib import Path
import json
import streamlit as st
import subprocess
import sys
import os

from prompt_service import generate_prompt, generate_prompt_from_desc_and_image  # add second import
from ui import act_description_input, show_existing_prompt, act_action_buttons, show_generated_output, act_reference_image_controls  # include the new controls


from config import ensure_roots, init_session_state, ROOT_SCENES, ROOT_PROMPTS
from storage import ensure_scene_dirs, save_json, append_jsonl, load_json
from export_service import create_export_zip
from ui_sidebar import render_sidebar
from text_utils import ensure_suffix, enforce_word_budget
import time
from typing import Dict, Any



# System prompt sent to llama
DEFAULT_SYSTEM = (
    "You are a prompt engineer generating clean and clear video prompts  "
    "for the Hunyuan text-to-video model in ComfyUI. Each output must be a single, "
    "concise sentence, describing the shot with a clear visual description of  "
    "the scene. Avoid exposition and meta-notes. No numbering, no lists, no  "
    "quotes. Use present tense, and British English."
)

# Entry log to build and run UI
def main() -> None:
    ensure_roots() # Locates root folders
    master_file = init_session_state() # Sets up session logs

    st.title("Biomations Prompt Builder")
    st.caption(f"Master file for this session: **{master_file.name}** → saved in `{master_file.parent}`")
    
    # Set up and render interactive sidebar options
    with st.sidebar:
        settings = render_sidebar(DEFAULT_SYSTEM)

    # Clears streamlit and runs fresh script
    if settings["reset_clicked"]:
        st.session_state.clear()
        st.experimental_rerun()

    # Reads default from settings and 
    scenes_count = settings["scenes_count"]
    acts_per_scene = settings["acts_per_scene"]

    # Makes/finds folders for scenes and prompts
    for scene_idx in range(1, int(scenes_count) + 1):
        st.subheader(f"Scene {scene_idx}")
        scene_dir, prompt_dir = ensure_scene_dirs(scene_idx)

        for act_idx in range(1, int(acts_per_scene[scene_idx]) + 1):
            st.markdown(f"**Act {act_idx}**")
            act_desc = act_description_input(scene_idx, act_idx) # Text box for user inputting prompt per act per scene
            use_ref, ref_file, ref_entity = act_reference_image_controls(scene_idx, act_idx) 

            # Two file paths for saving outputs
            raw_txt_path = scene_dir / f"act_{act_idx}.txt"
            json_path = prompt_dir / f"act_{act_idx}.json"

            # In case of propmt existing, find and display it
            existing = load_json(json_path)
            show_existing_prompt(existing, scene_idx, act_idx)

            # Show raw act text
            gen_clicked, show_raw = act_action_buttons(scene_idx, act_idx)
            if show_raw:
                st.code(act_desc or "(empty)")

            if gen_clicked:
                # Back up in case no text is entered when pressing 'Generate'
                if not act_desc or not act_desc.strip():
                    st.warning("Please enter a description before generating.")
                    st.stop()
                # Back up if prompt has already been genereated. Instructions for how to override
                if json_path.exists() and not settings["overwrite_ok"]:
                    st.info("A prompt already exists for this act. Enable 'Overwrite existing prompts' in the sidebar to regenerate.")
                    st.stop
                else:
                    try:
                        raw_txt_path.write_text(act_desc.strip() + "\n", encoding="utf-8")
                    except Exception as e:
                        st.error(f"Failed to write raw description: {e}")
                        st.stop()

                try:
                    # If user provided a reference image, save it and use the vision flow
                    ref_image_path = None
                    if use_ref and ref_file is not None:
                        try:
                            sid = st.session_state.get("session_timestamp", "session_default")
                            save_dir = ROOT_PROMPTS / sid / "act_refs" / f"scene_{scene_idx}_act_{act_idx}"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            ext = Path(ref_file.name).suffix or ".png"
                            ref_image_path = save_dir / f"reference{ext}"
                            with open(ref_image_path, "wb") as f:
                                f.write(ref_file.getbuffer())
                        except Exception as e:
                            st.warning(f"Could not save reference image. Proceeding without it. ({e})")
                            ref_image_path = None

                    if ref_image_path:
                        # Vision model: use the same model_name (llama3.2-vision:11b) or expose a separate setting if you prefer
                        prompt_text = generate_prompt_from_desc_and_image(
                            model=settings["model_name"],
                            system_msg=settings["system_prompt"],
                            act_desc=act_desc,
                            image_path=str(ref_image_path),
                            temperature=settings["temperature"],
                            num_predict=settings["num_predict"],
                            max_words=int(settings.get("max_words", 80)),
                            entity_hint=ref_entity,
                        )
                    else:
                        # Text-only fallback
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

                # Apply style suffix to end
                applied = False
                if settings.get("enable_suffix"):
                    prompt_text, applied = ensure_suffix(prompt_text, settings.get("suffix_text", ""))
                trimmed = False
                # protects style suffix is prompt is too long
                if settings.get("max_words"):
                    prompt_text, trimmed = enforce_word_budget(
                        prompt_text, int(settings.get("max_words")), protect_suffix=(settings.get("suffix_text") if settings.get("enable_suffix") else None)
                    )

                # Records everything
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
                    "reference": {
                        "image": str(ref_image_path) if ref_image_path else None,
                        "entity": ref_entity or None,
                    },
                    "paths": {"raw": str(raw_txt_path), "json": str(json_path), "master": str(master_file)},
                }


                # Write the JSON file and stick on the master JSONL
                try:
                    save_json(json_path, record)
                    append_jsonl(master_file, record)
                except Exception as e:
                    st.error(f"Failed to save outputs: {e}")
                    st.stop()

                # Confidence check
                st.success("Prompt generated and saved.")
                show_generated_output(prompt_text, scene_idx, act_idx)

        st.divider()

    # If JSONL was created successfully, option to download it
    if master_file.exists():
        try:
            master_text = master_file.read_text(encoding="utf-8")
            st.download_button(
                label="Download master prompts (JSONL)",
                data=master_text,
                file_name=master_file.name,
                mime="application/jsonl",
            )
        except Exception as e:
            st.error(f"Couldn't read master file: {e}")


    # Create .txt file from the master JSONL ('generated_prompt')
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
                        label="Download prompts.txt (one per line)",
                        data=out_text,
                        file_name="prompts.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Couldn't build prompts.txt: {e}")

    # Option for ZIP export for Scenes/ and Prompts/
    with st.expander("Export Scenes/ and Prompts/ as a ZIP"):
        if st.button("Create ZIP"):
            data = create_export_zip(ROOT_SCENES, ROOT_PROMPTS)
            st.download_button(label="Download export.zip", data=data, file_name="export.zip", mime="application/zip")

    
    # Funtion to bacth run the promts in  ComfyUI
    with st.expander("Batch run prompts in ComfyUI"):
        sid = st.session_state.get("session_timestamp", "session_default")
        default_prefix = f"Batch_{sid}"

        # Takes set scripts unless input changes
        default_batch = "run_batch_comfy.py"
        try:
            from pathlib import Path as _P
            if _P("run_batch_comfy.py").exists():
                default_batch = "run_batch_comfy.py"
            elif _P("run_batch_comfy.py").exists():
                default_batch = "run_batch_comfy.py"
        except Exception:
            pass

        # Inputs needed
        batch_script = st.text_input("Batch script path", value=default_batch)
        workflow_path = st.text_input("Workflow JSON path (API format)", value="")
        server_url = st.text_input("ComfyUI server URL", value="http://127.0.0.1:8188")
        prefix = st.text_input("Filename prefix", value=default_prefix)
        title_filter = st.text_input("CLIP node title filter", value="Positive Prompt")

        # Option to either send prompts one by one (to test) or send them all at once
        wait_done = st.checkbox("Wait for each job to complete", value=True)
        add_cli_suffix = st.checkbox("Add/override style suffix at batch time", value=False)
        cli_suffix = st.text_area("Batch-time suffix (optional)", value="", disabled=not add_cli_suffix)
        max_words_cli = st.number_input("Max words at batch time (0 = no trim)", min_value=0, max_value=200, value=0, step=5)


        c1, c2 = st.columns(2)
        # Sends the prompts!
        with c1:
            run_clicked = st.button("Run batch prompts now", use_container_width=True)
        # ComfyUI browser has to already be running, test options
        with c2:
            test_clicked = st.button("Test ComfyUI is running", use_container_width=True)
        if test_clicked:
            try:
                import requests
                resp = requests.get(f"{server_url.rstrip('/')}/system_stats", timeout=5)
                st.success(f"Server OK ({resp.status_code}) at {server_url}")
                st.code(resp.text[:1200] + ("..." if len(resp.text) > 1200 else ""), language="json")
            except Exception as e:
                st.error(f"Server test failed: {e}")

        if run_clicked:

            if not master_file.exists():
                st.error("Master JSONL not found. Generate at least one prompt first.")

            elif not workflow_path:
                st.error("Please provide a Workflow JSON path.")

            else:
                # Allow saving the stdout log like before
                save_logs = st.checkbox("Save logs to this session", value=True,
                                        help="Writes batch_log.txt into the session's Prompts folder.")

                args = [sys.executable, "-u", str(batch_script),
                        "--workflow", str(workflow_path),
                        "--jsonl", str(master_file),
                        "--server", server_url,
                        "--prefix", prefix,
                        "--title-filter", title_filter]

                if wait_done:
                    args.append("--wait")
                if add_cli_suffix and cli_suffix.strip():
                    args.extend(["--suffix", cli_suffix.strip()])
                if max_words_cli and int(max_words_cli) > 0:
                    args.extend(["--max-words", str(int(max_words_cli))])

                try:
                    if settings.get("override_params"):
                        if settings.get("steps") is not None:
                            args.extend(["--steps", str(int(settings["steps"]))])
                        if settings.get("guidance") is not None:
                            args.extend(["--guidance", str(float(settings["guidance"]))])
                        if settings.get("width") is not None:
                            args.extend(["--width", str(int(settings["width"]))])
                        if settings.get("height") is not None:
                            args.extend(["--height", str(int(settings["height"]))])
                        if settings.get("length") is not None:
                            args.extend(["--length", str(int(settings["length"]))])
                        if settings.get("fps") is not None:
                            args.extend(["--fps", str(int(settings["fps"]))])
                        if settings.get("format"):
                            args.extend(["--format", str(settings["format"]).strip()])
                        if settings.get("seed") is not None and settings["seed"] != "":
                            args.extend(["--seed", str(int(settings["seed"]))])
                except Exception as e:
                    st.warning(f"Could not add parameter overrides: {e}")

                # --- Simple streaming status (no WebSocket) ---
                start_ts = time.time()
                with st.status("🎥 Generating videos…", expanded=True) as status:
                    info_line = st.empty()
                    current_line = st.empty()
                    log_box = st.code("", language="bash")
                    lines = []
                    completed = 0
                    total = None
                    current_label = ""

                    # Launch the batch subprocess
                    try:
                        proc = subprocess.Popen(
                            args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )
                    except FileNotFoundError:
                        status.update(label="Could not launch Python or the batch script. Check the paths.", state="error")
                        st.stop()
                    except Exception as e:
                        status.update(label=f"Batch run error: {e}", state="error")
                        st.stop()

                    import re as _re
                    re_begin  = _re.compile(r'^BATCH_BEGIN\s+total=(\d+)\b')
                    re_jobbeg = _re.compile(r'^JOB_BEGIN\s+i=(\d+)\s+of=(\d+)\s+label=(.+)$')
                    re_done   = _re.compile(r'^JOB_DONE\s+i=(\d+)\b')
                    re_error  = _re.compile(r'^JOB_ERROR\s+i=(\d+)\s+error="?(.*?)"?$')
                    re_end    = _re.compile(r'^BATCH_END\b')

                    while True:
                        line = proc.stdout.readline() if proc.stdout else ""
                        if line:
                            line = line.rstrip("\n")
                            lines.append(line)

                            m = re_begin.match(line)
                            if m:
                                total = int(m.group(1))

                            m = re_jobbeg.match(line)
                            if m:
                                current_label = m.group(3).strip()

                            if re_done.match(line):
                                completed += 1

                            m = re_error.match(line)
                            if m:
                                completed += 1  # count error as finished

                            elapsed = int(time.time() - start_ts)
                            if total:
                                info_line.markdown(f"**Status:** {completed}/{total} completed • {elapsed}s elapsed")
                            else:
                                info_line.markdown(f"**Status:** starting… • {elapsed}s elapsed")
                            if current_label:
                                current_line.write(f"**Current:** {current_label}")

                            log_box.code("\n".join(lines[-400:]), language="bash")

                            if re_end.match(line):
                                break

                        if proc.poll() is not None and not line:
                            break

                        time.sleep(0.05)

                    # Save logs if requested
                    if save_logs:
                        try:
                            sid = st.session_state.get("session_timestamp", "session_default")
                            log_dir = (ROOT_PROMPTS / sid)
                            log_dir.mkdir(parents=True, exist_ok=True)
                            (log_dir / "batch_log.txt").write_text("\n".join(lines), encoding="utf-8")
                            st.success("Saved logs to Prompts/<session>/batch_log.txt")
                        except Exception as _e:
                            st.warning(f"Could not save logs: {_e}")

                    rc = proc.returncode or 0
                    if rc == 0:
                        status.update(label="✅ Batch finished. Check ComfyUI outputs.", state="complete")
                    else:
                        status.update(label=f"⚠️ Batch exited with code {rc}. See logs above.", state="error")



    
    # Browse ComfyUI outputs in streamlit
    with st.expander("Browse your ComfyUI outputs"):
        try:
            default_out = st.session_state.get("comfy_output_dir", str(Path.home() / "ComfyUI" / "output")) #Change folder to users liking
            out_dir = st.text_input("Output folder", value=default_out, help="Path where ComfyUI saves outputs (videos).")
            st.session_state.comfy_output_dir = out_dir

            show_prefix = st.text_input("Filter by filename prefix (optional)", value="", help="Use the same prefix you batch with to narrow results.")
            if st.button("Refresh list"):
                pass  # rerun refresh

            # Determine N from master JSONL (scene+act -> count)
            show_n = 0
            if master_file.exists():
                try:
                    last = {}
                    with master_file.open("r", encoding="utf-8") as f:
                        for ln, raw in enumerate(f, 1):
                            raw = raw.strip()
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                continue
                            sc = obj.get("scene")
                            ac = obj.get("act")
                            if sc is not None and ac is not None:
                                last[(sc, ac)] = ln
                    show_n = max(1, len(last))
                except Exception:
                    show_n = 20
            else:
                show_n = 20

            base = Path(os.path.expanduser(out_dir)).resolve()
            if not base.exists():
                st.warning(f"Folder does not exist: {base}")
            else:
                files = [p for p in base.rglob("*.mp4") if p.is_file()] # Only mp4s, no pngs
                if show_prefix:
                    files = [p for p in files if p.name.startswith(show_prefix)]
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                files = files[:show_n]

                if not files:
                    st.info("No matching MP4 files found yet. Generate a video in ComfyUI, then refresh.")
                else:
                    from datetime import datetime as _dt
                    st.caption(f"Showing the last {len(files)} MP4 file(s).")
                    for f in files:
                        try:
                            mtime = _dt.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            mtime = "(unknown time)"
                        st.markdown(f"**{f.name}**  —  _{mtime}_")
                        try:
                            st.video(str(f))
                        except Exception:
                            pass
                        try:
                            data_bytes = f.read_bytes()
                            st.download_button("Download", data=data_bytes, file_name=f.name, key=str(f))
                        except Exception as e:
                            st.warning(f"Can't read file for download: {e}")
        except Exception as e:
            st.error(f"Couldn't browse outputs: {e}")
        st.caption("Tip: Each act only generates on button click. Existing prompts are preserved unless you enable overwrite.")

if __name__ == "__main__":
    main()
