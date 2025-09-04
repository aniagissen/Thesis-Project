"""
Batch-run a ComfyUI workflow for each prompt in a JSONL file.

Usage:
  python run_batch_comfy.py --workflow Hunyuan_API.json --jsonl all_prompts.jsonl \
      --server http://127.0.0.1:8188 --prefix MyBatch --wait
"""
import argparse, json, time, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import re

# Input/Output helpers
# Reads JSONL and checks each line is compatible
def iter_jsonl_records(path: Path, key: str = "generated_prompt") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(f"Line {i} in {path} is not valid JSONL.")
            if key not in obj or not str(obj[key]).strip():
                # allow single-field JSONL, stores final text to '_text'
                if len(obj) == 1:
                    v = next(iter(obj.values()))
                    if v:
                        obj["_text"] = str(v).strip()
                        records.append(obj)
                        continue
                    #Shows errors
                raise RuntimeError(f"Line {i} missing '{key}' field.")
            obj["_text"] = str(obj[key]).strip()
            records.append(obj)
    if not records:
        raise RuntimeError(f"No prompts found in {path}.")
    return records

def load_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# Workflow helpers for API and JSON
# Registers the workflow being an API
def is_api_prompt(workflow: Dict[str, Any]) -> bool:
    # API format has dict of nodes keyed by id, no top-level "nodes" list
    return isinstance(workflow, dict) and "nodes" not in workflow and all(
        isinstance(v, dict) and ("class_type" in v and "inputs" in v) for v in workflow.values()
    )

# Locates nodes in workflow
def find_node(workflow: Dict[str, Any], type_name: str, title_contains: Optional[str] = None) -> Dict[str, Any]:
    if is_api_prompt(workflow):
        candidates = []
        for n in workflow.values():
            if n.get("class_type") == type_name:
                title = (n.get("_meta", {}) or {}).get("title", "")
                if title_contains is None or title_contains.lower() in str(title).lower():
                    candidates.append(n)
        if not candidates:
            raise RuntimeError(f"No node found with class_type={type_name!r} and title containing {title_contains!r}.")
        return candidates[0]
    else:
        candidates = []
        for n in workflow.get("nodes", []):
            if n.get("type") == type_name:
                if title_contains is None or title_contains.lower() in str(n.get("title", "")).lower():
                    candidates.append(n)
        if not candidates:
            raise RuntimeError(f"No node found with type={type_name!r} and title containing {title_contains!r}.")
        return sorted(candidates, key=lambda x: x.get("id", 1_000_000))[0]

# Sends users prompt into the CLIP text node
def set_clip_text(workflow: Dict[str, Any], node: Dict[str, Any], text: str) -> None:
    if is_api_prompt(workflow):
        node.setdefault("inputs", {})["text"] = text
    else:
        w = node.get("widgets_values")
        if not isinstance(w, list) or len(w) == 0:
            raise RuntimeError("UI CLIPTextEncode node has unexpected shape.")
        w[0] = text

# Edits the output name in VHSvideoCombine node
def set_video_prefix(workflow: Dict[str, Any], prefix: str) -> None:
    if is_api_prompt(workflow):
        for n in workflow.values():
            if n.get("class_type") == "VHS_VideoCombine":
                n.setdefault("inputs", {})["filename_prefix"] = prefix
                return
    else:
        try:
            vhs = find_node(workflow, "VHS_VideoCombine")
        except RuntimeError:
            return
        w = vhs.get("widgets_values")
        if isinstance(w, dict):
            w["filename_prefix"] = prefix
        elif isinstance(w, list) and w:
            # assume first widget is filename_prefix for VHS_VideoCombine in UI export
            w[0] = prefix

# Override helpers
def set_scheduler_steps(workflow: Dict[str, Any], steps: Optional[int]) -> None:
    if steps is None: return
    if is_api_prompt(workflow):
        try:
            node = find_node(workflow, "BasicScheduler")
            node["inputs"]["steps"] = int(steps)
        except Exception:
            pass

def set_flux_guidance(workflow: Dict[str, Any], guidance: Optional[float]) -> None:
    if guidance is None: return
    if is_api_prompt(workflow):
        try:
            node = find_node(workflow, "FluxGuidance")
            node["inputs"]["guidance"] = float(guidance)
        except Exception:
            pass

def set_latent_dims(workflow: Dict[str, Any], width: Optional[int], height: Optional[int], length: Optional[int]) -> None:
    if not any([width, height, length]): return
    if is_api_prompt(workflow):
        try:
            node = find_node(workflow, "EmptyHunyuanLatentVideo")
            if width  is not None: node["inputs"]["width"]  = int(width)
            if height is not None: node["inputs"]["height"] = int(height)
            if length is not None: node["inputs"]["length"] = int(length)
        except Exception:
            pass

def set_noise_seed(workflow: Dict[str, Any], seed: Optional[int]) -> None:
    if seed is None: return
    if is_api_prompt(workflow):
        try:
            node = find_node(workflow, "RandomNoise")
            node["inputs"]["noise_seed"] = int(seed) if int(seed) != 0 else int(time.time()*1000) % 2147483647
        except Exception:
            pass

def set_video_params(workflow: Dict[str, Any], fps: Optional[int], fmt: Optional[str]) -> None:
    if not any([fps, fmt]): return
    if is_api_prompt(workflow):
        try:
            node = find_node(workflow, "VHS_VideoCombine")
            if fps is not None: node["inputs"]["frame_rate"] = int(fps)
            if fmt: node["inputs"]["format"] = str(fmt)
        except Exception:
            pass

# ComfyUI HTTP
def queue_prompt(server: str, workflow: Dict[str, Any], client_id: str) -> str:
    url = f"{server.rstrip('/')}/prompt"
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    pid = data.get("prompt_id") or data.get("data", {}).get("prompt_id")
    if not pid:
        raise RuntimeError(f"Unexpected /prompt response: {data}")
    return pid

# Provides update when status is completed, or there is an error
def wait_for_completion(server: str, prompt_id: str, poll_s: float = 2.0, timeout_s: int = 3600) -> Dict[str, Any]:
    base = server.rstrip("/")
    start = time.time()
    while True:
        url = f"{base}/history/{prompt_id}"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            hist = r.json()
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed") or status.get("status_str") in {"success","completed"}:
                    return hist[prompt_id]
                if status.get("status_str") in {"error","failed"}:
                    raise RuntimeError(f"ComfyUI reported error: {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout waiting for job {prompt_id}.")
        time.sleep(poll_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, type=Path, help="ComfyUI workflow JSON (API format preferred)")
    ap.add_argument("--jsonl", required=True, type=Path, help="App-produced JSONL (needs 'generated_prompt')")
    ap.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    ap.add_argument("--prefix", default="Batch", help="Filename prefix for outputs")
    ap.add_argument("--wait", action="store_true", help="Wait for each job to complete and print outputs")
    ap.add_argument("--title-filter", default="Positive Prompt", help="Title substring to match the CLIP text node")
    # optional CLI additions from Streamlit
    ap.add_argument("--suffix", type=str, default=None, help="Extra suffix appended to each prompt at batch time")
    ap.add_argument("--max-words", type=int, default=0, help="Trim prompts to this many words (0=off)")
    ap.add_argument("--steps", type=int, default=None, help="Override BasicScheduler steps")
    ap.add_argument("--guidance", type=float, default=None, help="Override FluxGuidance guidance")
    ap.add_argument("--width", type=int, default=None, help="Override EmptyHunyuanLatentVideo width")
    ap.add_argument("--height", type=int, default=None, help="Override EmptyHunyuanLatentVideo height")
    ap.add_argument("--length", type=int, default=None, help="Override EmptyHunyuanLatentVideo length (frames)")
    ap.add_argument("--fps", type=int, default=None, help="Override VHS_VideoCombine frame_rate")
    ap.add_argument("--format", type=str, default=None, help="Override VHS_VideoCombine format (e.g., video/nvenc_h264-mp4)")
    ap.add_argument("--seed", type=int, default=None, help="Override RandomNoise noise_seed (0=random)")
    ap.add_argument("--client-id", dest="client_id", default=None, help="(Ignored) legacy WS client id")

    args = ap.parse_args()

    if not args.workflow.exists():
        raise SystemExit(f"Workflow not found: {args.workflow}")
    if not args.jsonl.exists():
        raise SystemExit(f"Prompts file not found: {args.jsonl}")

    records = iter_jsonl_records(args.jsonl)
    base_workflow = load_workflow(args.workflow)

    print(f"BATCH_BEGIN total={len(records)} server={args.server}", flush=True)

    for i, rec in enumerate(records, 1):
        prompt_text = str(rec.get("_text", "")).strip()

        # optional suffix & budget
        if args.suffix:
            if not prompt_text.endswith(('.', '!', '?')):
                prompt_text = prompt_text.rstrip() + "."
            if args.suffix.lower() not in prompt_text.lower():
                prompt_text = f"{prompt_text} {args.suffix}"
        if args.max_words and args.max_words > 0:
            words = re.findall(r"[\w'-]+", prompt_text)
            if len(words) > args.max_words:
                prompt_text = " ".join(words[:args.max_words]).rstrip(".")
                if not prompt_text.endswith(('.', '!', '?')):
                    prompt_text += "."

        # copy base workflow so each batch starts from the same template
        wf = json.loads(json.dumps(base_workflow))

        # Find CLIP node and set text
        clip_node = find_node(wf, "CLIPTextEncode", title_contains=args.title_filter)
        set_clip_text(wf, clip_node, prompt_text)

        # Build filename label used both for saving and UI
        scene = rec.get("scene")
        act = rec.get("act")
        created = rec.get("created_at")
        dt_date = dt_time = None
        if isinstance(created, str):
            try:
                from datetime import datetime as _dt
                t = _dt.fromisoformat(created.replace("Z", ""))
                dt_date = t.strftime("%Y%m%d")
                dt_time = t.strftime("%H%M%S")
            except Exception:
                pass
        if dt_date is None or dt_time is None:
            import time as _time
            tt = _time.localtime()
            dt_date = _time.strftime("%Y%m%d", tt)
            dt_time = _time.strftime("%H%M%S", tt)

        if scene is not None and act is not None:
            per_prefix = f"Scene{int(scene)}_Act{int(act)}_{dt_date}_{dt_time}"
        else:
            per_prefix = f"{args.prefix}_{i:04d}"

        # Apply filename and optional overrides
        set_video_prefix(wf, per_prefix)
        set_scheduler_steps(wf, args.steps)
        set_flux_guidance(wf, args.guidance)
        set_latent_dims(wf, args.width, args.height, args.length)
        set_noise_seed(wf, args.seed)
        set_video_params(wf, args.fps, args.format)

        print(f"JOB_BEGIN i={i} of={len(records)} label={per_prefix}", flush=True)

        # Queue the job
        pid = queue_prompt(args.server, wf, "batch")
        print(f"JOB_QUEUED i={i} prompt_id={pid}", flush=True)

        if args.wait:
            try:
                _ = wait_for_completion(args.server, pid)
                print(f"JOB_DONE i={i}", flush=True)
            except Exception as e:
                msg = (str(e) or "").replace("\n", " ").strip()
                print(f'JOB_ERROR i={i} error="{msg}"', flush=True)

    print("BATCH_END", flush=True)

if __name__ == "__main__":
    main()
