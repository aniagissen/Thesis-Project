# core/comfy.py
from __future__ import annotations
import os, time, json, requests
from pathlib import Path
from typing import Dict

COMFY_BASE = os.getenv("COMFY_BASE_URL", "http://127.0.0.1:8188")

def post_workflow(workflow_json: Dict) -> str:
    r = requests.post(f"{COMFY_BASE}/prompt", json=workflow_json, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("prompt_id")

def poll_result(prompt_id: str, wait_s: int = 2, timeout_s: int = 600) -> Dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=60)
        if r.status_code == 200 and r.json():
            return r.json()
        time.sleep(wait_s)
    raise TimeoutError("ComfyUI job timeout")

def build_prompt_from_plan(plan: Dict, style_suffix: str) -> str:
    # Simple textual prompt; adjust to your model
    parts = [
        plan.get("shot_type",""),
        plan.get("visual_level",""),
        plan.get("primary_subject",""),
        plan.get("action",""),
        " ".join(plan.get("keywords", [])),
        style_suffix,
    ]
    return ", ".join([p for p in parts if p])

def submit_comfy_generation(plan: Dict, style_suffix: str, seed: int = 1234, steps: int = 30, seconds: float = 4.0) -> Path:
    """
    Requires you to export a ComfyUI workflow JSON with placeholders:
      {PROMPT}, {SEED}, {STEPS}, {DURATION}
    Save it at 'comfy/workflows/diagram_v1.json' (for example).
    """
    wf_path = Path("comfy/workflows/diagram_v1.json")
    if not wf_path.exists():
        raise FileNotFoundError(f"Missing workflow template: {wf_path}")

    prompt_text = build_prompt_from_plan(plan, style_suffix)

    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    wf_str = json.dumps(wf)
    wf_str = wf_str.replace("{PROMPT}", prompt_text)\
                   .replace("{SEED}", str(seed))\
                   .replace("{STEPS}", str(steps))\
                   .replace("{DURATION}", str(max(2.0, float(seconds))))
    wf_payload = json.loads(wf_str)

    prompt_id = post_workflow(wf_payload)
    result = poll_result(prompt_id)

    # find the first video file path in result
    # Each history item typically contains an "outputs" dict; adapt if yours differs
    out_path = None
    for _, item in result.items():
        for node in item.get("outputs", {}).values():
            for f in node.get("files", []):
                if f.get("type","") in {"output","temp"} and f.get("filename","").endswith((".mp4",".mov",".webm")):
                    out_path = Path(f["filename"]).resolve()
                    break
    if not out_path:
        raise RuntimeError("No video output found in ComfyUI result")
    return out_path
