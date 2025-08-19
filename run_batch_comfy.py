#!/usr/bin/env python3

"""
Batch-run a ComfyUI workflow for each prompt in a JSONL file.

Usage:
  python run_batch_comfy.py --workflow Hunyant2vANIA.json --jsonl all_prompts.jsonl \
      --server http://127.0.0.1:8000 --prefix MyBatch --wait

What it does:
- Loads the workflow JSON
- For each prompt in JSONL (expects key "generated_prompt"), it sets the text of the node named
  "CLIP Text Encode (Positive Prompt)" (type "CLIPTextEncode"), updates the video filename prefix,
  and POSTs the workflow to ComfyUI's /prompt endpoint.
- Optionally waits for each job to complete and prints the output file info.

Requirements:
  pip install requests
  Start ComfyUI with API enabled (default on: http://127.0.0.1:8000) 
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def load_jsonl(path: Path, key: str = "generated_prompt") -> List[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(f"Line {i} in {path} is not valid JSONL.")
            if key not in obj or not str(obj[key]).strip():
                # allow fallback to pure text lines if the file is a simple .txt
                if len(obj) == 1:
                    # single key, take its value
                    v = next(iter(obj.values()))
                    if v:
                        prompts.append(str(v).strip())
                        continue
                raise RuntimeError(f"Line {i} missing '{key}' field.")
            prompts.append(str(obj[key]).strip())
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}.")
    return prompts


def load_workflow(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def is_api_prompt(workflow: Dict[str, Any]) -> bool:
    # API prompt: dict keyed by node ids, each a dict with 'class_type' and 'inputs'
    return isinstance(workflow, dict) and "nodes" not in workflow and all(
        isinstance(v, dict) and ("class_type" in v and "inputs" in v) for v in workflow.values()
    )

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

def set_clip_text(workflow: Dict[str, Any], node: Dict[str, Any], text: str) -> None:
    if is_api_prompt(workflow):
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise RuntimeError("API CLIPTextEncode node missing inputs dict.")
        inputs["text"] = text
    else:
        w = node.get("widgets_values")
        if not isinstance(w, list) or len(w) == 0:
            raise RuntimeError("UI CLIPTextEncode node does not have expected widgets_values list.")
        w[0] = text

def set_video_prefix(workflow: Dict[str, Any], prefix: str) -> None:
    if is_api_prompt(workflow):
        for n in workflow.values():
            if n.get("class_type") == "VHS_VideoCombine":
                inputs = n.get("inputs", {})
                if isinstance(inputs, dict):
                    inputs["filename_prefix"] = prefix
                return
    else:
        try:
            vhs = find_node(workflow, "VHS_VideoCombine")
        except RuntimeError:
            return
        w = vhs.get("widgets_values")
        if isinstance(w, dict):
            w["filename_prefix"] = prefix
def queue_prompt(server: str, workflow: Dict[str, Any], client_id: str) -> str:
    url = f"{server.rstrip('/')}/prompt"
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # ComfyUI returns either {"prompt_id": "..."} or a dict with "prompt_id" under "data"
    prompt_id = data.get("prompt_id") or data.get("data", {}).get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {data}")
    return prompt_id


def wait_for_completion(server: str, prompt_id: str, poll_s: float = 2.0, timeout_s: int = 3600) -> Dict[str, Any]:
    """
    Poll /history/{prompt_id} until results available or timeout.
    Returns the history entry.
    """
    base = server.rstrip("/")
    start = time.time()
    while True:
        url = f"{base}/history/{prompt_id}"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            hist = r.json()
            # Expect shape: { "prompt_id": { "status": {...}, "outputs": {...} } }
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed") or status.get("status_str") in {"success", "completed"}:
                    return hist[prompt_id]
                if status.get("status_str") in {"error", "failed"}:
                    raise RuntimeError(f"ComfyUI reported error: {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout waiting for job {prompt_id}.")
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, type=Path, help="ComfyUI workflow JSON")
    ap.add_argument("--jsonl", required=True, type=Path, help="JSONL file produced by the Streamlit app")
    ap.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    ap.add_argument("--prefix", default="Batch", help="Filename prefix for outputs")
    ap.add_argument("--wait", action="store_true", help="Wait for each job to complete and print outputs")
    ap.add_argument("--title-filter", default="Positive Prompt", help="Title substring to match the CLIP text node")
    args = ap.parse_args()

    if not args.workflow.exists():
        raise SystemExit(f"Workflow not found: {args.workflow}")
    if not args.jsonl.exists():
        raise SystemExit(f"Prompts file not found: {args.jsonl}")

    prompts = load_jsonl(args.jsonl)
    base_workflow = load_workflow(args.workflow)

    client_id = str(uuid.uuid4())
    print(f"Using client_id={client_id}")
    print(f"Queuing {len(prompts)} jobs to {args.server}")

    for i, prompt_text in enumerate(prompts, 1):
        wf = json.loads(json.dumps(base_workflow))  # deep copy via JSON
        clip_node = find_node(wf, "CLIPTextEncode", title_contains=args.title_filter)
        set_clip_text(wf, clip_node, prompt_text)

        set_video_prefix(wf, f"{args.prefix}_{i:04d}")

        pid = queue_prompt(args.server, wf, client_id)
        print(f"[{i}/{len(prompts)}] queued prompt_id={pid}")

        if args.wait:
            try:
                hist = wait_for_completion(args.server, pid)
                outputs = hist.get("outputs", {})
                print(f"  -> completed. outputs keys: {list(outputs.keys())}")
            except Exception as e:
                print(f"  !! error while waiting: {e}")

    print("Done queuing.")
    

if __name__ == "__main__":
    main()
