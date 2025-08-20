#!/usr/bin/env python3
"""
Batch-run a ComfyUI workflow for each prompt in a JSONL or TXT file.
"""
import argparse, json, time, uuid, re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

def load_jsonl(path: Path, key: str = "generated_prompt") -> List[dict]:  # also supports .txt where each line is a prompt
    out = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    def _try_json(line):
        try:
            return json.loads(line)
        except Exception:
            return None

    any_json = any(_try_json(l) is not None for l in raw_lines if l.strip())

    if any_json:
        for i, line in enumerate(raw_lines, 1):
            if not line.strip():
                continue
            obj = _try_json(line)
            if obj is None:
                continue
            if key in obj and str(obj[key]).strip():
                prompt_text = str(obj[key]).strip()
            else:
                if isinstance(obj, dict) and len(obj) == 1:
                    v = next(iter(obj.values()))
                    prompt_text = str(v).strip() if v else ""
                else:
                    continue
            scene = obj.get("scene") if isinstance(obj.get("scene"), int) else None
            act = obj.get("act") if isinstance(obj.get("act"), int) else None
            if prompt_text:
                out.append({"prompt": prompt_text, "scene": scene, "act": act})
    else:
        for ln, line in enumerate(raw_lines, 1):
            t = line.strip()
            if not t:
                continue
            out.append({"prompt": t, "scene": None, "act": None})

    if not out:
        raise RuntimeError(f"No prompts found in {path}.")
    return out

def load_workflow(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def find_node(workflow: Dict[str, Any], type_name: str, title_contains: Optional[str] = None) -> Dict[str, Any]:
    candidates = []
    for n in workflow.get("nodes", []):
        if n.get("type") == type_name:
            if title_contains is None or title_contains.lower() in str(n.get("title", "")).lower():
                candidates.append(n)
    if not candidates:
        raise RuntimeError(f"No node found with type={type_name!r} and title containing {title_contains!r}.")
    return sorted(candidates, key=lambda x: x.get("id", 1_000_000))[0]

def set_clip_text(node: Dict[str, Any], text: str) -> None:
    w = node.get("widgets_values")
    if not isinstance(w, list) or len(w) == 0:
        raise RuntimeError("CLIPTextEncode node does not have expected widgets_values list.")
    w[0] = text

def set_video_prefix(workflow: Dict[str, Any], prefix: str) -> None:
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
    prompt_id = data.get("prompt_id") or data.get("data", {}).get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {data}")
    return prompt_id

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
    ap.add_argument("--jsonl", required=True, type=Path, help="JSONL (or .txt) file with prompts")
    ap.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    ap.add_argument("--prefix", default="Batch", help="Filename prefix for outputs")
    ap.add_argument("--wait", action="store_true", help="Wait for each job to complete and print outputs")
    ap.add_argument("--title-filter", default="Positive Prompt", help="Title substring to match the CLIP text node")
    ap.add_argument("--suffix", default="", help="Optional style suffix to append to each prompt")
    ap.add_argument("--max-words", type=int, default=0, help="Trim final prompt to this many words (keeps suffix)")
    args = ap.parse_args()

    if not args.workflow.exists():
        raise SystemExit(f"Workflow not found: {args.workflow}")
    if not args.jsonl.exists():
        raise SystemExit(f"Prompts file not found: {args.jsonl}")

    items = load_jsonl(args.jsonl)
    base_workflow = load_workflow(args.workflow)

    def _apply_suffix(text: str) -> str:
        s = (args.suffix or "").strip()
        if not s:
            return text.strip()
        nt = text.strip()
        if s.lower() in nt.lower():
            return nt
        if not nt.endswith(('.', '!', '?')):
            nt = nt.rstrip() + "."
        return nt + " " + s

    def _enforce_budget(text: str) -> str:
        mw = int(args.max_words or 0)
        if mw <= 0:
            return text
        base = text.strip()
        words = re.findall(r"[\w'-]+", base)
        if len(words) <= mw:
            return base
        s = (args.suffix or "").strip()
        if s and base.lower().endswith(s.lower()):
            body = base[:-len(s)].rstrip()
            body_words = re.findall(r"[\w'-]+", body)
            keep = max(1, mw - len(re.findall(r"[\w'-]+", s)))
            new_body = " ".join(body_words[:keep]).rstrip(".")
            if not new_body.endswith(('.', '!', '?')):
                new_body = new_body + "."
            return (new_body + " " + s).strip()
        trimmed = " ".join(words[:mw]).rstrip(".")
        if not trimmed.endswith(('.', '!', '?')):
            trimmed = trimmed + "."
        return trimmed

    client_id = str(uuid.uuid4())
    print(f"Using client_id={client_id}")
    print(f"Queuing {len(items)} jobs to {args.server}")

    for i, item in enumerate(items, 1):
        wf = json.loads(json.dumps(base_workflow))
        clip_node = find_node(wf, "CLIPTextEncode", title_contains=args.title_filter)
        final_text = _apply_suffix(item["prompt"])
        final_text = _enforce_budget(final_text)
        set_clip_text(clip_node, final_text)

        if item.get("scene") is not None and item.get("act") is not None:
            prefix = f"{args.prefix}_S{int(item['scene']):02d}_A{int(item['act']):02d}"
        else:
            prefix = f"{args.prefix}_{i:04d}"
        set_video_prefix(wf, prefix)

        pid = queue_prompt(args.server, wf, client_id)
        print(f"[{i}/{len(items)}] queued prompt_id={pid}")
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
