#!/usr/bin/env python3
"""Ingest a folder of video clips:
- crawl for media files
- extract basic tech metadata via ffprobe
- extract 3 representative keyframes via ffmpeg (thumbnail filter)
- compute a single CLIP embedding per clip by mean-pooling keyframe embeddings
- write:
    * data/library.parquet  (rows: one per clip, with metadata)
    * data/vectors.npy      (shape: [num_clips, d], CLIP image embeddings)
    * data/id_index.json    (list of clip_ids aligned with vectors.npy row order)

Usage:
  python scripts/ingest_clips.py --assets-dir assets/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Optional GPU if you have it; falls back to CPU
import torch  # type: ignore
import open_clip  # type: ignore


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class ClipMeta:
    id: str
    uri: str
    title: str
    description: str
    source: str  # "library" | "comfy"
    primary_subject: str | None
    anatomy: str | None
    topics: List[str]
    shot_type: str | None
    visual_level: str | None
    color_style: str | None
    movement: str | None
    camera_notes: str | None
    sensitivity: str | None
    style_suffix_id: str | None
    duration: float | None
    fps: float | None
    resolution: str | None
    aspect: float | None
    keywords: List[str]
    checksum: str
    license: str | None
    seed: int | None
    model: str | None
    lora: str | None
    created_at: str


def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ffprobe_metadata(path: Path) -> Dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {err}")
    return json.loads(out or "{}")


def extract_keyframes(path: Path, out_dir: Path, max_frames: int = 3, size: int = 224) -> List[Path]:
    """Use ffmpeg's thumbnail filter to extract representative frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "kf_%02d.jpg"
    vf = f"thumbnail={max_frames},scale={size}:{size}:force_original_aspect_ratio=decrease:eval=frame,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-y", "-i", str(path), "-vf", vf, "-frames:v", str(max_frames), str(pattern)]
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg keyframes failed for {path}: {err}")
    frames = sorted(out_dir.glob("kf_*.jpg"))[:max_frames]
    return frames


def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "openai") -> Tuple[torch.nn.Module, object, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess, device


@torch.no_grad()
def embed_keyframes(model, preprocess, device, frames: List[Path]) -> np.ndarray:
    feats = []
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu().numpy())
    if not feats:
        # fallback zero vector (shouldn't happen if ffmpeg worked)
        return np.zeros((512,), dtype=np.float32)
    mat = np.vstack(feats)  # [n, d]
    mean = mat.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    return mean.astype(np.float32)


def derive_title_from_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def process_clip(model, preprocess, device, root: Path, media_path: Path, kf_root: Path) -> Tuple[ClipMeta, np.ndarray]:
    rel = media_path.relative_to(root)
    checksum = sha256_file(media_path)
    probe = ffprobe_metadata(media_path)
    # duration
    duration = None
    try:
        duration = float(probe.get("format", {}).get("duration"))
    except Exception:
        pass
    # fps & resolution
    fps = None
    width = None
    height = None
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            # fps can be in r_frame_rate like "24000/1001"
            r = s.get("r_frame_rate") or "0/1"
            try:
                num, den = r.split("/")
                fps = float(num) / float(den)
            except Exception:
                fps = None
            width = s.get("width")
            height = s.get("height")
            break
    resolution = f"{width}x{height}" if width and height else None
    aspect = (float(width) / float(height)) if width and height and float(height) != 0 else None

    # keyframes + embedding
    kf_dir = kf_root / checksum[:8]
    frames = extract_keyframes(media_path, kf_dir, max_frames=3, size=224)
    emb = embed_keyframes(model, preprocess, device, frames)

    meta = ClipMeta(
        id=checksum[:16],
        uri=str(rel).replace("\\", "/"),
        title=derive_title_from_name(media_path),
        description="",
        source="library",
        primary_subject=None,
        anatomy=None,
        topics=[],
        shot_type=None,
        visual_level=None,
        color_style=None,
        movement=None,
        camera_notes=None,
        sensitivity=None,
        style_suffix_id=None,
        duration=duration,
        fps=fps,
        resolution=resolution,
        aspect=aspect,
        keywords=[],
        checksum=checksum,
        license=None,
        seed=None,
        model=None,
        lora=None,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    return meta, emb


def crawl_assets(assets_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in assets_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest video clips into a searchable embedding index.")
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "assets"),
        help="Root folder containing your video clips",
    )    
    parser.add_argument("--out-dir", type=str, default="data", help="Where to write outputs (parquet, vectors.npy, id_index.json, keyframes/)")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model name for open_clip")
    parser.add_argument("--pretrained", type=str, default="openai", help="open_clip pretrained tag")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_root = out_dir / "keyframes"
    kf_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning assets in: {assets_dir}")
    media_files = crawl_assets(assets_dir)
    if not media_files:
        print("No media files found. Supported exts:", ", ".join(sorted(VIDEO_EXTS)))
        return

    print(f"Loading CLIP model: {args.model} ({args.pretrained})")
    model, preprocess, device = load_clip_model(args.model, args.pretrained)
    print("Device:", device)

    rows: List[Dict] = []
    vecs: List[np.ndarray] = []

    for path in tqdm(media_files, desc="Ingesting"):
        try:
            meta, emb = process_clip(model, preprocess, device, assets_dir, path, kf_root)
            rows.append(asdict(meta))
            vecs.append(emb)
        except Exception as e:
            print("ERROR processing", path, "-", e)

    if not rows:
        print("No rows ingested.")
        return

    # Save outputs
    df = pd.DataFrame(rows)
    parquet_path = out_dir / "library.parquet"
    vectors_path = out_dir / "vectors.npy"
    id_index_path = out_dir / "id_index.json"

    df.to_parquet(parquet_path, index=False)
    np.save(vectors_path, np.vstack(vecs).astype(np.float32))
    with open(id_index_path, "w", encoding="utf-8") as f:
        json.dump([r["id"] for r in rows], f)

    print("Wrote:")
    print(" ", parquet_path)
    print(" ", vectors_path)
    print(" ", id_index_path)


if __name__ == "__main__":
    main()
