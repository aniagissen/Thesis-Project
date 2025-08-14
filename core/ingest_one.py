# core/ingest_one.py
from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
from .ingest_utils import ffprobe_metadata, sha256_file, extract_keyframes, embed_keyframes  # factor these from your script

def ingest_one(src: Path, assets_root: Path, df: pd.DataFrame, vecs: np.ndarray, ids: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    assets_root = assets_root.resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    # move/copy into assets if needed
    dst = assets_root / src.name
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())

    checksum = sha256_file(dst)
    probe = ffprobe_metadata(dst)
    duration = float(probe.get("format",{}).get("duration", 0.0))
    fps = None; w=None; h=None
    for s in probe.get("streams",[]):
        if s.get("codec_type") == "video":
            r = (s.get("r_frame_rate") or "0/1").split("/")
            fps = float(r[0])/float(r[1]) if float(r[1]) else None
            w, h = s.get("width"), s.get("height")
            break
    resolution = f"{w}x{h}" if w and h else None
    aspect = (float(w)/float(h)) if w and h and float(h) else None

    # keyframes + embedding
    kf_dir = Path("data/keyframes") / checksum[:8]
    frames = extract_keyframes(dst, kf_dir, max_frames=12, size=224)
    emb = embed_keyframes(*load_clip_model(), frames)  # or import model/preprocess/device elsewhere

    row = {
        "id": checksum[:16], "uri": dst.name, "title": dst.stem, "description":"", "source":"comfy",
        "primary_subject": None, "anatomy": None, "topics": [], "shot_type": None, "visual_level": None,
        "color_style": None, "movement": None, "camera_notes": None, "sensitivity": None, "style_suffix_id": None,
        "duration": duration, "fps": fps, "resolution": resolution, "aspect": aspect, "keywords": [],
        "checksum": checksum, "license": None, "seed": None, "model": None, "lora": None,
        "created_at": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds")+"Z",
    }
    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    vecs2 = np.vstack([vecs, emb[None, :]])
    ids2 = ids + [row["id"]]
    # persist
    df2.to_parquet("data/library.parquet", index=False)
    np.save("data/vectors.npy", vecs2)
    json.dump(ids2, open("data/id_index.json","w"))
    return df2, vecs2, ids2
