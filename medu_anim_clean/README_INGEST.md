# Ingest script (embeddings + metadata)

This script turns your `assets/` folder into a searchable index (no server required). It writes:
- `data/library.parquet` : one row per clip with technical metadata and placeholders for tags
- `data/vectors.npy`     : CLIP image embeddings (mean-pooled over 3 keyframes)
- `data/id_index.json`   : clip IDs aligned by row with `vectors.npy`

## Requirements

- ffmpeg/ffprobe installed and on PATH
- Python packages: `torch`, `open-clip-torch`, `pillow`, `numpy`, `pandas`, `tqdm`

## Run

```bash
python scripts/ingest_clips.py --assets-dir assets --out-dir data
```

Use `--model ViT-B-32 --pretrained openai` (default), or try LAION weights if you prefer.
