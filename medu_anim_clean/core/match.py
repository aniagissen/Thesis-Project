# core/match.py
from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import torch
import open_clip

from .constants import TOP_K_MATCHES
from .models import Take, VisualPlan

@lru_cache(maxsize=1)
def _load_text_model(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, device

@torch.no_grad()
def _encode_texts(texts: List[str]) -> np.ndarray:
    model, tokenizer, device = _load_text_model()
    toks = tokenizer(texts)
    feats = model.encode_text(toks.to(device))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()

def _compositional_queries(vp: VisualPlan) -> List[Tuple[str, float]]:
    parts: List[Tuple[str, float]] = []
    if getattr(vp, "primary_subject", None):
        parts.append((vp.primary_subject, 0.45))
    if getattr(vp, "action", None):
        parts.append((vp.action, 0.35))
    if getattr(vp, "keywords", None):
        parts.append((" ".join(vp.keywords), 0.20))
    return parts or [("medical animation", 1.0)]

def clip_match(vp: VisualPlan, df, vecs: np.ndarray, ids: List[str], k: int = TOP_K_MATCHES) -> List[Take]:
    # 1) build a weighted text embedding from the visual plan
    queries = _compositional_queries(vp)
    texts = [q for q, _ in queries]
    weights = np.array([w for _, w in queries], dtype=np.float32)
    weights = weights / (weights.sum() or 1.0)

    text_embs = _encode_texts(texts)                       # [Q, D]
    q_vec = (weights[:, None] * text_embs).sum(axis=0)     # [D]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    # 2) cosine similarity to precomputed clip embeddings
    sims = (vecs @ q_vec).astype(np.float32)               # [N]

    # 3) small duration penalty vs requested duration
    if getattr(vp, "duration_s", None):
        dur = df["duration"].to_numpy(dtype=np.float32)
        vp_d = max(float(vp.duration_s), 0.1)
        pen = np.clip(np.abs((dur - vp_d) / vp_d), 0.0, 1.0)
        sims = sims - 0.05 * pen

    # 4) take top-k
    top_idx = np.argsort(-sims)[: max(k * 4, k)]           # overfetch a bit
    picks: List[Take] = []
    for i in top_idx:
        row = df.iloc[int(i)]
        picks.append(
            Take(
                source="library",
                clip_id=str(row["id"]),
                clip_uri=str(row["uri"]),
                duration=float(row["duration"]) if row["duration"] else 0.0,
                similarity=float(sims[i]),
                metadata=row.to_dict(),
            )
        )
    picks.sort(key=lambda t: t.similarity, reverse=True)
    return picks[:k]
