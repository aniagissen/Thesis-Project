from pathlib import Path
from typing import Dict, Any
from pypdf import PdfReader

def extract_pdf(path: Path) -> Dict[str, Any]:
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages, 1):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"n": i, "text": txt.strip()})
    title = (getattr(reader, "metadata", None) or {}).get("/Title") if hasattr(reader, "metadata") else None
    return {"title": title or path.stem, "num_pages": len(pages), "pages": pages}
