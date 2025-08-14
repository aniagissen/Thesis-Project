import io, zipfile
from pathlib import Path

def create_export_zip(*paths: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in paths:
            base = Path(base)
            if base.is_file():
                zf.write(base, base.as_posix())
            else:
                for p in base.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.as_posix())
    buf.seek(0)
    return buf.read()
