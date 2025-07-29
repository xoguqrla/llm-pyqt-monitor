# core/files_registry.py
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
from dataclasses import dataclass, asdict

REG_PATH = Path("./data/files_registry.json")

@dataclass
class FileEntry:
    file_id: str
    path: str
    sha256: str
    status: str           # "indexed" | "needs_reindex" | "error"
    rows: int = 0
    cols: int = 0
    updated_at: float = time.time()

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_registry() -> dict:
    if REG_PATH.exists():
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    return {}

def save_registry(reg: dict) -> None:
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REG_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_entry(path: Path, rows: int, cols: int, status: str) -> FileEntry:
    reg = load_registry()
    digest = sha256_of(path)
    fid = f"{path.name}:{digest[:10]}"
    entry = FileEntry(file_id=fid, path=str(path), sha256=digest, status=status,
                      rows=rows, cols=cols, updated_at=time.time())
    reg[fid] = asdict(entry)
    save_registry(reg)
    return entry
