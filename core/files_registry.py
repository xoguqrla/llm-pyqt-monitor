# core/files_registry.py
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
from dataclasses import dataclass, asdict, field

REG_PATH = Path("./data/files_registry.json")

@dataclass
class FileEntry:
    file_id: str
    path: str
    sha256: str
    status: str           # "indexed" | "needs_reindex" | "error"
    rows: int = 0
    cols: int = 0
    updated_at: float = field(default_factory=time.time)  # 안전한 시간 기록

def sha256_of(path: Path) -> str:
    """파일의 SHA-256 해시 계산"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sanitize_dict(d: dict) -> dict:
    """
    예약 키('file') 충돌 방지를 위해 dict 키를 정리.
    필요 시 'file' -> '_file' 로 치환.
    """
    clean = {}
    for k, v in d.items():
        if k == "file":
            clean["_file"] = v
        else:
            clean[k] = v
    return clean

def load_registry() -> dict:
    """레지스트리 파일 로드 (깨진 경우 빈 dict 반환, 키 정리 포함)"""
    if REG_PATH.exists():
        try:
            reg = json.loads(REG_PATH.read_text(encoding="utf-8"))
            # 혹시 'file' 키가 남아 있다면 치환
            return {fid: sanitize_dict(meta) for fid, meta in reg.items()}
        except json.JSONDecodeError:
            return {}
    return {}

def save_registry(reg: dict) -> None:
    """레지스트리 저장"""
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 저장 전에도 sanitize
    reg = {fid: sanitize_dict(meta) for fid, meta in reg.items()}
    REG_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_entry(path: Path, rows: int, cols: int, status: str) -> FileEntry:
    """파일 엔트리를 레지스트리에 삽입 또는 갱신"""
    reg = load_registry()
    digest = sha256_of(path)
    fid = f"{path.name}:{digest[:10]}"

    entry = FileEntry(
        file_id=fid,
        path=str(path),
        sha256=digest,
        status=status,
        rows=rows,
        cols=cols,
        updated_at=time.time()
    )

    d = asdict(entry)
    d = sanitize_dict(d)  # 혹시라도 'file' 키 생기면 치환

    reg[fid] = d
    save_registry(reg)
    return entry
