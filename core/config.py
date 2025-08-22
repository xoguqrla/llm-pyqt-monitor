# # core/config.py
# from __future__ import annotations
# import os
# from dataclasses import dataclass
# from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv()

# @dataclass
# class Settings:
#     db_url: str
#     openai_key: str
#     openai_model: str
#     embed_model: str
#     meta_json_dir: Path
#     vector_db_dir: Path
#     uploads_dir: Path

# def get_settings() -> Settings:
#     db_url = os.getenv("DATABASE_URL", "").strip()
#     openai_key = os.getenv("OPENAI_API_KEY", "").strip()
#     openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
#     embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
#     meta_json_dir = Path(os.getenv("META_JSON_DIR", "./data/meta_json"))
#     vector_db_dir = Path(os.getenv("VECTOR_DB_DIR", "./data/chroma_openai"))
#     uploads_dir = Path("./data/uploads")

#     # ensure dirs
#     for p in (meta_json_dir, vector_db_dir, uploads_dir):
#         p.mkdir(parents=True, exist_ok=True)

#     return Settings(
#         db_url=db_url,
#         openai_key=openai_key,
#         openai_model=openai_model,
#         embed_model=embed_model,
#         meta_json_dir=meta_json_dir,
#         vector_db_dir=vector_db_dir,
#         uploads_dir=uploads_dir,
#     )

# core/config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # --- ▼▼▼ 1. 여기에 data_dir를 추가합니다 ▼▼▼ ---
    data_dir: Path
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---
    db_url: str
    openai_key: str
    openai_model: str
    embed_model: str
    meta_json_dir: Path
    vector_db_dir: Path
    uploads_dir: Path

def get_settings() -> Settings:
    # --- ▼▼▼ 2. data_dir 변수를 정의합니다 ▼▼▼ ---
    data_dir = Path("./data")
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---
    db_url = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{data_dir / 'local.db'}"
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    meta_json_dir = Path(os.getenv("META_JSON_DIR", str(data_dir / "meta_json")))
    vector_db_dir = Path(os.getenv("VECTOR_DB_DIR", str(data_dir / "chroma_openai")))
    uploads_dir = data_dir / "uploads"

    # ensure dirs
    # --- ▼▼▼ 3. 생성할 폴더 목록에 data_dir를 추가합니다 ▼▼▼ ---
    for p in (data_dir, meta_json_dir, vector_db_dir, uploads_dir):
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---
        p.mkdir(parents=True, exist_ok=True)

    return Settings(
        # --- ▼▼▼ 4. 반환하는 Settings 객체에 data_dir를 포함시킵니다 ▼▼▼ ---
        data_dir=data_dir,
        # --- ▲▲▲ 수정 완료 ▲▲▲ ---
        db_url=db_url,
        openai_key=openai_key,
        openai_model=openai_model,
        embed_model=embed_model,
        meta_json_dir=meta_json_dir,
        vector_db_dir=vector_db_dir,
        uploads_dir=uploads_dir,
    )