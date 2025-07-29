# core/rag_ops.py
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 안정성: 텔레메트리 끄기
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")


# ---------- Embeddings ----------
def build_embeddings(api_key: str, model: str) -> OpenAIEmbeddings:
    """OpenAI 임베딩 빌더"""
    return OpenAIEmbeddings(model=model, api_key=api_key)


# ---------- Chroma (0.4.x) ----------
def build_chroma(
    embeddings: OpenAIEmbeddings,
    persist_dir: str | Path,
    collection: str = "file_meta_openai",
) -> Chroma:
    """
    - Chroma 0.4.x용 VectorStore 생성
    - 기존 다른 버전으로 만들어진 인덱스를 열 때 스키마 불일치가 나면
      폴더를 자동 백업 후 새로 생성하여 복구
    """
    pdir = Path(persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    def _make() -> Chroma:
        return Chroma(
            collection_name=collection,
            persist_directory=str(pdir),
            embedding_function=embeddings,
        )

    try:
        return _make()
    except Exception as e:
        # 예: sqlite3.OperationalError: no such column: collections.topic
        msg = str(e).lower()
        if "no such column" in msg or "schema" in msg or "mismatch" in msg:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = pdir.with_name(pdir.name + f"_backup_{ts}")
            try:
                shutil.move(str(pdir), str(backup))
            except Exception:
                pass
            pdir.mkdir(parents=True, exist_ok=True)
            return _make()
        raise


# ---------- Simple (파일 단위 요약) ----------
def meta_to_text(meta: dict) -> str:
    """CSV 메타를 단일 텍스트로 직렬화 (간단 RAG용)"""
    lines = [f"file:{meta['file']} rows:{meta['rows']} cols:{meta['cols']}"]
    for k, v in meta["columns"].items():
        s = f"column:{k} dtype:{v['dtype']} non_null:{v['non_null']} nulls:{v['nulls']}"
        if "stats" in v:
            st = v["stats"]
            s += f" stats(min:{st['min']}, max:{st['max']}, mean:{st['mean']})"
        lines.append(s)
    return "\n".join(lines)


def index_meta(chroma: Chroma, meta: dict) -> None:
    """요약 텍스트를 1건 문서로 색인"""
    doc = meta_to_text(meta)
    try:
        chroma.add_texts(
            [doc],
            metadatas=[{"type": "file_meta", "file": meta["file"]}],
            ids=[f"meta:{meta['file']}"],
        )
    except Exception as e:
        # 0.4.x에선 차원/컬렉션 충돌 시 다양한 예외가 섞여 나올 수 있어 broad-catch
        raise RuntimeError(
            "Chroma add_texts 실패(차원/컬렉션 충돌 가능). "
            "인덱스 디렉터리를 비우거나 새 컬렉션으로 재생성하세요."
        ) from e


def retrieve_meta(chroma: Chroma, query: str, k: int = 6):
    """유사도 검색"""
    return chroma.similarity_search(query, k=k)


# ---------- Fine-grained (컬럼 단위) ----------
def build_embedding_texts_from_meta(meta: dict) -> List[Tuple[str, dict]]:
    """
    파일 단위 + 컬럼 단위로 쪼갠 텍스트/메타를 생성
    반환: [(text, metadata), ...]
    """
    out: List[Tuple[str, dict]] = []
    out.append(
        (
            f"file:{meta['file']} rows:{meta['rows']} cols:{meta['cols']}",
            {"chunk_type": "file", "file": meta["file"], "rows": meta["rows"], "cols": meta["cols"]},
        )
    )
    for col, v in meta["columns"].items():
        line = f"column:{col} dtype:{v['dtype']} non_null:{v['non_null']} nulls:{v['nulls']}"
        if "stats" in v:
            s = v["stats"]
            line += f" stats(min:{s['min']}, max:{s['max']}, mean:{s['mean']})"
        out.append((line, {"chunk_type": "column", "file": meta["file"], "column": col}))
    return out


def upsert_texts(chroma: Chroma, file_id: str, texts_meta: List[Tuple[str, dict]]) -> None:
    """
    컬럼 단위 텍스트를 id와 함께 색인 (중복 방지)
    ids: {file_id}:{0000}
    """
    texts = [t for t, _ in texts_meta]
    metas = []
    ids = []
    for i, (_, m) in enumerate(texts_meta):
        m = dict(m)
        m["file_id"] = file_id
        metas.append(m)
        ids.append(f"{file_id}:{i:04d}")
    try:
        chroma.add_texts(texts, metadatas=metas, ids=ids)
    except Exception as e:
        raise RuntimeError(
            "Chroma add_texts 실패(차원/컬렉션 충돌 가능). "
            "인덱스 디렉터리를 비우거나 새 컬렉션으로 재생성하세요."
        ) from e
