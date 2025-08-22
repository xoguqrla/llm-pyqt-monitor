# core/pdf_store.py
from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
from datetime import datetime

def _is_postgres(engine) -> bool:
    try:
        return (engine.dialect.name or "").lower() in ("postgresql", "postgres")
    except Exception:
        return False

def _is_sqlite(engine) -> bool:
    try:
        return (engine.dialect.name or "").lower() == "sqlite"
    except Exception:
        return False

def ensure_pdf_tables(engine) -> None:
    """
    DB 방언에 맞춰 pdf_docs / pdf_chunks 테이블을 생성한다.
    - PostgreSQL: BIGSERIAL / CREATE INDEX IF NOT EXISTS
    - SQLite: INTEGER PRIMARY KEY AUTOINCREMENT
    (created_at는 양쪽 모두 TEXT로 저장하여 단순화)
    """
    if _is_postgres(engine):
        DDL_DOCS = """
        CREATE TABLE IF NOT EXISTS pdf_docs(
          id BIGSERIAL PRIMARY KEY,
          filename TEXT UNIQUE,
          pages INTEGER,
          created_at TEXT
        );
        """
        DDL_CHUNKS = """
        CREATE TABLE IF NOT EXISTS pdf_chunks(
          id BIGSERIAL PRIMARY KEY,
          doc_id INTEGER NOT NULL REFERENCES pdf_docs(id) ON DELETE CASCADE,
          page INTEGER NOT NULL,
          chunk_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          token_len INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_doc   ON pdf_chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_page  ON pdf_chunks(page);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text  ON pdf_chunks(text);
        """
    else:
        # 기본은 SQLite 가정
        DDL_DOCS = """
        CREATE TABLE IF NOT EXISTS pdf_docs(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          filename TEXT UNIQUE,
          pages INTEGER,
          created_at TEXT
        );
        """
        DDL_CHUNKS = """
        CREATE TABLE IF NOT EXISTS pdf_chunks(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id INTEGER NOT NULL,
          page INTEGER NOT NULL,
          chunk_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          token_len INTEGER,
          FOREIGN KEY (doc_id) REFERENCES pdf_docs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_doc   ON pdf_chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_page  ON pdf_chunks(page);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text  ON pdf_chunks(text);
        """
    with engine.begin() as c:
        c.exec_driver_sql(DDL_DOCS)
        c.exec_driver_sql(DDL_CHUNKS)

def insert_pdf(engine, filename: str, pages: int) -> int:
    with engine.begin() as c:
        c.exec_driver_sql(
            "INSERT INTO pdf_docs(filename,pages,created_at) VALUES(?,?,?) "
            "ON CONFLICT (filename) DO NOTHING" if _is_sqlite(engine) else
            "INSERT INTO pdf_docs(filename,pages,created_at) VALUES(%s,%s,%s) "
            "ON CONFLICT (filename) DO NOTHING",
            (filename, pages, datetime.utcnow().isoformat()),
        )
        # sqlite/postgres 모두에서 동작하도록 별도 SELECT
        row = c.exec_driver_sql(
            ("SELECT id FROM pdf_docs WHERE filename=?" if _is_sqlite(engine) else
             "SELECT id FROM pdf_docs WHERE filename=%s"),
            (filename,),
        ).first()
    return int(row[0])

def insert_chunks(engine, doc_id: int, rows: Iterable[Tuple[int,int,str]]) -> List[int]:
    """
    rows: iterable of (page, chunk_index, text)
    return: inserted chunk ids (in order)
    """
    ids: List[int] = []
    with engine.begin() as c:
        for page, idx, text in rows:
            c.exec_driver_sql(
                ("INSERT INTO pdf_chunks(doc_id,page,chunk_index,text,token_len) VALUES(?,?,?,?,?)"
                 if _is_sqlite(engine) else
                 "INSERT INTO pdf_chunks(doc_id,page,chunk_index,text,token_len) VALUES(%s,%s,%s,%s,%s)"),
                (doc_id, page, idx, text, len(text)),
            )
            rid = c.exec_driver_sql(
                "SELECT last_insert_rowid()" if _is_sqlite(engine) else
                "SELECT currval(pg_get_serial_sequence('pdf_chunks','id'))"
            ).scalar()
            ids.append(int(rid))
    return ids

def list_chunk_ids_by_doc(engine, doc_id: int) -> List[int]:
    with engine.begin() as c:
        rows = c.exec_driver_sql(
            "SELECT id FROM pdf_chunks WHERE doc_id=? ORDER BY id" if _is_sqlite(engine) else
            "SELECT id FROM pdf_chunks WHERE doc_id=%s ORDER BY id",
            (doc_id,),
        ).fetchall()
    return [int(r[0]) for r in rows]

def fetch_chunks_by_ids(engine, ids: List[int]) -> List[str]:
    if not ids: return []
    qmarks = ",".join(("?" if _is_sqlite(engine) else "%s") for _ in ids)
    with engine.begin() as c:
        rows = c.exec_driver_sql(
            f"SELECT id, text FROM pdf_chunks WHERE id IN ({qmarks})",
            tuple(ids),
        ).fetchall()
    m = {int(i): t for i, t in rows}
    return [m.get(i, "") for i in ids]

def delete_doc(engine, doc_id: int) -> None:
    with engine.begin() as c:
        c.exec_driver_sql(
            "DELETE FROM pdf_chunks WHERE doc_id=?" if _is_sqlite(engine) else
            "DELETE FROM pdf_chunks WHERE doc_id=%s",
            (doc_id,),
        )
        c.exec_driver_sql(
            "DELETE FROM pdf_docs WHERE id=?" if _is_sqlite(engine) else
            "DELETE FROM pdf_docs WHERE id=%s",
            (doc_id,),
        )

def keyword_search_chunks(engine, query: str, limit: int = 6) -> List[Tuple[int, str]]:
    """
    아주 단순한 백업 검색: LIKE 기반. (원하면 Postgres FTS/SQLite FTS5로 확장 가능)
    return: [(chunk_id, text), ...]
    """
    import re
    tokens = [t for t in re.findall(r"[A-Za-z0-9가-힣]{2,}", query) if len(t) >= 2][:5]
    if not tokens:
        tokens = [query.strip()][:1]
    like = " OR ".join(["text LIKE ?"] * len(tokens)) if _is_sqlite(engine) \
        else " OR ".join(["text ILIKE %s"] * len(tokens))
    params = [f"%{t}%" for t in tokens]
    sql = f"SELECT id, text FROM pdf_chunks WHERE {like} ORDER BY token_len LIMIT {limit}"
    with engine.begin() as c:
        rows = c.exec_driver_sql(sql, tuple(params)).fetchall()
    return [(int(r[0]), r[1]) for r in rows]


# --- 아래 함수를 core/pdf_store.py 맨 아래에 추가 ---
def list_all_docs(engine):
    """pdf_docs 테이블의 모든 문서를 반환: [(doc_id, filename, pages, created_at), ...]"""
    with engine.begin() as c:
        rows = c.exec_driver_sql(
            "SELECT id, filename, pages, created_at FROM pdf_docs ORDER BY filename"
        ).fetchall()
    return [(int(r[0]), r[1], int(r[2] or 0), r[3]) for r in rows]
