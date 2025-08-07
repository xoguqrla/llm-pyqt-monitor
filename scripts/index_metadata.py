# scripts/index_metadata.py
# 목표:
# - session_summary + events를 한 줄 설명 문서로 만들어 Chroma에 인덱싱
# - 일부 세션만 재인덱싱/세션만 또는 이벤트만 선택 인덱싱 지원
# - 대량 이벤트는 배치로 처리
#
# 사용 예:
#   python scripts/index_metadata.py                        # sessions+events 전체
#   python scripts/index_metadata.py --sessions t_001_1_data,t_005_1_data
#   python scripts/index_metadata.py --sessions-only
#   python scripts/index_metadata.py --events-only --events-limit 80000
#
# 컬렉션 이름 기본값: file_meta_openai (core.rag_ops.build_chroma와 호환)

from __future__ import annotations
from pathlib import Path
import sys
import argparse
from typing import Iterable, List, Sequence, Tuple

from sqlalchemy import create_engine, text

# --- repo import path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.rag_ops import build_embeddings, build_chroma


# ---------------- 유틸 ----------------
def _fmt(x, nd=3):
    """숫자 서식 안전화 (None -> 'nan')"""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"

def _chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------- 세션 요약 인덱싱 ----------------
def _session_row_to_text(r) -> str:
    # r: Row(session_id, start_ts, end_ts, row_count, mpt_avg, mpt_max, mpw_avg, source_table, anomalies_cnt?)
    # anomalies_cnt가 없을 수도 있어 get 사용
    anomalies = getattr(r, "anomalies_cnt", None)
    return (
        f"[SESSION] id={r.session_id} rows={r.row_count} "
        f"mpt_avg={_fmt(r.mpt_avg)} mpt_max={_fmt(r.mpt_max)} mpw_avg={_fmt(r.mpw_avg)} "
        f"anomalies={anomalies if anomalies is not None else 'NA'} "
        f"table={r.source_table} start={r.start_ts} end={r.end_ts}"
    )

def fetch_session_rows(con, session_ids: List[str] | None) -> List:
    if session_ids:
        rows = con.execute(text("""
            SELECT session_id, start_ts, end_ts, row_count, mpt_avg, mpt_max, mpw_avg, anomalies_cnt, source_table
            FROM session_summary
            WHERE session_id = ANY(:ids)
        """), {"ids": session_ids}).fetchall()
    else:
        rows = con.execute(text("""
            SELECT session_id, start_ts, end_ts, row_count, mpt_avg, mpt_max, mpw_avg, anomalies_cnt, source_table
            FROM session_summary
        """)).fetchall()
    return rows

def index_sessions(db_url: str, persist_dir: str, collection: str, session_ids: List[str] | None) -> int:
    eng = create_engine(db_url, future=True)
    with eng.begin() as con:
        rows = fetch_session_rows(con, session_ids)

    if not rows:
        return 0

    s = get_settings()
    emb = build_embeddings(s.openai_key, s.embed_model)
    chroma = build_chroma(emb, persist_dir, collection=collection)

    texts, metas, ids = [], [], []
    for r in rows:
        texts.append(_session_row_to_text(r))
        metas.append({"type": "session_summary", "session_id": r.session_id})
        ids.append(f"session:{r.session_id}")

    # 덮어쓰기 위해 동일 ID 삭제 후 추가
    try:
        chroma.delete(ids=ids)
    except Exception:
        pass
    chroma.add_texts(texts, metadatas=metas, ids=ids)
    return len(ids)


# ---------------- 이벤트 인덱싱 ----------------
def _event_row_to_text(r) -> str:
    # r: Row(session_id, ts, event_type, metric, value, threshold, severity)
    return (
        f"[EVENT] session={r.session_id} ts={r.ts} {r.event_type} "
        f"metric={r.metric} value={_fmt(r.value)} thr={_fmt(r.threshold)} sev={r.severity}"
    )

def fetch_event_rows(con, session_ids: List[str] | None, limit: int | None) -> List:
    base_sql = """
        SELECT session_id, ts, event_type, metric, value, threshold, severity
        FROM events
    """
    where = ""
    params = {}
    if session_ids:
        where = " WHERE session_id = ANY(:ids)"
        params["ids"] = session_ids
    order = " ORDER BY session_id, ts NULLS LAST"
    lim = f" LIMIT {int(limit)}" if (limit and limit > 0) else ""
    sql = base_sql + where + order + lim
    return con.execute(text(sql), params).fetchall()

def index_events(db_url: str, persist_dir: str, collection: str,
                 session_ids: List[str] | None,
                 limit: int | None,
                 batch_size: int = 5000) -> int:
    eng = create_engine(db_url, future=True)
    with eng.begin() as con:
        rows = fetch_event_rows(con, session_ids, limit)

    if not rows:
        return 0

    s = get_settings()
    emb = build_embeddings(s.openai_key, s.embed_model)
    chroma = build_chroma(emb, persist_dir, collection=collection)

    # ID는 session별로 안정적으로 만들되, 간단히 enumerate 인덱스 사용
    total = 0
    # 먼저 동일 세션 범위 삭제 (선택 세션만 들어온 경우)
    if session_ids:
        # events는 수가 많으니 세션 기준으로 와일드카드 삭제가 API에 없으면,
        # 안전하게 '세션별 재색인' 대신 동일 ID를 정확히 만들어 삭제하는 쪽을 권장.
        # 다만 기존에 어떤 인덱스가 들어갔는지 모르면 전량 삭제가 안전.
        pass

    # 배치 업서트 (delete→add_texts 처리)
    # 이벤트는 기존 ID를 모르면 삭제가 어려워서, "덮어쓰기" 개념이 모호합니다.
    # 운영에서는 컬렉션을 세션/버전별로 분리하거나, 기존 event:* 삭제 리스트를 별도로 유지하는 걸 권장.
    # 여기서는 간단히 "추가" 전략을 사용합니다. 필요 시 컬렉션 drop 후 재색인을 채택하세요.
    for i, batch in enumerate(_chunk(rows, batch_size), 1):
        texts, metas, ids = [], [], []
        for j, r in enumerate(batch):
            # id: event:<session_id>:<글로벌인덱스>
            # 단, 글로벌 인덱스가 세션 경계를 모르면 충돌은 없습니다(항상 새로 추가).
            eid = f"event:{r.session_id}:{total + j:08d}"
            ids.append(eid)
            metas.append({
                "type": "event",
                "session_id": r.session_id,
                "metric": r.metric,
                "event_type": r.event_type,
                "severity": r.severity,
            })
            texts.append(_event_row_to_text(r))

        chroma.add_texts(texts, metadatas=metas, ids=ids)
        total += len(batch)

    return total


# ---------------- 일괄 실행(엔트리) ----------------
def main():
    ap = argparse.ArgumentParser(description="Index session_summary + events into Chroma.")
    ap.add_argument("--sessions", type=str, default="", help="쉼표로 구분된 세션ID 목록 (예: t_001_1_data,t_005_1_data)")
    ap.add_argument("--sessions-only", action="store_true", help="세션 요약만 인덱싱")
    ap.add_argument("--events-only", action="store_true", help="이벤트만 인덱싱")
    ap.add_argument("--events-limit", type=int, default=50000, help="이벤트 인덱싱 상한 (기본 50,000)")
    ap.add_argument("--collection", type=str, default="file_meta_openai", help="Chroma 컬렉션명")
    args = ap.parse_args()

    s = get_settings()
    db_url = s.db_url
    persist_dir = str(s.vector_db_dir)
    session_ids = [x.strip() for x in args.sessions.split(",") if x.strip()] or None

    # 실행 모드 분기
    indexed_sessions = indexed_events = 0
    if args.events_only and args.sessions_only:
        print("[warn] --events-only 와 --sessions-only 를 동시에 지정하면 아무 것도 하지 않습니다.")
        return

    if not args.events_only:
        indexed_sessions = index_sessions(db_url, persist_dir, args.collection, session_ids)
        print(f"indexed sessions: {indexed_sessions}")

    if not args.sessions_only:
        indexed_events = index_events(db_url, persist_dir, args.collection, session_ids, limit=args.events_limit)
        print(f"indexed events: {indexed_events}")

    if (args.sessions_only and indexed_sessions == 0) or (args.events_only and indexed_events == 0):
        print("no rows indexed.")


if __name__ == "__main__":
    main()