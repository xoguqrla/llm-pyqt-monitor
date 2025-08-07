# scripts/build_metadata.py
# 목적:
# - 업로드된 원본 테이블들(예: 001_1_data 등)을 훑어 메타테이블 2개를 채움
#   1) session_summary: 세션(테이블) 단위 요약
#   2) events         : 외곽치 이벤트 목록 (시각은 TEXT로 보존)
# - 또한 테이블별 일 단위 집계 뷰 v_daily_temp__<table> 생성
#
# 주요 안전장치:
# - 이벤트 시각(events.ts)은 TEXT로 저장 → 이상한 포맷(예: 02_24_18_07_05_995)도 수용
# - 요약의 start_ts/end_ts는 파싱된 경우에만 TIMESTAMP로 저장, 아니면 NULL
# - boolean 컬럼은 외곽치 연산에서 제외

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# --- project imports (레포 루트 추가) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.db_ops import make_engine

# ---- 상수 --------------------------------------------------------------------
METADATA_TABLE_SUMMARY = "session_summary"
METADATA_TABLE_EVENTS  = "events"

# 후보 시간 컬럼 이름 (소문자 비교)
TIME_CANDIDATES = ["ts", "time", "timestamp", "datetime", "date"]

# 외곽치 경계 (1% / 99%)
LOW_Q  = 0.01
HIGH_Q = 0.99

# 한 세션에서 최대 이벤트 저장 개수(너무 많으면 요약성 떨어짐)
MAX_EVENTS_PER_SESSION = 2000


# ---- 유틸: 스키마/테이블 준비 ---------------------------------------------------
def _find_time_column(eng: Engine, table: str) -> Optional[str]:
    """원본 테이블에서 시간 역할을 할만한 컬럼 후보 탐색."""
    insp = inspect(eng)
    for c in insp.get_columns(table):
        if c["name"].lower() in TIME_CANDIDATES:
            return c["name"]
    return None


def _ensure_metadata_tables(eng: Engine):
    """메타테이블 생성 + 인덱스 + events.ts TEXT 보장(마이그레이션 시도)."""
    with eng.begin() as c:
        # 세션 요약
        c.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE_SUMMARY} (
          session_id     TEXT PRIMARY KEY,
          start_ts       TIMESTAMP NULL,
          end_ts         TIMESTAMP NULL,
          row_count      INTEGER,
          mpt_avg        DOUBLE PRECISION NULL,
          mpt_max        DOUBLE PRECISION NULL,
          mpw_avg        DOUBLE PRECISION NULL,
          anomalies_cnt  INTEGER DEFAULT 0,
          source_table   TEXT
        );
        """))

        # 이벤트(시각은 TEXT로 보관)
        c.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE_EVENTS} (
          id         BIGSERIAL PRIMARY KEY,
          session_id TEXT,
          ts         TEXT NULL,
          event_type TEXT,
          metric     TEXT,
          value      DOUBLE PRECISION,
          threshold  DOUBLE PRECISION,
          severity   INTEGER,
          note       TEXT
        );
        """))

        # 인덱스
        c.execute(text(
            f'CREATE INDEX IF NOT EXISTS idx_events_session_ts ON {METADATA_TABLE_EVENTS}(session_id, ts);'
        ))

        # 기존 DB가 TIMESTAMP였을 가능성 → TEXT로 변환 시도(실패해도 무시)
        try:
            c.execute(text(
                f'ALTER TABLE {METADATA_TABLE_EVENTS} '
                f'ALTER COLUMN ts TYPE TEXT USING ts::text;'
            ))
        except Exception:
            pass


def _list_source_tables(eng: Engine) -> List[str]:
    """메타/시스템 테이블을 제외한 원본 테이블 목록."""
    insp = inspect(eng)
    all_tables = insp.get_table_names()
    exclude = {METADATA_TABLE_SUMMARY, METADATA_TABLE_EVENTS}
    out = []
    for t in all_tables:
        if t in exclude:
            continue
        if t.startswith("pg_") or t.startswith("sqlalchemy_"):
            continue
        out.append(t)
    return out


def _read_table_sample(eng: Engine, table: str, ts_col: Optional[str]) -> pd.DataFrame:
    """원본 테이블 샘플 로딩(최대 20만 행). 시간 컬럼이 있으면 정렬."""
    base = f'SELECT * FROM "{table}"'
    order = f' ORDER BY "{ts_col}"' if ts_col else ""
    sql = base + order + " LIMIT 200000"
    return pd.read_sql(sql, eng)


# ---- 유틸: 시간 처리 -----------------------------------------------------------
def _coerce_ts_series(s: pd.Series) -> Optional[pd.Series]:
    """
    가능하면 Timestamp로 파싱. 실패값은 NaT로 남김.
    - 숫자형: epoch ms → 실패 시 epoch s
    - 문자열: pandas 일반 파서 시도(경고 유발 인자 제거)
    - 특수 포맷(언더스코어 등) → 연도 정보 없으면 파싱 보류(NaT 유지)
    """
    if s is None:
        return None
    s = s.copy()

    # 숫자형(Epoch 가정)
    if is_numeric_dtype(s):
        dt = pd.to_datetime(s, unit="ms", errors="coerce")
        # 대부분 NaT면 초 단위 재시도
        if dt.isna().mean() > 0.8:
            dt = pd.to_datetime(s, unit="s", errors="coerce")
        return dt

    # 문자열 일반 파싱 (infer_datetime_format 인자 제거)
    dt = pd.to_datetime(s, errors="coerce")

    # 언더스코어 패턴 MM_DD_HH_mm_SS_mmm → 연도 불명이라 파싱 보류(NaT 유지)
    mask = dt.isna() & s.astype(str).str.match(r"^\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3}$")
    if mask.any():
        # 필요 시: 고정 연도를 넣는 커스텀 파서 추가 가능(요구될 때 확장)
        pass

    return dt


def _event_ts_as_text(val) -> Optional[str]:
    """
    events.ts 컬럼용 문자열 생성.
    - Timestamp → ISO 문자열
    - 그 외       → str(val)
    """
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return str(val)


# ---- 이벤트/요약 로직 ----------------------------------------------------------
def _detect_events(df: pd.DataFrame, session_id: str, ts_col: Optional[str]) -> List[dict]:
    """
    외곽치(1%/99%) 기반으로 각 수치형 컬럼에서 이벤트 검출.
    - boolean 컬럼 제외
    - events.ts는 TEXT로 넣을 값만 준비
    """
    out = []
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    if not numeric_cols:
        return out

    # 이벤트 시각을 위한 보조 시리즈 (가능하면 Timestamp)
    ts_series = None
    if ts_col and ts_col in df.columns:
        ts_series = _coerce_ts_series(df[ts_col])

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 20:
            continue

        lo, hi = s.quantile([LOW_Q, HIGH_Q])
        peaks = df[df[col] > hi]
        dips  = df[df[col] < lo]

        # 상위/하위 각각 절반 제한
        for _, row in peaks.head(MAX_EVENTS_PER_SESSION // 2).iterrows():
            ts_val = row[ts_col] if (ts_col and ts_col in row) else None
            if ts_series is not None:
                ts_val = _event_ts_as_text(ts_series.loc[row.name])
            else:
                ts_val = _event_ts_as_text(ts_val)

            out.append({
                "session_id": session_id,
                "ts": ts_val,                    # TEXT
                "event_type": "HIGH_SPIKE",
                "metric": col,
                "value": float(row[col]),
                "threshold": float(hi),
                "severity": 2,
                "note": ""
            })

        for _, row in dips.head(MAX_EVENTS_PER_SESSION // 2).iterrows():
            ts_val = row[ts_col] if (ts_col and ts_col in row) else None
            if ts_series is not None:
                ts_val = _event_ts_as_text(ts_series.loc[row.name])
            else:
                ts_val = _event_ts_as_text(ts_val)

            out.append({
                "session_id": session_id,
                "ts": ts_val,                    # TEXT
                "event_type": "LOW_DIP",
                "metric": col,
                "value": float(row[col]),
                "threshold": float(lo),
                "severity": 2,
                "note": ""
            })

    return out


def _upsert_session_summary(
    eng: Engine,
    session_id: str,
    source_table: str,
    df: pd.DataFrame,
    ts_col: Optional[str],
    events_n: int
):
    """
    세션 요약 1행 upsert.
    - start_ts / end_ts: 파싱 성공 시에만 TIMESTAMP로 저장 (아니면 NULL)
    - 그 외 평균/최대/건수 저장
    """
    row_count = len(df)

    start_ts = end_ts = None
    if ts_col and ts_col in df:
        dt = _coerce_ts_series(df[ts_col])
        if dt is not None and not dt.dropna().empty:
            start_ts = dt.min()
            end_ts   = dt.max()

    mpt_avg = float(df["mpt"].mean()) if "mpt" in df else None
    mpt_max = float(df["mpt"].max()) if "mpt" in df else None
    mpw_avg = float(df["mpw"].mean()) if "mpw" in df else None

    with eng.begin() as c:
        c.execute(text(f"""
        INSERT INTO {METADATA_TABLE_SUMMARY}
        (session_id, start_ts, end_ts, row_count, mpt_avg, mpt_max, mpw_avg, anomalies_cnt, source_table)
        VALUES (:sid, :st, :et, :rc, :mavg, :mmax, :wavg, :an, :src)
        ON CONFLICT (session_id) DO UPDATE SET
          start_ts      = excluded.start_ts,
          end_ts        = excluded.end_ts,
          row_count     = excluded.row_count,
          mpt_avg       = excluded.mpt_avg,
          mpt_max       = excluded.mpt_max,
          mpw_avg       = excluded.mpw_avg,
          anomalies_cnt = excluded.anomalies_cnt,
          source_table  = excluded.source_table;
        """), dict(
            sid=session_id,
            st=(start_ts.to_pydatetime() if isinstance(start_ts, pd.Timestamp) else None),
            et=(end_ts.to_pydatetime()   if isinstance(end_ts,   pd.Timestamp) else None),
            rc=row_count,
            mavg=mpt_avg, mmax=mpt_max, wavg=mpw_avg, an=events_n, src=source_table
        ))


def _insert_events(eng: Engine, events: List[dict]):
    """events 테이블에 일괄 append."""
    if not events:
        return
    df = pd.DataFrame(events)
    df.to_sql(METADATA_TABLE_EVENTS, eng, if_exists="append", index=False)


def _ensure_daily_view_per_table(eng: Engine, source_table: str, ts_col: Optional[str]):
    """
    테이블별 일 집계 뷰 생성: v_daily_temp__<table>
    - 시간 컬럼 타입이 timestamp/date/epoch 숫자일 때에만 생성
    - TEXT(포맷 불명)면 안전하게 건너뜀
    - 'mpt'가 있는 경우에만 생성
    """
    if not ts_col:
        return

    insp = inspect(eng)
    cols_meta = {c["name"]: str(c["type"]).lower() for c in insp.get_columns(source_table)}
    if "mpt" not in cols_meta:
        return

    ts_type = cols_meta.get(ts_col, "")
    view_name = f'v_daily_temp__{source_table}'

    # 1) timestamp/date 류
    if any(k in ts_type for k in ["timestamp", "timestamptz", "date"]):
        expr = f'"{ts_col}"'

    # 2) 숫자형 → epoch(초/밀리초) 추정해 to_timestamp 변환
    elif any(k in ts_type for k in ["int", "real", "double", "float", "numeric", "decimal"]):
        # 1e12 이상을 ms로 보고 초로 변환
        expr = (
            f"(to_timestamp(CASE WHEN \"{ts_col}\" > 1000000000000 "
            f"THEN \"{ts_col}\"/1000.0 ELSE \"{ts_col}\" END))"
        )

    # 3) TEXT 등 → 뷰 생성 건너뜀
    else:
        print(f"[skip view] {source_table}.{ts_col} type='{ts_type}' (text/unknown) → v_daily_temp__ 생략")
        return

    with eng.begin() as c:
        c.execute(text(f"""
        CREATE OR REPLACE VIEW "{view_name}" AS
        SELECT date_trunc('day', {expr}) AS day,
               MAX(mpt) AS max_mpt, AVG(mpt) AS avg_mpt, COUNT(*) AS n
        FROM "{source_table}"
        GROUP BY 1;
        """))


# ---- 공개 API ------------------------------------------------------------------
def build_for_table(db_url: str, table: str) -> List[str]:
    """
    원본 테이블 1개 처리:
    - events / session_summary 갱신
    - v_daily_temp__<table> 생성
    """
    eng = make_engine(db_url)
    _ensure_metadata_tables(eng)

    ts_col = _find_time_column(eng, table)
    df = _read_table_sample(eng, table, ts_col)

    # 이벤트 & 요약
    events = _detect_events(df, session_id=table, ts_col=ts_col)

    with eng.begin() as c:
        c.execute(text(f'DELETE FROM {METADATA_TABLE_EVENTS} WHERE session_id = :sid'), {"sid": table})
    _insert_events(eng, events)
    _upsert_session_summary(eng, session_id=table, source_table=table, df=df, ts_col=ts_col, events_n=len(events))

    # 보조 뷰
    _ensure_daily_view_per_table(eng, table, ts_col)
    return [table]


def build_all(db_url: str) -> int:
    """모든 원본 테이블 일괄 처리."""
    eng = make_engine(db_url)
    _ensure_metadata_tables(eng)
    tables = _list_source_tables(eng)
    total = 0
    for t in tables:
        build_for_table(db_url, t)
        total += 1
    return total


def main():
    s = get_settings()
    n = build_all(s.db_url)
    print(f"완료: session_summary / events 생성 및 갱신. 처리 테이블 수={n}")


if __name__ == "__main__":
    main()