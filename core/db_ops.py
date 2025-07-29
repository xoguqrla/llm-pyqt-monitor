# core/db_ops.py
from __future__ import annotations
import re
import pandas as pd
from sqlalchemy import create_engine, text

def make_engine(db_url: str):
    return create_engine(db_url, pool_pre_ping=True, future=True)

def table_name_from_file(filename: str) -> str:
    base = filename.rsplit('.', 1)[0].lower()
    safe = re.sub(r'[^0-9a-z_]+', '_', base)
    return f"t_{safe}"

def ingest_df(engine, df: pd.DataFrame, table: str):
    df.to_sql(table, engine, if_exists="append", index=False)

def ensure_indexes(engine, table: str, candidate_time_cols=('time','timestamp','ts','date')):
    with engine.begin() as conn:
        for c in candidate_time_cols:
            res = conn.execute(text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name=:t AND column_name=:c
            """), {"t": table, "c": c}).fetchone()
            if res:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{table}_{c} ON {table} ("{c}");'))
                break

def run_sql(engine, sql: str) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(sql, conn)
