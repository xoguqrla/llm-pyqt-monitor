# core/agent.py
from __future__ import annotations
from sqlalchemy import inspect

from core.db_ops import run_sql
from core.llm_ops import generate_sql_from_nlq, rag_answer, chat_answer
from core.rag_ops import retrieve_meta
from core.hybrid import fuse_sql_and_rag

def build_schema_hint(engine, max_tables: int = 8, max_cols: int = 8) -> str:
    insp = inspect(engine)
    tables = insp.get_table_names()
    parts = []
    for t in tables[:max_tables]:
        cols = [c["name"] for c in insp.get_columns(t)[:max_cols]]
        parts.append(f'{t}({", ".join(cols)})')
    return " | ".join(parts) if parts else "no tables"

def classify_mode(llm, question: str, schema_hint: str) -> str:
    prompt = (
        "You are a router. Return exactly one token in {SQL|RAG|HYB|CHAT}.\n"
        "- SQL: numeric aggregation, filtering, trend/time-series, top-k, joins.\n"
        "- RAG: questions about schema/columns/availability/metadata summaries.\n"
        "- HYB: needs both numeric result and explanatory context.\n"
        "- CHAT: general conversation unrelated to data.\n\n"
        f"Schema: {schema_hint}\nQuestion: {question}\nAnswer:"
    )
    out = llm.invoke(prompt).content.strip().upper()
    for m in ("SQL", "RAG", "HYB", "CHAT"):
        if m in out:
            return m
    return "HYB"

def run_agent(llm, engine, chroma, sql_chain, question: str, tone: str = "전문"):
    schema = build_schema_hint(engine)
    mode = classify_mode(llm, question, schema)

    df, sql, docs = None, "", []

    if mode in ("SQL", "HYB"):
        try:
            sql = generate_sql_from_nlq(sql_chain, question, engine_or_url=engine)
            df = run_sql(engine, sql)
        except Exception:
            df, sql = None, ""

    if mode in ("RAG", "HYB"):
        docs = retrieve_meta(chroma, question, k=6)

    if mode == "SQL" and df is not None:
        return ("SQL", df, sql)

    if mode == "RAG":
        ans = rag_answer(llm, question, docs, tone=tone)
        return ("RAG", ans, None)

    if mode == "HYB":
        df_snip = df.head(20).to_csv(index=False) if df is not None else ""
        meta_snip = "\n\n".join(d.page_content for d in docs[:4]) if docs else ""
        final = fuse_sql_and_rag(llm, question, df_snip, meta_snip, tone=tone)
        return ("HYB", (df, sql, final), None)

    # CHAT (fallback)
    ans = chat_answer(llm, question, tone=tone)
    return ("CHAT", ans, None)
