# core/llm_ops.py
from __future__ import annotations
import re
from typing import Any, Iterable, List, Union

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# ---------- LLM ----------
def build_llm(model: str, api_key: str, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 챗 모델 래퍼"""
    return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

# ---------- SQL Chain ----------
def _ensure_engine(engine_or_url: Union[str, Engine]) -> Engine:
    return create_engine(engine_or_url) if isinstance(engine_or_url, str) else engine_or_url

def _get_table_names(engine: Engine) -> List[str]:
    insp = inspect(engine)
    try:
        return insp.get_table_names()
    except Exception:
        return []

def build_sql_chain(llm: ChatOpenAI, db_url: str):
    """DB 스키마를 인스펙트하여 include_tables에 정확한 목록을 넣어 생성."""
    eng = create_engine(db_url)
    tables = _get_table_names(eng)
    db = SQLDatabase(eng, include_tables=tables) if tables else SQLDatabase(eng)
    return create_sql_query_chain(llm, db)

# ---------- SQL Guardrails ----------
_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.I | re.M)
_LABELS = re.compile(r"(?im)^(?:sql\s*query\s*:|sql\s*:|query\s*:|answer\s*:|final\s*sql\s*:)\s*")
_DML_DDL = re.compile(r"\b(update|delete|insert|drop|alter|truncate)\b", re.I)

def _first_select(sql: str) -> str:
    m = re.search(r"(?is)\bselect\b", sql)
    return sql[m.start():] if m else sql

def sanitize_sql(s: Any) -> str:
    """
    - 코드펜스/접두 라벨/주석 제거
    - 본문에서 첫 SELECT부터 취득
    - 작은따옴표로 감싼 식별자('col','table')를 큰따옴표로 교정
    """
    if s is None:
        return ""
    s = s if isinstance(s, str) else str(s)
    s = _FENCE.sub("", s).strip()
    s = _LABELS.sub("", s)
    s = re.sub(r"(?m)^\s*--.*$", "", s)     # 한 줄 주석 제거
    s = _first_select(s).strip()

    # FROM/JOIN 'table' -> "table"
    s = re.sub(r"(?i)\b(from|join)\s*'([\w\.]+)'", r'\1 "\2"', s)
    # 함수/리스트 내 'identifier' -> "identifier"
    s = re.sub(r"(?i)([\(\s,])'([A-Za-z_][\w$]*)'([\s,\)])", r'\1"\2"\3', s)
    # 혼합 따옴표 교정: "name' -> "name"
    s = re.sub(r'"([\w\.]+)\'', r'"\1"', s)
    return s.strip()

def enforce_limit(sql: str, default_limit: int = 1000) -> str:
    if re.search(r"(?i)\blimit\s+\d+\b", sql):
        return sql.rstrip(";") + ";"
    return sql.rstrip(";") + f" LIMIT {default_limit};"

def _schema_hint(engine: Engine, max_tables: int = 12, max_cols: int = 12) -> str:
    insp = inspect(engine)
    parts = []
    try:
        tables = insp.get_table_names()
        for t in tables[:max_tables]:
            cols = [c["name"] for c in insp.get_columns(t)[:max_cols]]
            parts.append(f'{t}({", ".join(cols)})')
    except Exception:
        pass
    return " | ".join(parts) if parts else "no tables"

def _extract_text_from_chain_output(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("result") or raw.get("query") or raw.get("text") or raw.get("sql") or ""
    return str(raw or "")

def generate_sql_from_nlq(
    sql_chain,
    nlq: str,
    engine_or_url: Union[str, Engine, None] = None,
    default_limit: int = 1000,
) -> str:
    """
    NLQ → SQL 생성
    - 엔진 전달 시 스키마 힌트를 질문에 주입하여 정확도 향상
    - 라벨/코드펜스/따옴표 오류 정리 + DML/DDL 차단 + LIMIT 강제
    """
    prompt_q = nlq
    if engine_or_url is not None:
        eng = _ensure_engine(engine_or_url)
        hint = _schema_hint(eng)
        prompt_q = (
            "규칙: 오직 SQL만 출력. 접두어/설명/코드펜스 금지. "
            "식별자는 가능하면 큰따옴표(\") 사용. LIMIT 포함.\n\n"
            f"[Schema]\n{hint}\n\n질문: {nlq}"
        )

    raw = sql_chain.invoke({"question": prompt_q})
    txt = _extract_text_from_chain_output(raw)
    sql = sanitize_sql(txt)
    if not sql:
        raise ValueError("SQL을 생성하지 못했습니다. 질문을 더 구체화해 주세요.")
    if _DML_DDL.search(sql):
        raise ValueError("Unsafe SQL verb detected (DML/DDL). 읽기 전용 쿼리만 허용됩니다.")
    return enforce_limit(sql, default_limit=default_limit)

# ---------- Answers (RAG/Chat) ----------
def _style_prefix(tone: str) -> str:
    return ("말투는 친근하고 공감 있게, 군더더기 없이 자연스럽게."
            if tone == "친근" else "말투는 단정하고 간결하게, 불필요한 수식은 피한다.")

def rag_answer(llm: ChatOpenAI, question: str, docs: Iterable[Any], tone: str = "전문") -> str:
    ctx = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    prompt = (
        "역할: 제조 공정 데이터 분석 파트너.\n"
        f"{_style_prefix(tone)}\n"
        "아래 컨텍스트(메타요약)를 근거로 한국어로 답하세요. 불확실하면 그렇게 밝혀라.\n\n"
        f"[컨텍스트]\n{ctx}\n\n[질문]\n{question}\n\n"
        "가능하면 수치를 구체적으로 제시하세요."
    )
    return llm.invoke(prompt).content

def chat_answer(llm: ChatOpenAI, question: str, tone: str = "전문") -> str:
    prompt = (
        "역할: 든든한 업무 파트너.\n"
        f"{_style_prefix(tone)}\n"
        "한국어로 간결하고 명확하게 답하라.\n\n"
        f"[질문]\n{question}"
    )
    return llm.invoke(prompt).content
