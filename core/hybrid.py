# core/hybrid.py
from __future__ import annotations

def route(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["최대","평균","합계","개수","count","group by","where","기간","분포","추세","top"]):
        return "SQL"
    if any(k in q for k in ["왜","원인","설명","해석","가이드","보고","요약"]):
        return "RAG"
    return "Hybrid"

# core/hybrid.py
def fuse_sql_and_rag(llm, question: str, df_csv_snip: str, meta_snip: str, tone: str = "전문"):
    style = ("말투는 친근하고 공감 있게, 너무 딱딱하지 않게."
             if tone=="친근" else "말투는 간결하고 단정하게.")
    prompt = (
        "역할: 제조 공정 데이터 분석 파트너.\n"
        f"{style}\n"
        "우선순위: 실측 데이터(SQL) > 메타요약(RAG). 상충하면 SQL을 따른다.\n"
        "출력: 한국어 Markdown, 섹션 3개\n"
        "1) 최종 답변(한 문단)\n2) 근거(불릿, SQL/메타 구분)\n3) 추가 확인 항목(3개)\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 결과 샘플 CSV]\n{df_csv_snip if df_csv_snip else '없음'}\n\n"
        f"[메타데이터 요약]\n{meta_snip}\n"
    )
    return llm.invoke(prompt).content

