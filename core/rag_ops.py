# core/rag_ops.py
from __future__ import annotations
from typing import List, Tuple, Any, Dict
import os, json, math, hashlib, re
from pathlib import Path
from dataclasses import dataclass

# ---------------------------
# Embeddings (간단 더미 임베딩)
# ---------------------------

class SimpleEmbedder:
    """
    외부 API 없이 동작하는 더미 임베딩(embedding).
    텍스트를 해시 기반 고정 길이 벡터로 변환.
    실제 검색 품질은 낮지만, 파이프라인 동작 확인에 충분.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            hv = hashlib.blake2b(t.encode("utf-8"), digest_size=32).digest()
            # 32바이트 → dim으로 반복 확장
            vec = []
            for i in range(self.dim):
                b = hv[i % len(hv)]
                # -0.5 ~ +0.5 범위의 값
                vec.append((b / 255.0) - 0.5)
            out.append(vec)
        return out

def build_embeddings(openai_key: str, embed_model: str) -> SimpleEmbedder:
    """
    (호환용) 임베딩 빌더.
    실제 OpenAI Embedding으로 바꾸려면 여기만 교체하면 됩니다.
    """
    _ = openai_key, embed_model
    return SimpleEmbedder(dim=384)

# --------------------------------
# Mini Vector Store (경량 RAG 백엔드)
# --------------------------------

@dataclass
class _Doc:
    id: str
    text: str
    metadata: dict

    # main 코드가 d.page_content를 참조하므로 호환 속성 제공
    @property
    def page_content(self):
        return self.text

class MiniVectorStore:
    """
    ids: str -> _Doc
    index: 단순한 bag-of-words 역색인 + 코사인 유사도(더미 임베딩) 혼합 점수
    """
    def __init__(self, embedder: SimpleEmbedder, persist_dir: str | Path):
        self.embedder = embedder
        self.persist_dir = str(persist_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._docs: Dict[str, _Doc] = {}           # id -> doc
        self._embs: Dict[str, List[float]] = {}    # id -> embedding
        self._bow: Dict[str, set] = {}             # token -> set(ids)
        self._load_if_exists()

    # ----- persistence -----
    def _persist_path(self) -> Path:
        return Path(self.persist_dir) / "mini_store.json"

    def _load_if_exists(self):
        p = self._persist_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            docs = data.get("docs", {})
            embs = data.get("embs", {})
            self._docs = {k: _Doc(id=k, text=v["text"], metadata=v.get("metadata", {})) for k, v in docs.items()}
            self._embs = {k: v for k, v in embs.items()}
            # 역색인 재구성
            self._bow = {}
            for doc_id, d in self._docs.items():
                for tok in self._tokenize(d.text):
                    self._bow.setdefault(tok, set()).add(doc_id)
        except Exception:
            # 깨졌으면 초기화
            self._docs, self._embs, self._bow = {}, {}, {}

    def _flush(self):
        p = self._persist_path()
        obj = {
            "docs": {k: {"text": v.text, "metadata": v.metadata} for k, v in self._docs.items()},
            "embs": self._embs,
        }
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- utility -----
    @staticmethod
    def _tokenize(s: str) -> List[str]:
        # 단순 토크나이저: 영숫자 토큰
        return [t for t in re.findall(r"[A-Za-z0-9_]+", s.lower()) if t]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ----- public methods -----
    def add(self, ids: List[str], texts: List[str], metadatas: List[dict]):
        embs = self.embedder.embed(texts)
        for i, _id in enumerate(ids):
            doc = _Doc(id=_id, text=texts[i], metadata=metadatas[i] if i < len(metadatas) else {})
            self._docs[_id] = doc
            self._embs[_id] = embs[i]
            for tok in self._tokenize(texts[i]):
                self._bow.setdefault(tok, set()).add(_id)
        self._flush()

    def delete(self, ids: List[str]):
        for _id in ids:
            if _id in self._docs:
                text = self._docs[_id].text
                for tok in self._tokenize(text):
                    if tok in self._bow and _id in self._bow[tok]:
                        self._bow[tok].remove(_id)
                        if not self._bow[tok]:
                            del self._bow[tok]
                del self._docs[_id]
            if _id in self._embs:
                del self._embs[_id]
        self._flush()

    def query(self, query_text: str, top_k: int = 5) -> List[_Doc]:
        # 키워드 매칭 + 임베딩 코사인 혼합
        query_tokens = set(self._tokenize(query_text))
        cand_ids = set()
        for tok in query_tokens:
            cand_ids |= self._bow.get(tok, set())
        # 키워드로 후보가 없으면 전체 대상으로(소규모라 가능)
        if not cand_ids:
            cand_ids = set(self._docs.keys())

        q_emb = self.embedder.embed([query_text])[0]
        scored = []
        for _id in cand_ids:
            doc = self._docs[_id]
            # 키워드 점수
            dtoks = set(self._tokenize(doc.text))
            keyword_score = len(query_tokens & dtoks) / (len(query_tokens) + 1e-9)
            # 임베딩 점수
            emb_score = self._cosine(q_emb, self._embs.get(_id, []))
            # 혼합
            score = 0.4 * keyword_score + 0.6 * emb_score
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:max(1, top_k)]]

# --------------------------------
# Public RAG API (main에서 사용하는 함수들)
# --------------------------------

def build_chroma(embedder: SimpleEmbedder, persist_dir: str | Path) -> MiniVectorStore:
    """
    (호환용) 벡터 DB 빌더. 실제 ChromaDB로 교체 가능.
    """
    return MiniVectorStore(embedder, persist_dir)

# ---- 메타 유연 처리: 'file' / 'columns' 미존재 시 fallback 허용 ----

def _meta_filename(meta: dict) -> str:
    return meta.get("file") or meta.get("file_name") or meta.get("file_path") or "unknown_file"

def _ensure_columns(meta: dict) -> dict:
    cols = meta.get("columns")
    if isinstance(cols, dict):
        return cols
    # 최소 형태라도 반환
    return {}

def meta_to_text(meta: dict) -> str:
    """
    메타 전체를 one-shot 텍스트(plain)로 변환 (RAG 색인용).
    """
    fname = _meta_filename(meta)
    rows = meta.get("rows")
    cols_cnt = meta.get("cols")
    lines = [f"file:{fname} rows:{rows} cols:{cols_cnt}"]
    columns = _ensure_columns(meta)
    for k, v in columns.items():
        dtype = v.get("dtype")
        non_null = v.get("non_null")
        nulls = v.get("nulls")
        s = f"column:{k} dtype:{dtype} non_null:{non_null} nulls:{nulls}"
        st = v.get("stats")
        if st:
            s += f" stats(min:{st.get('min')}, max:{st.get('max')}, mean:{st.get('mean')})"
        lines.append(s)
    # process_metrics / process_stability_score도 함께 요약(있으면)
    pm = meta.get("process_metrics") or {}
    if pm:
        lines.append("process_metrics:" + json.dumps(pm, ensure_ascii=False))
    pss = meta.get("process_stability_score") or {}
    if pss:
        lines.append("process_stability_score:" + json.dumps(pss, ensure_ascii=False))
    return "\n".join(lines)

def build_embedding_texts_from_meta(meta: dict) -> List[Tuple[str, dict]]:
    """
    메타로부터 여러 '텍스트+메타데이터' 조각 생성.
    유사도 및 안정성 점수도 자연어 문장으로 변환하여 포함.
    반환 형식: List[(text, metadata)]
    """
    out: List[Tuple[str, dict]] = []
    fname = _meta_filename(meta)
    rows = meta.get("rows")
    cols_cnt = meta.get("cols")

    # 파일 요약
    summary_text = f"File '{fname}' has {rows} rows and {cols_cnt} columns."
    out.append((summary_text, {"chunk_type": "file_summary", "file": fname}))

    # 컬럼별 정보
    columns = _ensure_columns(meta)
    for col, v in columns.items():
        dtype = v.get("dtype")
        line = f"Column '{col}' has data type {dtype}."
        st = v.get("stats")
        if st:
            line += f" Basic stats: min={st.get('min', 'N/A')}, max={st.get('max', 'N/A')}, mean={st.get('mean', 'N/A')}."
        out.append((line, {"chunk_type": "column_details", "file": fname, "column": col}))

    # 공정 메트릭 정보
    metrics = meta.get("process_metrics", {})
    if metrics:
        metric_summary = f"Process metrics for {fname}: "
        metric_parts = []
        if "utilization_ratio" in metrics:
            metric_parts.append(f"utilization is {metrics['utilization_ratio']:.2%}")
        if "energy_input_total" in metrics:
            metric_parts.append(f"total energy input is {metrics['energy_input_total']:.1f}")
        if "cum_volume_est" in metrics:
            metric_parts.append(f"estimated volume is {metrics['cum_volume_est']:.2f}")
        metric_summary += ", ".join(metric_parts) + "."
        out.append((metric_summary, {"chunk_type": "process_metrics", "file": fname}))

    # 유사도 점수 정보
    sim_scores = meta.get("process_similarity_score", {})
    if sim_scores and "final_similarity_score" in sim_scores:
        score = sim_scores["final_similarity_score"]
        sim_text = f"The process similarity score compared to the golden standard is {score} out of 100."
        out.append((sim_text, {"chunk_type": "similarity_score", "file": fname, "score": score}))

    # 안정성 점수 정보
    stab_scores = meta.get("process_stability_score", {})
    if stab_scores and "final_score" in stab_scores:
        score = stab_scores["final_score"]
        temp_stab = stab_scores.get("temperature_stability")
        load_stab = stab_scores.get("load_stability")
        stab_text = f"The process stability score is {score} out of 100, with temperature stability at {temp_stab} and load stability at {load_stab}."
        out.append((stab_text, {"chunk_type": "stability_score", "file": fname, "score": score}))

    return out

def upsert_texts(chroma: MiniVectorStore, file_id: str, texts_and_meta: List[Tuple[str, dict]]) -> None:
    """
    주어진 file_id에 대해 텍스트들을 ID 규칙(file_id:0000)으로 저장/갱신.
    main에서 delete 시 f"{fid}:{i:04d}" 패턴을 쓰므로 동일 규칙 준수.
    """
    ids, texts, metas = [], [], []
    for i, (txt, m) in enumerate(texts_and_meta):
        ids.append(f"{file_id}:{i:04d}")
        texts.append(txt)
        metas.append(m or {})
    # 덮어쓰기 위해 먼저 같은 id들 삭제 후 추가
    chroma.delete(ids)
    chroma.add(ids=ids, texts=texts, metadatas=metas)

def retrieve_meta(chroma: MiniVectorStore, query: str, top_k: int = 5) -> List[_Doc]:
    """
    질의(query)에 가장 관련 있는 문서 조각 상위 top_k 반환.
    반환 원소는 d.page_content 텍스트 접근 가능.
    """
    return chroma.query(query_text=query, top_k=top_k)
