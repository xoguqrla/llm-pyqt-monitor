# 🚀 LLM\_PyQt\_Platform

**LLM\_PyQt\_Platform**은 **실시간 데이터 분석, 시각화, 그리고 LLM 기반 질의응답**을 통합한 차세대 분석 플랫폼입니다.
PyQt 기반의 직관적인 GUI와 Python 생태계를 활용해 **데이터 업로드 → 저장 → 분석 → 시각화 → LLM 분석**까지 한 번에 처리할 수 있습니다.

---

## 🌐 프로젝트 비전 (Vision)

> **데이터 중심(Data-Driven) 의사결정**을 누구나 쉽게.
>
> LLM\_PyQt\_Platform은 복잡한 데이터 처리 과정을 하나의 GUI에 통합하여,
> 연구원·엔지니어·분석가가 **데이터의 흐름을 시각적으로 이해하고,
> LLM의 도움으로 인사이트를 즉시 얻을 수 있는 환경**을 제공합니다.

**핵심 가치:**

* **직관성** – 코드 없이 GUI로 데이터 탐색 및 분석
* **신뢰성** – DB·통계 기반 데이터 관리
* **확장성** – 모듈형 구조로 다양한 산업·연구 환경에 적용 가능

---

## 🏗️ 아키텍처 (Architecture)

```mermaid
flowchart TD
    A[사용자 GUI (PyQt5)] --> B[데이터 관리 모듈]
    B -->|CSV/DB 업로드| C[PostgreSQL]
    B -->|요약 통계/전처리| D[Analysis Visualizer]

    A --> E[LLM 분석 모듈]
    E -->|NLQ → SQL 변환| C
    E -->|RAG 질의응답| F[Vector DB (Chroma)]

    D --> A
    F --> A
    C --> A
```

* **GUI (PyQt5)** – 사용자가 모든 기능을 한눈에 조작할 수 있는 대시보드
* **데이터 관리 모듈** – CSV/DB 업로드, 자동 통계 요약, DB 연동
* **PostgreSQL** – 구조화된 데이터 저장소
* **Analysis Visualizer** – SQL 실행 결과/그래프를 GUI에 출력
* **LLM 분석 모듈** – NLQ→SQL 변환 및 RAG 기반 PDF/CSV 질의응답
* **Vector DB (Chroma)** – 문서 임베딩과 Evidence 기반 검색

---

## ✨ 주요 기능 (Features)

* **데이터 관리 (Data Management)**

  * CSV/DB 업로드 및 전처리
  * PostgreSQL 연동으로 안정적인 데이터 관리
  * 자동 통계 요약 (row count, mean, max, min, std 등)

* **데이터 분석 & 시각화 (Analysis & Visualization)**

  * PyQt 대시보드에서 표/그래프 실시간 출력
  * Matplotlib, Plotly 기반 고급 시각화
  * SQL 결과와 Evidence 패널 자동 연동

* **LLM 기반 분석 (LLM-Powered Analysis)**

  * LangChain + OpenAI API 연동
  * 자연어 질의(NLQ) → SQL 자동 변환
  * RAG 기반 PDF/CSV 문서 질의응답
  * Evidence 기반 구조화된 응답 제공

* **시뮬레이션 및 확장성 (Simulation & Extensibility)**

  * PyVista 기반 3D 데이터 시각화 및 시뮬레이터
  * 모듈형 구조로 다양한 분석 시나리오에 확장 가능

---

## 📚 기술 스택 (Tech Stack)

* **Frontend (GUI):** PyQt5, PyVista
* **Backend:** Python 3.11, FastAPI
* **Database:** PostgreSQL 17, Chroma (VectorDB)
* **AI/LLM:** OpenAI API, LangChain
* **Visualization:** Matplotlib, Plotly

---
