# main10_pdf_rag_modified8.py
# 최종 업데이트 : 2025-08-20
# PyQt5 기반 공정 데이터 LLM 분석기 (V4.0)
# - V3.1 기능 전체 포함
# - [신규] 3번째 탭: PDF 문서 대상 RAG 챗봇 기능 추가
# - 문서 전용 벡터DB 분리(vector_db_dir/docs), 드래그&드롭/진행률/취소, 삭제, Evidence/Images 패널
# - PDF 텍스트를 DB에 저장 후, DB에서 정확 텍스트를 꺼내 컨텍스트 구성 (벡터 검색 + LIKE 백업)
# - LLM 기반 CSV 분석 답변도 섹션/불릿으로 구조화해 깔끔하게 출력

from __future__ import annotations
import sys, traceback, html
from pathlib import Path
import json
import os
from typing import List, Tuple

import pandas as pd
import pyvista as pv

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem, QSlider, QStackedWidget,
    QSplitter
)
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor

# --- PDF RAG 기능 추가에 필요한 라이브러리 ---
try:
    import fitz  # PyMuPDF (텍스트+이미지 추출 고품질)
except Exception:
    fitz = None
from pypdf import PdfReader  # 폴백 텍스트 추출
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- core modules ---
from core.config import get_settings
from core.csv_ops import load_and_meta
from core.db_ops import make_engine, ingest_df, ensure_indexes, run_sql, table_name_from_file
from core.rag_ops import (
    build_embeddings, build_chroma, retrieve_meta,
    build_embedding_texts_from_meta, upsert_texts
)
from core.llm_ops import build_llm, build_sql_chain, generate_sql_from_nlq
from core.plotting import df_to_table, plot_df_line
from core.files_registry import upsert_entry, load_registry
from core2.analysis_visualizer import AnalysisVisualizer

# --- pdf_store: 모듈이 없으면 폴백 함수 정의 ---
try:
    from core2.pdf_store import (
    ensure_pdf_tables, insert_pdf, insert_chunks,
    list_chunk_ids_by_doc, fetch_chunks_by_ids, delete_doc, keyword_search_chunks,
    list_all_docs,   # ← 추가
)

except Exception:
    # 폴백: 이 파일 하나만으로 동작하도록 최소 구현
    def ensure_pdf_tables(engine) -> None:
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
          FOREIGN KEY (doc_id) REFERENCES pdf_docs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_doc ON pdf_chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_page ON pdf_chunks(page);
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text ON pdf_chunks(text);
        """
        with engine.begin() as c:
            c.exec_driver_sql(DDL_DOCS)
            c.exec_driver_sql(DDL_CHUNKS)

    def insert_pdf(engine, filename: str, pages: int) -> int:
        from datetime import datetime
        with engine.begin() as c:
            c.exec_driver_sql(
                "INSERT OR IGNORE INTO pdf_docs(filename,pages,created_at) VALUES(?,?,?)",
                (filename, pages, datetime.utcnow().isoformat()),
            )
            row = c.exec_driver_sql("SELECT id FROM pdf_docs WHERE filename=?", (filename,)).first()
        return int(row[0])

    def insert_chunks(engine, doc_id: int, rows: List[Tuple[int,int,str]]) -> List[int]:
        ids: List[int] = []
        with engine.begin() as c:
            for page, idx, text in rows:
                c.exec_driver_sql(
                    "INSERT INTO pdf_chunks(doc_id,page,chunk_index,text,token_len) VALUES(?,?,?,?,?)",
                    (doc_id, page, idx, text, len(text)),
                )
                rid = c.exec_driver_sql("SELECT last_insert_rowid()").scalar()
                ids.append(int(rid))
        return ids

    def list_chunk_ids_by_doc(engine, doc_id: int) -> List[int]:
        with engine.begin() as c:
            rows = c.exec_driver_sql("SELECT id FROM pdf_chunks WHERE doc_id=? ORDER BY id", (doc_id,)).fetchall()
        return [int(r[0]) for r in rows]

    def fetch_chunks_by_ids(engine, ids: List[int]) -> List[str]:
        if not ids: return []
        qmarks = ",".join("?" for _ in ids)
        with engine.begin() as c:
            rows = c.exec_driver_sql(f"SELECT id, text FROM pdf_chunks WHERE id IN ({qmarks})", tuple(ids)).fetchall()
        m = {int(i): t for i, t in rows}
        return [m.get(i, "") for i in ids]

    def delete_doc(engine, doc_id: int) -> None:
        with engine.begin() as c:
            c.exec_driver_sql("DELETE FROM pdf_chunks WHERE doc_id=?", (doc_id,))
            c.exec_driver_sql("DELETE FROM pdf_docs   WHERE id=?", (doc_id,))

    def keyword_search_chunks(engine, query: str, limit: int = 6):
        import re
        tokens = [t for t in re.findall(r"[A-Za-z0-9가-힣]{2,}", query) if len(t) >= 2][:5]
        if not tokens:
            tokens = [query.strip()][:1]
        like = " OR ".join(["text LIKE ?"] * len(tokens))
        params = [f"%{t}%" for t in tokens]
        sql = f"SELECT id, text FROM pdf_chunks WHERE {like} ORDER BY token_len LIMIT {limit}"
        with engine.begin() as c:
            rows = c.exec_driver_sql(sql, tuple(params)).fetchall()
        return [(int(r[0]), r[1]) for r in rows]

# --- optional: metadata build & indexing scripts ---
try:
    from scripts.build_metadata import build_for_table as _build_meta_for_table
except Exception:
    _build_meta_for_table = None
try:
    from scripts.index_metadata import index_for_sessions as _index_sessions
except Exception:
    _index_sessions = None

# --------- [대화 로그/메모리 상수] ---------
HISTORY_PATH = "history.json"
MAX_HISTORY_TURNS = 3
MAX_HISTORY_RECORDS = 50

# ---------------- global excepthook ----------------
def _excepthook(et, ev, tb):
    msg = "".join(traceback.format_exception(et, ev, tb))[-4000:]
    print(msg, file=sys.stderr)
    try:
        QMessageBox.critical(None, "Unhandled Error", msg)
    except Exception:
        pass
sys.excepthook = _excepthook

# ---------------- threading helper ----------------
class Worker(QObject):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int, str)  # (index, message)

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn, self.a, self.kw = fn, a, kw

    def run(self):
        try:
            if "progress_callback" in self.kw:
                self.kw["progress_callback"] = self.progress.emit
            self.finished.emit(self.fn(*self.a, **self.kw), None)
        except Exception as e:
            self.finished.emit(None, e)

def run_in_thread(parent, fn, cb, *a, **kw):
    th = QThread(parent)
    wk = Worker(fn, *a, **kw)
    wk.moveToThread(th)
    def _done(r, e):
        try:
            cb(r, e)
        finally:
            th.quit()
            wk.deleteLater()
            th.deleteLater()
    wk.finished.connect(_done)
    th.started.connect(wk.run)
    th.start()
    return wk, th


# ---------------- drag & drop ----------------
class DropArea(QFrame):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None, file_type="CSV"):
        super().__init__(parent)
        self.file_type = file_type.lower()
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setStyleSheet("""
        QFrame { border: 2px dashed #9ca3af; border-radius: 10px; background: #fafafa; color:#374151; }
        QFrame[drag='true'] { border-color:#2563eb; background:#eef2ff; }
        """)
        lay = QVBoxLayout(self)
        lab = QLabel(f"📥 여기에 {file_type.upper()} 파일을 드래그 & 드롭")
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("font-weight:600;")
        lay.addWidget(lab)

    def dragEnterEvent(self, e):
        ok = any(u.isLocalFile() and u.toLocalFile().lower().endswith(f".{self.file_type}") for u in e.mimeData().urls())
        if ok:
            self.setProperty("drag", True)
            self.style().unpolish(self); self.style().polish(self)
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.setProperty("drag", False)
        self.style().unpolish(self); self.style().polish(self)
        super().dragLeaveEvent(e)

    def dropEvent(self, e):
        self.setProperty("drag", False)
        self.style().unpolish(self); self.style().polish(self)
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.isLocalFile() and u.toLocalFile().lower().endswith(f".{self.file_type}")]
        if paths:
            self.filesDropped.emit(paths)
        e.acceptProposedAction()


# ---------------- chat bubbles ----------------
class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self.setWidget(self._container)
        self.vbox = QVBoxLayout(self._container)
        self.vbox.setSpacing(8); self.vbox.setContentsMargins(8, 8, 8, 8)
        self._top_spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vbox.addItem(self._top_spacer)
        self._user_style = "QFrame {background:#f3f4f6; border-radius:12px; padding:8px 10px;} QLabel {color:#111827; font-size:13px;}"
        self._bot_style  = "QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;} QLabel {color:#0f5132; font-size:13px;}"
        self._container.installEventFilter(self)

    def _scroll_to_bottom_later(self):
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))

    def _bubble(self, text: str, role: str) -> QWidget:
        safe = html.escape(text).replace("\n", "<br>")
        lab = QLabel(safe); lab.setWordWrap(True); lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        fr = QFrame(); fr.setStyleSheet(self._bot_style if role == "bot" else self._user_style)
        fl = QHBoxLayout(fr); fl.setContentsMargins(10, 6, 10, 6); fl.addWidget(lab)
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        if role == "bot": hl.addStretch(); hl.addWidget(fr)
        else: hl.addWidget(fr); hl.addStretch()
        return row

    def add_user(self, text: str):
        self.vbox.addWidget(self._bubble(text, "user"))
        self._scroll_to_bottom_later()

    def add_bot(self, text: str):
        self.vbox.addWidget(self._bubble(text, "bot"))
        self._scroll_to_bottom_later()

    def clear(self):
        while self.vbox.count() > 1:
            item = self.vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._scroll_to_bottom_later()

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj is self._container and event.type() in (QEvent.LayoutRequest, QEvent.Show):
            self._scroll_to_bottom_later()
        return super().eventFilter(obj, event)


# ---------------- LLM prompt helpers (CSV/DB용, PDF용) ----------------
def _tone_style(tone: str) -> str:
    return (
        "말투는 친근하고 공감 있게, 군더더기 없이 자연스럽게."
        if tone == "친근"
        else "말투는 단정하고 간결하며, 근거 중심으로 정확하게 설명한다. 불필요한 수식은 피한다."
    )

def llm_final_only(llm, question: str, df_snip: str, meta_snip: str, tone: str) -> str:
    """
    CSV/DB(+메타) 기반 질의의 구조화 응답.
    - 자료가 전혀 없을 때만 간단한 불가 사유.
    - 표/메타가 조금이라도 있으면 [요약/핵심 결과/간단 해석/제한/추천 후속] 섹션으로 출력.
    - 영어 데이터라도 생각은 영어로, 출력은 항상 한국어. 용어/약어/코드/단위는 원문 보존(최초 1회 (원문) 병기 권장).
    """
    has_any = bool(df_snip and df_snip.strip()) or bool(meta_snip and meta_snip.strip())
    if not has_any:
        return "제공된 자료가 없어 확답할 수 없습니다. 표/요약 등 최소한의 근거를 제공해 주세요."
    prompt = (
        "역할: 제조 공정 데이터 분석 파트너.\n"
        f"{_tone_style(tone)}\n"
        "언어 규칙:\n"
        "- 데이터/문서가 영어여도 이해·추론은 영어 맥락을 보존하되, 최종 출력은 반드시 한국어로 작성한다.\n"
        "- 기술 용어·제품명·약어·코드·단위·수치는 원문을 보존하고, 최초 1회만 한국어 뒤에 (원문)을 병기한다.\n"
        "출력 형식(아래 템플릿 그대로):\n"
        "[요약]\n"
        "• 질문에 대한 핵심 결론을 1~2문장으로.\n\n"
        "[핵심 결과]\n"
        "• 표/통계에서 직접 관찰 가능한 결과 3~5개(수치/범위/추세 언급).\n\n"
        "[간단 해석]\n"
        "1. 결과가 의미하는 바를 한 문장씩 2~3개.\n\n"
        "[제한/주의]\n"
        "• 데이터 품질/가정/범위의 제약을 1~3개.\n\n"
        "[추천 후속 분석]\n"
        "• 바로 실행 가능한 다음 단계 2~3개.\n\n"
        "규칙: 섹션 제목과 불릿/번호만 사용. HTML/Markdown/이모지 금지. 제공 자료 밖의 추정 금지.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n"
    )
    return llm.invoke(prompt).content

def llm_checks_only(llm, question: str, df_snip: str, meta_snip: str) -> str:
    """실무 점검 체크리스트(3~6개, 불릿만)."""
    prompt = (
        "역할: 제조 공정 데이터 분석 점검관.\n"
        "언어 규칙:\n"
        "- 자료가 영어여도 이해는 영어 맥락으로 하되, 결과는 한국어로 작성한다.\n"
        "- 기술 용어·제품명·약어·코드·단위·수치는 원문을 보존하고, 최초 1회만 한국어 뒤 (원문)을 병기한다.\n"
        "출력 규칙:\n"
        "- 하이픈('- ') 불릿 리스트로 3~6개 항목을 제시한다. 머리말/제목/이모지/Markdown 금지.\n"
        "- 각 항목은 1~2문장, 데이터/메타를 근거로 실행 가능한 검증 방법을 제안한다.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n\n"
        "- …\n- …\n- …"
    )
    return llm.invoke(prompt).content

def llm_pdf_rag_answer(llm, question: str, context: str) -> str:
    """
    PDF RAG 구조화 응답(영문 문서도 고려): [요약/상세 설명/관련 정보·제한/출처]
    - 최종 출력은 한국어, 용어·약어·단위·코드는 원문 보존(최초 1회 (원문) 병기 권장)
    - 컨텍스트의 '[파일명 | p.N]' 라인에서 출처를 수집해 중복 제거
    """
    prompt = (
        "역할: 기술 문서를 기반으로 질문에 답변하는 전문 분석가.\n"
        "언어 규칙:\n"
        "1) 문서는 영어일 수 있으나, 이해/추론은 원문 맥락으로 유지하고 최종 출력은 반드시 한국어로 작성한다.\n"
        "2) 기술 용어·제품명·약어·코드·단위·수치는 번역하지 말고 원문을 보존한다(최초 1회 한국어 뒤 (원문) 병기 권장).\n"
        "3) 컨텍스트에 없는 내용은 적지 않는다.\n"
        "출력 템플릿(그대로 사용):\n"
        "[요약]\n"
        "…한 줄 요약…\n\n"
        "[상세 설명]\n"
        "1. …\n"
        "2. …\n"
        "3. …\n\n"
        "[관련 정보 / 제한 사항]\n"
        "• …  (없으면 '해당 없음')\n\n"
        "[출처]\n"
        "• 파일명 | p.N\n"
        "• 파일명 | p.N\n\n"
        f"[사용자 질문]\n{question}\n\n"
        f"[참고 문서 내용 (컨텍스트)]\n{context}\n"
    )
    return llm.invoke(prompt).content


# ---------------- main window ----------------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def __init__(self):
        super().__init__()
        self.history: List[Tuple[str, str]] = []

        # Deep report state and current DataFrame for visualizations
        self.in_deep_report: bool = False
        self.deep_report_inputs: List[str] = []
        self.current_df: pd.DataFrame | None = None
        # For deep report: allow multiple datasets to be loaded
        self.current_dfs: List[Tuple[pd.DataFrame, str]] = []

        # 시뮬 상태/타이머
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setTimerType(Qt.PreciseTimer)
        self.simulation_timer.timeout.connect(self._update_simulation_frame)
        self.sim_handles = None
        self.sim_frame_index = 0
        self.active_plotter: QtInteractor | None = None  # 현재 렌더 대상(일반/통합)

        # PDF RAG 상태
        self.pdf_rag_history: List[Tuple[str, str]] = []
        self.pdf_rag_files = {}          # {filename: doc_id}
        self.pdf_chunk_counts = {}       # {filename: n_chunks}
        self.pdf_images_map = {}         # {filename: [image_path, ...]}

        self.setupUi()
        self.init_backend()
        self.load_history()
        self.repopulate_chat()

    def setupUi(self):
        self.setWindowTitle("공정 데이터 LLM 분석기 V4.0 (PDF RAG 추가)")
        self.resize(1700, 900)

        main_layout = QHBoxLayout(self)
        self.tab_widget = QTabWidget(); main_layout.addWidget(self.tab_widget)

        # 1. LLM 탭
        self.llm_tab = QWidget(); self.tab_widget.addTab(self.llm_tab, "LLM 기반 CSV 분석")
        self.setup_llm_tab()

        # 2. 시각화 탭
        self.viz_tab = QWidget(); self.tab_widget.addTab(self.viz_tab, "데이터 시각화 분석")
        self.setup_viz_tab()

        # 3. PDF RAG 챗봇 탭 (신규)
        self.pdf_tab = QWidget(); self.tab_widget.addTab(self.pdf_tab, "📄 PDF 문서 챗봇")
        self.setup_pdf_tab()

        # 공유 상태
        self.csv_files, self.file_ids = [], {}
        self.last_df, self.df_viz = None, None
        self.visualizer, self.viz_context = None, None

        # Flag to track whether an advanced visualization was shown for a query
        self._advanced_triggered = False

    # app/main10_pdf_rag.py 파일에서 이 함수를 찾아 통째로 교체하세요.

# app/main10_pdf_rag.py 파일에서 이 함수를 찾아 통째로 교체하세요.

    def setup_llm_tab(self):
        """
        Construct the UI for the LLM-based CSV analysis page.  This page consists of
        three primary regions: a file management panel on the left, a chat/input
        panel in the middle, and a results/visualization panel on the right.  A
        QSplitter is used between the middle and right regions so the user can
        interactively resize the chat area versus the results area.  The left
        panel maintains a fixed width relative to the splitter contents.
        """
        # Main horizontal layout for the tab
        layout = QHBoxLayout(self.llm_tab)
        # Create individual layouts for left, center, and right sections
        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # ----------------------------------------------------------------------
        # Left Panel: file management (CSV selection, upload, delete)
        # ----------------------------------------------------------------------
        # Top label
        left_layout.addWidget(QLabel("📁 소스 파일 (RAG 및 SQL 대상)"))

        # (위) 새로 추가 섹션
        box_new = QFrame(); box_new.setFrameShape(QFrame.StyledPanel)
        ln = QVBoxLayout(box_new); ln.setContentsMargins(8,8,8,8)
        title_new = QLabel("➕ 새로 추가(이번 세션)")
        title_new.setStyleSheet("font-weight:600;")
        ln.addWidget(title_new)

        self.drop_csv = DropArea(file_type="csv")
        self.drop_csv.filesDropped.connect(self.handle_csv_paths)
        ln.addWidget(self.drop_csv)

        btn_row_new = QHBoxLayout()
        self.btn_upload_csv = QPushButton("파일 선택…")
        self.btn_upload_csv.clicked.connect(self.on_upload)
        self.btn_del_csv_new = QPushButton("선택 삭제")
        self.btn_del_csv_new.clicked.connect(lambda: self.on_delete_files(target="new"))
        btn_row_new.addWidget(self.btn_upload_csv)
        btn_row_new.addStretch(1)
        btn_row_new.addWidget(self.btn_del_csv_new)
        ln.addLayout(btn_row_new)

        self.csv_new_list = QListWidget() # <-- 에러가 발생했던 'csv_new_list'가 여기서 생성됩니다.
        self.csv_new_list.setToolTip("이번 세션에서 추가한 CSV 파일 목록")
        ln.addWidget(self.csv_new_list, 1)
        left_layout.addWidget(box_new)

        # (아래) 저장됨 섹션
        box_saved = QFrame(); box_saved.setFrameShape(QFrame.StyledPanel)
        ls = QVBoxLayout(box_saved); ls.setContentsMargins(8,8,8,8)
        title_saved = QLabel("📚 저장된 파일(로컬 DB)")
        title_saved.setStyleSheet("font-weight:600;")
        ls.addWidget(title_saved)

        btn_row_saved = QHBoxLayout()
        self.btn_refresh_csv_saved = QPushButton("새로고침")
        self.btn_refresh_csv_saved.clicked.connect(self.refresh_csv_saved_list)
        self.btn_del_csv_saved = QPushButton("선택 삭제")
        self.btn_del_csv_saved.clicked.connect(lambda: self.on_delete_files(target="saved"))
        btn_row_saved.addWidget(self.btn_refresh_csv_saved)
        btn_row_saved.addStretch(1)
        btn_row_saved.addWidget(self.btn_del_csv_saved)
        ls.addLayout(btn_row_saved)

        self.csv_saved_list = QListWidget() # <-- 저장된 파일 목록 위젯
        self.csv_saved_list.setToolTip("DB에 저장된 모든 CSV 파일")
        ls.addWidget(self.csv_saved_list, 1)
        left_layout.addWidget(box_saved, 1)

        # ----------------------------------------------------------------------
        # Centre Panel: chat and prompt input
        # ----------------------------------------------------------------------
        center_layout.addWidget(QLabel("💬 LLM 질의"))
        tone_row = QHBoxLayout(); tone_row.addWidget(QLabel("톤"))
        self.tone = QComboBox(); self.tone.addItems(["전문", "친근"]); tone_row.addWidget(self.tone)
        tone_row.addStretch(1); center_layout.addLayout(tone_row)
        self.chat = ChatView(); center_layout.addWidget(self.chat, 1)
        self.btn_clear_history = QPushButton("채팅 로그 초기화"); self.btn_clear_history.clicked.connect(self.on_clear_history); center_layout.addWidget(self.btn_clear_history)
        send_row = QHBoxLayout(); self.inp = QLineEdit(); self.inp.setPlaceholderText("질문을 입력하고 Enter…"); self.inp.returnPressed.connect(self.on_ask)
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask); self.status = QLabel("")
        send_row.addWidget(self.inp, 1); send_row.addWidget(self.btn_send); send_row.addWidget(self.status)
        center_layout.addLayout(send_row)

        # ----------------------------------------------------------------------
        # Right Panel: results and visualizations
        # ----------------------------------------------------------------------
        right_layout.addWidget(QLabel("📊 LLM 결과/리포트"))
        # 결과/리포트 탭 영역
        self.tabs = QTabWidget(); right_layout.addWidget(self.tabs, 1)
        # (1) 표: SQL 결과를 보여주는 테이블
        self.tbl = QTableWidget()
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl, "표(Table)")
        # (2) 기본 차트: SQL 결과를 간단한 라인 차트로 표시
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        # Allow the chart canvas to expand within its container for better resizing
        try:
            from PyQt5.QtWidgets import QSizePolicy
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        self.tabs.addTab(self.canvas, "그래프(Chart)")
        # (3) Evidence: SQL 및 RAG 근거, 프리뷰 출력
        self.evidence = QTextEdit()
        self.evidence.setReadOnly(True)
        self.tabs.addTab(self.evidence, "근거(Evidence)")
        # (4) 보고서: 기본 보고서 + 심층 리포트 요청 버튼을 포함하는 탭
        #    생성 시 보고서를 포함할 위젯과 레이아웃을 구성
        self.report_tab = QWidget()
        report_layout = QVBoxLayout(self.report_tab)
        # 심층 리포트 요청 버튼
        self.btn_deep_report = QPushButton("심층 리포트 요청")
        self.btn_deep_report.clicked.connect(self.start_deep_report)
        report_layout.addWidget(self.btn_deep_report)
        # 실제 보고서 텍스트 영역
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        report_layout.addWidget(self.report)
        self.tabs.addTab(self.report_tab, "보고서(Report)")
        # (5) 고급 시각화: 2페이지의 분석 결과를 1페이지에서도 볼 수 있도록 하는 탭
        self.adv_fig = plt.figure()
        self.adv_canvas = FigureCanvas(self.adv_fig)
        # Allow the advanced visualization canvas to expand for better resizing
        try:
            from PyQt5.QtWidgets import QSizePolicy
            self.adv_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        self.adv_tab_index = self.tabs.addTab(self.adv_canvas, "시각화 결과")

        # (6) 전문가 코멘트: 도메인 지식 메모장 탭
        self.expert_tab = QWidget()
        expert_layout = QVBoxLayout(self.expert_tab)
        # 입력 영역 (여러 줄)
        self.expert_input = QTextEdit()
        self.expert_input.setPlaceholderText("전문가의 코멘트를 입력하세요...")
        expert_layout.addWidget(self.expert_input)
        # 저장 버튼
        self.btn_save_expert = QPushButton("저장")
        self.btn_save_expert.clicked.connect(self.on_save_expert_comment)
        expert_layout.addWidget(self.btn_save_expert)
        # 스크롤 영역 내 컨테이너 (저장된 코멘트를 나열)
        self.expert_scroll = QScrollArea()
        self.expert_scroll.setWidgetResizable(True)
        self.expert_container = QWidget()
        self.expert_container_layout = QVBoxLayout(self.expert_container)
        self.expert_container_layout.setContentsMargins(0, 0, 0, 0)
        self.expert_container_layout.setSpacing(4)
        self.expert_scroll.setWidget(self.expert_container)
        expert_layout.addWidget(self.expert_scroll, 1)
        # 탭 추가
        self.expert_tab_index = self.tabs.addTab(self.expert_tab, "전문가 코멘트")
        # Populate expert comments from DB
        QTimer.singleShot(0, self.load_expert_comments)

        # 시작 시 저장된 CSV 목록 채우기
        QTimer.singleShot(0, self.refresh_csv_saved_list)

        # 더블클릭으로 저장된 CSV 로딩 기능 연결
        try:
            self.csv_saved_list.itemDoubleClicked.connect(self.on_load_saved_csv)
        except Exception:
            pass

        # ----------------------------------------------------------------------
        # Combine centre and right layouts into widgets and wrap in a splitter
        # ----------------------------------------------------------------------
        center_widget = QWidget(); center_widget.setLayout(center_layout)
        right_widget = QWidget(); right_widget.setLayout(right_layout)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        # Set stretch factors so the centre and right panels share space proportionally
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        # Add left panel and splitter to the main layout
        left_widget = QWidget(); left_widget.setLayout(left_layout)
        # Ensure left panel doesn't collapse; give it a minimum width
        left_widget.setMinimumWidth(260)
        layout.addWidget(left_widget)
        layout.addWidget(splitter, 1)

    def setup_viz_tab(self):
        layout = QVBoxLayout(self.viz_tab)

        # 컨트롤 패널
        control_panel = QFrame(); control_panel.setFixedHeight(60)
        control_layout = QHBoxLayout(control_panel)

        # 실행 버튼
        self.btn_load_csv_viz = QPushButton("CSV 파일 불러오기")
        self.btn_run_stability   = QPushButton("1. 안정성 분석");      self.btn_run_stability.setEnabled(False)
        self.btn_run_correlation = QPushButton("2. 상관관계 대시보드"); self.btn_run_correlation.setEnabled(False)
        self.btn_run_3d_path     = QPushButton("3. 3D 경로 (정적)");   self.btn_run_3d_path.setEnabled(False)
        self.btn_run_simulation  = QPushButton("4. 공정 시뮬레이션");   self.btn_run_simulation.setEnabled(False)
        self.btn_run_integrated  = QPushButton("5. 통합 대시보드");     self.btn_run_integrated.setEnabled(False)
        self.btn_run_aw_volume   = QPushButton("6. A*W 적층두께/부피"); self.btn_run_aw_volume.setEnabled(False)
        self.btn_ask_llm_about_viz = QPushButton("🤖 이 분석에 대해 질문하기"); self.btn_ask_llm_about_viz.setEnabled(False)

        control_layout.addWidget(self.btn_load_csv_viz)
        control_layout.addWidget(self.btn_run_stability)
        control_layout.addWidget(self.btn_run_correlation)
        control_layout.addWidget(self.btn_run_3d_path)
        control_layout.addWidget(self.btn_run_simulation)
        control_layout.addWidget(self.btn_run_integrated)
        control_layout.addWidget(self.btn_run_aw_volume)
        control_layout.addStretch(1)

        # 시뮬 제어 그룹
        self.sim_controls_widget = QWidget()
        sim_layout = QHBoxLayout(self.sim_controls_widget); sim_layout.setContentsMargins(0,0,0,0)
        self.btn_toggle_playback = QPushButton("▶ 재생"); self.btn_toggle_playback.setFixedWidth(100)
        self.btn_reset_simulation = QPushButton("↩ 초기화"); self.btn_reset_simulation.setFixedWidth(100)
        speed_label = QLabel("재생 속도:")
        self.speed_slider = QSlider(Qt.Horizontal); self.speed_slider.setRange(1, 40); self.speed_slider.setValue(10); self.speed_slider.setFixedWidth(120)
        self.speed_value_label = QLabel("1.0x")
        self.speed_slider.valueChanged.connect(lambda val: self.speed_value_label.setText(f"{val/10.0:.1f}x"))
        sim_layout.addWidget(self.btn_toggle_playback)
        sim_layout.addWidget(self.btn_reset_simulation)
        sim_layout.addSpacing(20)
        sim_layout.addWidget(speed_label)
        sim_layout.addWidget(self.speed_slider)
        sim_layout.addWidget(self.speed_value_label)

        # 색상 변수 콤보
        sim_layout.addSpacing(20)
        self.color_label = QLabel("색상 변수:")
        self.color_by = QComboBox()
        self.color_by.addItems(["MPT","MPA","MPW","LOAD","R_LP","R_WS","A*W(합성)"])
        self.color_by.setEnabled(False)
        sim_layout.addWidget(self.color_label)
        sim_layout.addWidget(self.color_by)

        control_layout.addWidget(self.sim_controls_widget)
        control_layout.addWidget(self.btn_ask_llm_about_viz)
        self.sim_controls_widget.hide()

        # 뷰 스택
        self.viz_stack = QStackedWidget()

        # 인덱스 0: Matplotlib 뷰
        self.mpl_widget = QWidget()
        mpl_layout = QVBoxLayout(self.mpl_widget)
        self.viz_fig = plt.figure(tight_layout=True)
        self.viz_canvas = FigureCanvas(self.viz_fig)
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, self)
        mpl_layout.addWidget(self.viz_toolbar); mpl_layout.addWidget(self.viz_canvas)
        self.viz_stack.addWidget(self.mpl_widget)

        # 인덱스 1: PyVista 단독 뷰
        self.pv_plotter = QtInteractor(self.viz_tab, auto_update=False)
        self.viz_stack.addWidget(self.pv_plotter.interactor)

        # 인덱스 2: 통합(좌 PyVista / 우 Matplotlib) 뷰
        self.integrated_page = QWidget()
        h = QHBoxLayout(self.integrated_page); h.setContentsMargins(0,0,0,0)
        self.pv_plotter_integrated = QtInteractor(self.integrated_page, auto_update=False)
        self.viz_fig_sync = plt.figure(tight_layout=True)
        self.viz_canvas_sync = FigureCanvas(self.viz_fig_sync)
        h.addWidget(self.pv_plotter_integrated.interactor, 2)
        h.addWidget(self.viz_canvas_sync, 1)
        self.viz_stack.addWidget(self.integrated_page)

        # 초기 안내
        ax = self.viz_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Please load a CSV file to start analysis.", ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')

        # 레이아웃 배치
        layout.addWidget(control_panel)
        layout.addWidget(self.viz_stack, 1)

        # 시그널 연결
        self.btn_load_csv_viz.clicked.connect(self.load_csv_for_viz)
        self.btn_run_stability.clicked.connect(self.run_stability_analysis)
        self.btn_run_correlation.clicked.connect(self.run_correlation_dashboard)
        self.btn_run_3d_path.clicked.connect(self.run_3d_path_analysis)
        self.btn_run_simulation.clicked.connect(self.run_process_simulation)
        self.btn_run_integrated.clicked.connect(self.run_integrated_dashboard)
        self.btn_run_aw_volume.clicked.connect(self.run_aw_volume_dashboard)
        self.btn_ask_llm_about_viz.clicked.connect(self.ask_llm_about_viz)
        self.btn_toggle_playback.clicked.connect(self.toggle_simulation_playback)
        self.btn_reset_simulation.clicked.connect(self.reset_simulation)
        self.color_by.currentTextChanged.connect(self.on_change_color_by)


    def refresh_pdf_saved_list(self):
        """DB에 저장된 PDF 목록을 아래 리스트에 반영."""
        try:
            docs = list_all_docs(self.engine)  # [(doc_id, filename, pages, created_at), ...]
        except Exception as e:
            QMessageBox.critical(self, "목록 갱신 오류", str(e))
            return

        # 맵 업데이트
        self.pdf_rag_files = {}  # {filename: doc_id}
        self.pdf_saved_list.clear()
        for doc_id, fname, pages, _ts in docs:
            self.pdf_rag_files[fname] = doc_id
            it = QListWidgetItem(f"{fname}   (p.{pages})")
            it.setData(Qt.UserRole, fname)  # 표시명과 실명 분리
            it.setCheckState(Qt.Unchecked)
            self.pdf_saved_list.addItem(it)


    # --- 신규: PDF RAG 챗봇 탭 UI 설정 ---





    def setup_pdf_tab(self):
        layout = QHBoxLayout(self.pdf_tab)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 10)

        # ============== 좌측: 파일 관리 ==============
        left.addWidget(QLabel("📂 PDF 문서 관리"))

        # (위) 새로 추가 섹션
        box_new = QFrame(); box_new.setFrameShape(QFrame.StyledPanel)
        ln = QVBoxLayout(box_new); ln.setContentsMargins(8,8,8,8)
        title_new = QLabel("➕ 새로 추가(이번 세션)")
        title_new.setStyleSheet("font-weight:600;")
        ln.addWidget(title_new)

        self.drop_pdf = DropArea(file_type="pdf")
        self.drop_pdf.setMinimumHeight(90)      # 드롭영역 높이 축소
        self.drop_pdf.filesDropped.connect(self.handle_pdf_paths)
        ln.addWidget(self.drop_pdf)

        btn_row_new = QHBoxLayout()
        self.btn_upload_pdf = QPushButton("파일 선택…")
        self.btn_upload_pdf.clicked.connect(self.on_upload_pdf)
        self.btn_del_pdf_new = QPushButton("선택 삭제")
        self.btn_del_pdf_new.clicked.connect(lambda: self.on_delete_pdf(target="new"))
        btn_row_new.addWidget(self.btn_upload_pdf)
        btn_row_new.addStretch(1)
        btn_row_new.addWidget(self.btn_del_pdf_new)
        ln.addLayout(btn_row_new)

        self.pdf_new_list = QListWidget()
        self.pdf_new_list.setToolTip("이번 세션에서 추가한 파일 목록")
        ln.addWidget(self.pdf_new_list, 1)

        left.addWidget(box_new)

        # (아래) 저장됨 섹션
        box_saved = QFrame(); box_saved.setFrameShape(QFrame.StyledPanel)
        ls = QVBoxLayout(box_saved); ls.setContentsMargins(8,8,8,8)
        title_saved = QLabel("📚 저장된 파일(로컬 DB)")
        title_saved.setStyleSheet("font-weight:600;")
        ls.addWidget(title_saved)

        btn_row_saved = QHBoxLayout()
        self.btn_refresh_saved = QPushButton("새로고침")
        self.btn_refresh_saved.clicked.connect(self.refresh_pdf_saved_list)
        self.btn_del_pdf_saved = QPushButton("선택 삭제")
        self.btn_del_pdf_saved.clicked.connect(lambda: self.on_delete_pdf(target="saved"))
        btn_row_saved.addWidget(self.btn_refresh_saved)
        btn_row_saved.addStretch(1)
        btn_row_saved.addWidget(self.btn_del_pdf_saved)
        ls.addLayout(btn_row_saved)

        self.pdf_saved_list = QListWidget()
        self.pdf_saved_list.setToolTip("DB에 저장된 모든 PDF")
        ls.addWidget(self.pdf_saved_list, 1)

        left.addWidget(box_saved, 1)

        # ============== 우측: 챗/Evidence ==============
        right.addWidget(QLabel("💬 PDF 내용에 대해 질문하기"))
        self.pdf_chat = ChatView()
        self.pdf_chat.add_bot("PDF를 드롭/업로드하거나, 아래 저장된 목록에서 바로 질문하세요.")
        right.addWidget(self.pdf_chat, 1)

        self.pdf_tabs = QTabWidget()
        self.pdf_evidence = QTextEdit(); self.pdf_evidence.setReadOnly(True)
        self.pdf_tabs.addTab(self.pdf_evidence, "Evidence")
        right.addWidget(self.pdf_tabs, 1)

        self.pdf_status = QLabel("")
        right.addWidget(self.pdf_status)

        send_row = QHBoxLayout()
        self.pdf_inp = QLineEdit(); self.pdf_inp.setPlaceholderText("질문을 입력하고 Enter…")
        self.pdf_inp.returnPressed.connect(self.on_ask_pdf)
        self.btn_send_pdf = QPushButton("▶")
        self.btn_send_pdf.clicked.connect(self.on_ask_pdf)
        send_row.addWidget(self.pdf_inp, 1)
        send_row.addWidget(self.btn_send_pdf)
        right.addLayout(send_row)

        # 시작 시 저장된 목록 채우기
        QTimer.singleShot(0, self.refresh_pdf_saved_list)


    # 공용: 시뮬 정지
    def _stop_simulation_if_running(self):
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.btn_toggle_playback.setText("▶ 재생")

    # 백엔드 초기화
    def init_backend(self):
        s = self.s = get_settings()
        self.engine = make_engine(s.db_url)
        try:
            with self.engine.begin() as c:
                c.exec_driver_sql("SELECT 1")
        except Exception as e:
            QMessageBox.critical(self, "DB 연결 실패", str(e))
        self.llm = build_llm(s.openai_model, s.openai_key, 0)
        self.sql_chain = build_sql_chain(self.llm, s.db_url)
        self.emb = build_embeddings(s.openai_key, s.embed_model)

        # 기존(메타/CSV)용
        self.chroma = build_chroma(self.emb, s.vector_db_dir)

        # 문서 전용 벡터DB (vector_db_dir/docs)
        docs_dir = Path(s.vector_db_dir) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_docs = build_chroma(self.emb, str(docs_dir))

        # PDF 텍스트 테이블 보장
        ensure_pdf_tables(self.engine)

        # PDF 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

        # 문서 이미지 저장 폴더
        self.images_dir = Path(self.s.vector_db_dir) / "doc_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # 전문가 코멘트 테이블 보장
        try:
            with self.engine.begin() as c:
                # Adjust primary key syntax based on the database dialect
                dialect = getattr(self.engine, "dialect", None)
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    # Use SERIAL for PostgreSQL
                    ddl = (
                        "\n"
                        "CREATE TABLE IF NOT EXISTS expert_comments (\n"
                        "    id SERIAL PRIMARY KEY,\n"
                        "    content TEXT NOT NULL,\n"
                        "    created_at TEXT\n"
                        ")\n"
                    )
                else:
                    # Use AUTOINCREMENT for SQLite and others that support it
                    ddl = (
                        "\n"
                        "CREATE TABLE IF NOT EXISTS expert_comments (\n"
                        "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                        "    content TEXT NOT NULL,\n"
                        "    created_at TEXT\n"
                        ")\n"
                    )
                c.exec_driver_sql(ddl)
        except Exception as e:
            print(f"[expert_comments table init error] {e}")

    # LLM 보조
    def build_prompt(self, question: str) -> str:
        full_question = question
        # Include visualization context if present
        if self.viz_context:
            full_question = f"[Current Analysis Context]\n{self.viz_context}\n\n[User's Question]\n{question}"
            self.viz_context = None
        # If there are any selected CSV files, prepend their names to the question context
        try:
            selected = self.get_selected_filenames()
        except Exception:
            selected = []
        if selected:
            try:
                summary = self.get_selected_files_summary()
            except Exception:
                summary = ""
            # Attach summary and list of selected filenames
            selected_context = "[선택된 데이터 파일 요약]\n" + summary + "\n\n"
            full_question = selected_context + full_question
        # Include expert comments from DB as domain knowledge
        try:
            with self.engine.begin() as c:
                rows = c.exec_driver_sql("SELECT content FROM expert_comments").fetchall()
                exp_comments = [r[0] for r in rows]
        except Exception:
            exp_comments = []
        if exp_comments:
            comments_str = "\n\n".join(exp_comments)
            expert_context = "[전문가 코멘트]\n" + comments_str + "\n\n"
            full_question = expert_context + full_question
        context = "".join(f"이전 Q: {q}\n이전 A: {a}\n" for q, a in self.history[-MAX_HISTORY_TURNS:])
        return context + f"질문: {full_question}"

    def save_history(self):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.history[-MAX_HISTORY_RECORDS:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[히스토리 저장 오류] {e}")

    def load_history(self):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []
        except Exception as e:
            print(f"[히스토리 불러오기 오류] {e}"); self.history = []

    def repopulate_chat(self):
        self.chat.clear()
        if not self.history:
            self.chat.add_bot("안녕하세요! 업로드 후 질문을 입력해 주세요.")
            return
        for q, a in self.history:
            self.chat.add_user(q); self.chat.add_bot(a)

    def on_clear_history(self):
        if QMessageBox.question(self, "확인", "정말 모든 채팅 로그를 삭제할까요?") == QMessageBox.Yes:
            self.history = []; self.chat.clear(); self.chat.add_bot("채팅 로그가 초기화되었습니다.")
            if os.path.exists(HISTORY_PATH):
                try: os.remove(HISTORY_PATH)
                except Exception as e: print(f"[히스토리 파일 삭제 오류] {e}")

    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy)  # guard
        self.inp.setReadOnly(busy)
        self.status.setText("🤖 답변 생성 중…" if busy else "")

    # --- 신규: PDF 탭 바쁨 표시 ---
    def set_pdf_busy(self, busy: bool):
        self.btn_send_pdf.setEnabled(not busy)
        self.pdf_inp.setReadOnly(busy)
        self.pdf_status.setText("답변 생성 중..." if busy else "")



# app/main10_pdf_rag.py 파일에서
# 기존 on_upload, handle_csv_paths, on_delete_files를 지우고 아래 코드로 대체

# app/main10_pdf_rag.py 파일에서 이 함수를 찾아 통째로 교체하세요.

# app/main10_pdf_rag.py 파일에서 이 함수를 찾아 통째로 교체하세요.

    def refresh_csv_saved_list(self):
        """files_registry.json을 읽어 저장된 CSV 목록을 갱신."""
        try:
            # core/files_registry.py에 있는 load_registry 함수를 직접 사용
            entries = load_registry()
        except Exception as e:
            QMessageBox.critical(self, "CSV 레지스트리 로딩 오류", str(e))
            return

        self.file_ids = {} # {filename: file_id}
        self.csv_saved_list.clear()
        for file_id, data in entries.items():
            # path 키에서 파일명을 안전하게 추출
            path_str = data.get("path")
            if not path_str: continue
            fname = Path(path_str).name
            
            # CSV 파일만 필터링하여 목록에 추가
            if fname.lower().endswith(".csv"):
                self.file_ids[fname] = file_id
                rows = data.get("rows", 0)
                cols = data.get("cols", 0)
                it = QListWidgetItem(f"{fname} ({rows}x{cols})")
                it.setData(Qt.UserRole, fname)
                it.setCheckState(Qt.Unchecked)
                self.csv_saved_list.addItem(it)

    def on_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
        if files:
            self.handle_csv_paths(files)

    def handle_csv_paths(self, paths: list[str]):
        # PDF 처리 방식과 동일하게 스레드 및 진행률 표시 사용
        from threading import Event
        prog = QProgressDialog("CSV 처리 및 인덱싱 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        cancel_event = Event()
        try: prog.canceled.connect(cancel_event.set)
        except: pass

        def _task(file_paths, progress_callback, cancel_event):
            """
            Ingest CSV files and compute basic statistics/preview for each.

            Returns (ok, fail, results) where results contains tuples:
            (filename, rows, cols, missing_total, desc_str, preview_str, df)
            for each successfully processed file. These additional values
            enable later rendering of summary and preview in the UI.
            """
            ok, fail, results = 0, [], []
            for i, p_str in enumerate(file_paths, 1):
                if cancel_event.is_set():
                    break
                path = Path(p_str)
                progress_callback(i - 1, f"{path.name} 처리 중...")
                try:
                    # Load CSV and metadata
                    df, meta, _ = load_and_meta(path, self.s.meta_json_dir)
                    # Register file into registry
                    entry = upsert_entry(path, rows=meta["rows"], cols=meta["cols"], status="indexed")
                    # Insert embeddings per column
                    upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                    # Ingest into SQL table
                    table = table_name_from_file(path.name)
                    ingest_df(self.engine, df, table)
                    ensure_indexes(self.engine, table)
                    # Compute statistics
                    try:
                        missing_total = int(df.isna().sum().sum())
                    except Exception:
                        missing_total = 0
                    try:
                        desc_df = df.describe(include="all")
                        desc_str = desc_df.to_string(max_cols=6, max_rows=10)
                    except Exception:
                        desc_str = ""
                    try:
                        preview_df = df.head(20)
                        preview_str = preview_df.to_string(index=False)
                    except Exception:
                        preview_str = ""
                    results.append((path.name, meta["rows"], meta["cols"], missing_total, desc_str, preview_str, df))
                    ok += 1
                except Exception as e:
                    fail.append(f"{path.name}: {e}")
            return ok, fail, results

        def _done(res, err):
            prog.close()
            if err:
                return QMessageBox.critical(self, "CSV 처리 오류", str(err))
            ok, fail, results = res
            last_df = None
            # Populate UI for new files and append summaries
            for (name, rows, cols, missing_total, desc_str, preview_str, df) in results:
                # Add to new files list
                it = QListWidgetItem(f"{name} ({rows}x{cols})")
                it.setData(Qt.UserRole, name)
                it.setCheckState(Qt.Unchecked)
                self.csv_new_list.addItem(it)
                # Notify via chat
                self.chat.add_bot(f"✅ 업로드 완료: {name}")
                # Build summary text
                summary_text = (
                    f"[파일] {name}\n"
                    f"- 행 수: {rows}\n"
                    f"- 열 수: {cols}\n"
                    f"- 결측치 총합: {missing_total}\n"
                )
                if desc_str:
                    summary_text += f"\n[통계 요약]\n{desc_str}\n"
                if preview_str:
                    summary_text += f"\n[미리보기]\n{preview_str}\n"
                try:
                    # Append to report tab
                    self.report.append(summary_text)
                except Exception:
                    current_text = self.report.toPlainText()
                    self.report.setPlainText(current_text + "\n" + summary_text)
                last_df = df
            # Refresh saved list
            self.refresh_csv_saved_list()
            # Set current dataset and visualizer to the last successfully uploaded file
            if last_df is not None:
                try:
                    self.current_df = last_df
                    self.df_viz = last_df
                    self.visualizer = AnalysisVisualizer(last_df)
                except Exception:
                    self.visualizer = None
                # Update viz tab UI similar to on_load_saved_csv
                try:
                    self._stop_simulation_if_running()
                    self.sim_controls_widget.hide()
                    self.viz_stack.setCurrentIndex(0)
                    self.viz_fig.clear()
                    ax = self.viz_fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"'{results[-1][0]}' loaded.\nPlease select an analysis.", ha='center', va='center')
                    ax.axis('off')
                    self.viz_canvas.draw()
                    for btn in [self.btn_run_stability, self.btn_run_correlation, self.btn_run_3d_path,
                                self.btn_run_simulation, self.btn_run_integrated, self.btn_run_aw_volume]:
                        btn.setEnabled(True)
                    self.btn_ask_llm_about_viz.setEnabled(False)
                except Exception:
                    pass
                # Render preview of last uploaded file in table/chart tabs
                try:
                    self.render_all(last_df, sql=None)
                except Exception:
                    pass
            # Summary message
            summary_msg = f"성공: {ok}건"
            if fail:
                summary_msg += f"\n실패: {len(fail)}건\n" + "\n".join(fail)
            QMessageBox.information(self, "CSV 처리 완료", summary_msg)

        worker, thread = run_in_thread(self, _task, _done, paths, progress_callback=None, cancel_event=cancel_event)
        worker.progress.connect(lambda i, msg: (prog.setValue(i), prog.setLabelText(msg)))


    def on_delete_files(self, target: str):
        widget = self.csv_new_list if target == "new" else self.csv_saved_list
        items = [widget.item(i) for i in range(widget.count()) if widget.item(i).checkState() == Qt.Checked]
        if not items: return QMessageBox.information(self, "알림", "체크된 파일이 없습니다.")
        if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 파일을 삭제합니다. 계속할까요?") != QMessageBox.Yes: return

        errors = []
        fnames_to_delete = [it.data(Qt.UserRole) for it in items]

        for fname in fnames_to_delete:
            try:
                # 1) DB 테이블 삭제
                table = table_name_from_file(fname)
                with self.engine.begin() as c:
                    c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
                
                # 2) Chroma 벡터 삭제
                fid = self.file_ids.get(fname)
                if fid:
                    # 메타데이터는 대략 2000개 미만으로 가정하고 삭제
                    self.chroma.delete(ids=[f"{fid}:{i:04d}" for i in range(2000)])
                
                # 3) 레지스트리에서 삭제 (core.files_registry에 삭제 함수가 없으므로 직접 구현)
                registry_path = Path(self.s.data_dir) / "files_registry.json"
                if registry_path.exists():
                    with open(registry_path, "r+", encoding="utf-8") as f:
                        entries = json.load(f)
                        fid_to_del = self.file_ids.get(fname)
                        if fid_to_del in entries:
                            del entries[fid_to_del]
                        f.seek(0); f.truncate()
                        json.dump(entries, f, indent=2)

            except Exception as e:
                errors.append(f"{fname}: {e}")

        # UI 갱신
        self.refresh_csv_saved_list()
        # 'new' 리스트에서는 직접 제거
        for i in reversed(range(self.csv_new_list.count())):
            it = self.csv_new_list.item(i)
            if it.data(Qt.UserRole) in fnames_to_delete:
                self.csv_new_list.takeItem(i)

        if errors:
            self.chat.add_bot("일부 삭제 실패:\n" + "\n".join(errors))
        else:
            self.chat.add_bot("🗑️ 선택 파일 삭제 완료")



    # # 파일 업로드/인덱싱 (CSV)
    # def on_upload(self):
    #     files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
    #     if files:
    #         self.handle_csv_paths(files)

    # def handle_csv_paths(self, paths: list[str]):
    #     prog = QProgressDialog("CSV 처리 중...", "취소", 0, len(paths), self)
    #     prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
    #     ok, fail = 0, 0
    #     for i, p in enumerate(paths, 1):
    #         prog.setValue(i - 1); QApplication.processEvents()
    #         if prog.wasCanceled(): break
    #         try:
    #             df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)
    #             entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
    #             upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
    #             self.file_ids[Path(p).name] = entry.file_id
    #             table = table_name_from_file(Path(p).name)
    #             ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)
    #             try:
    #                 if _build_meta_for_table:
    #                     sessions = _build_meta_for_table(self.s.db_url, table)
    #                     if _index_sessions: _index_sessions(self.s.db_url, str(self.s.vector_db_dir), sessions)
    #             except Exception as _e:
    #                 self.chat.add_bot(f"⚠️ 메타/인덱싱 경고: {Path(p).name}\n{_e}")
    #             self.csv_files.append((Path(p).name, df))
    #             it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
    #             self.chat.add_bot(f"✅ 업로드 완료: {Path(p).name}\n(table={table})"); ok += 1
    #         except Exception as e:
    #             self.chat.add_bot(f"❌ 업로드 실패: {p}\n{e}"); fail += 1
    #     prog.setValue(len(paths))
    #     QMessageBox.information(self, "완료", f"성공 {ok} / 실패 {fail}")
    #     self.update_report_summary()

    # def on_delete_files(self):
    #     items = [self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).checkState() == Qt.Checked]
    #     if not items:
    #         return QMessageBox.information(self, "알림", "체크된 파일이 없습니다.")
    #     if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 파일을 삭제합니다. 계속할까요?") != QMessageBox.Yes:
    #         return
    #     for it in items:
    #         fname = it.text()
    #         self.csv_files = [(f, df) for f, df in self.csv_files if f != fname]
    #         self.file_list.takeItem(self.file_list.row(it))
    #         table = table_name_from_file(fname)
    #         try:
    #             with self.engine.begin() as c:
    #                 c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
    #         except Exception as e:
    #             self.chat.add_bot(f"⚠️ DB 테이블 삭제 경고: {table} / {e}")
    #         fid = self.file_ids.pop(fname, None)
    #         if fid:
    #             try: self.chroma.delete(ids=[f"{fid}:{i:04d}" for i in range(2000)])
    #             except Exception as e: self.chat.add_bot(f"⚠️ 임베딩 삭제 경고: {fname} / {e}")
    #     self.update_report_summary(); self.chat.add_bot("🗑️ 선택 파일 삭제 완료")

    # LLM 질의 (CSV/DB + 메타)
    def on_ask(self):
        q = self.inp.text().strip()
        if not q:
            return
        # Clear input and echo user input in chat
        self.inp.clear()
        self.chat.add_user(q)
        # Reset advanced visualization flag for this query
        self._advanced_triggered = False
        # Deep report mode handling: accumulate user inputs until '끝' or '완료'
        if self.in_deep_report:
            # Check for termination keywords
            kw = q.strip().lower()
            if kw in ["끝", "완료", "finish", "done"]:
                # Generate the report and exit deep report mode
                self.generate_deep_report()
            else:
                # Append to context and prompt for more input
                self.deep_report_inputs.append(q)
                self.chat.add_bot("계속 입력해주세요. 보고서를 완료하려면 '끝' 또는 '완료'라고 입력하세요.")
            return
        # Attempt to load a dataset from selected files if none is loaded.
        # This allows analysis and visualization on datasets selected in the saved list
        # without requiring an explicit upload or double-click.
        try:
            self.ensure_dataset_loaded()
        except Exception:
            pass
        # Normal LLM processing
        tone = self.tone.currentText()
        self.set_busy(True)

        def _task():
            sql, df, err_sql = "", None, ""
            prompt_for_llm = self.build_prompt(q)
            try:
                sql = generate_sql_from_nlq(self.sql_chain, prompt_for_llm, engine_or_url=self.engine)
                df = run_sql(self.engine, sql)
                if isinstance(df, pd.DataFrame) and df.empty:
                    df = None
            except Exception as e:
                err_sql = str(e)
            try:
                docs = retrieve_meta(self.chroma, prompt_for_llm, 6)
            except Exception:
                docs = []
            df_snip = df.head(20).to_csv(index=False) if df is not None else ""
            meta_snip = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs[:4])
            final_text = llm_final_only(self.llm, prompt_for_llm, df_snip, meta_snip, tone)
            checks_list = llm_checks_only(self.llm, prompt_for_llm, df_snip, meta_snip)

            # Build evidence lines
            ev_lines = ["## 사용 근거"]
            if sql:
                ev_lines += ["### 사용 SQL", "```sql", sql.strip(), "```"]
            # Include basic info about SQL result
            if isinstance(df, pd.DataFrame):
                ev_lines += [
                    "### SQL 결과 개요",
                    f"- 행 수: {len(df)}",
                    f"- 열 수: {df.shape[1]}"
                ]
                # Append preview of the first 10 rows
                try:
                    preview = df.head(10).to_string(index=False)
                    ev_lines += ["", "### SQL 결과 미리보기", preview]
                except Exception:
                    pass
            # Include RAG evidence snippet
            if docs:
                ev_lines.append("### RAG 근거(상위 문서 첫 줄)")
                for i, d in enumerate(docs[:5], 1):
                    ev_lines.append(f"{i}. {getattr(d, 'page_content', str(d)).splitlines()[0][:200]}")
            # Additional checks list
            if checks_list:
                ev_lines += ["", "## 추가 확인 항목", checks_list]
            # SQL generation/ execution errors
            if err_sql and not sql:
                ev_lines += ["", "### SQL 생성/실행 참고", err_sql]
            return (q, final_text, df, sql, "\n".join(ev_lines))

        def _done(res, err):
            self.set_busy(False)
            if err:
                return QMessageBox.critical(self, "질의 오류", str(err))
            q, final_text, df, sql, evidence_text = res
            # Display chatbot response
            self.chat.add_bot(final_text)
            # Save history
            self.history.append((q, final_text))
            self.save_history()
            # Render table and chart for SQL result (if any)
            if isinstance(df, pd.DataFrame):
                try:
                    self.render_all(df, sql)
                except Exception:
                    pass
            # Always display evidence text
            self.evidence.setPlainText(evidence_text)
            # Determine visualization trigger keywords and parse custom visual requests
            try:
                query_lower = q.lower()
            except Exception:
                query_lower = q
            # Trigger appropriate visualization on the LLM page
            # First, try to parse a custom variable request for multi-series or 3D plots
            vis_request = None
            try:
                vis_request = self.parse_visual_request(q)
            except Exception:
                vis_request = None
            if vis_request:
                vtype, vars_list = vis_request
                self.show_visualization(vtype, vars_list)
            elif any(k in query_lower for k in ["상관관계", "correlation"]):
                # Show correlation dashboard in advanced tab
                self.show_visualization('correlation')
            elif any(k in query_lower for k in ["3d", "경로", "path"]):
                # Show 3D path visualization in advanced tab
                self.show_visualization('3d')
            elif any(k in query_lower for k in ["시뮬레이션", "simulation"]):
                # Notify that simulation is only available in the viz tab
                self.show_visualization('simulation')
            elif any(k in query_lower for k in ["a*w", "aw"]):
                # Notify that AW dashboard is only available in the viz tab
                self.show_visualization('aw')
            elif ("mpt" in query_lower or "m.p.t" in query_lower) and any(k in query_lower for k in ["시간", "time"]):
                # Plot MPT vs Time if both keywords present
                self.show_visualization('mpt_time')
            # If no advanced visualization was triggered, select the most appropriate results tab
            if not self._advanced_triggered:
                # Prefer table view if a DataFrame result exists
                if isinstance(df, pd.DataFrame):
                    idx = self.tabs.indexOf(self.tbl)
                    if idx >= 0:
                        self.tabs.setCurrentIndex(idx)
                else:
                    # Otherwise show evidence tab
                    idx = self.tabs.indexOf(self.evidence)
                    if idx >= 0:
                        self.tabs.setCurrentIndex(idx)
            # Ensure we stay on the LLM tab (index 0 of the main tab widget)
            try:
                if self.tab_widget.currentIndex() != 0:
                    self.tab_widget.setCurrentIndex(0)
            except Exception:
                pass

        run_in_thread(self, _task, _done)

    # --------------- 유틸: 이미지 갤러리 갱신 ---------------
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _refresh_images_gallery(self):
        """self.pdf_images_map을 기준으로 Images 탭을 다시 그림."""
        self._clear_layout(self.pdf_images_layout)
        if not self.pdf_images_map:
            lab = QLabel("추출된 이미지가 없습니다.")
            lab.setStyleSheet("color:#6b7280;")
            self.pdf_images_layout.addWidget(lab)
            self.pdf_images_layout.addStretch(1)
            return
        for fname, paths in self.pdf_images_map.items():
            title = QLabel(f"📄 {fname}  (이미지 {len(paths)}개)")
            title.setStyleSheet("font-weight:600;")
            self.pdf_images_layout.addWidget(title)
            # 썸네일 나열 (한 행 5개)
            def flush_row(row):
                w = QWidget(); w.setLayout(row); self.pdf_images_layout.addWidget(w)
            cnt = 0
            row = QHBoxLayout(); row.setSpacing(12)
            for p in paths:
                try:
                    pm = QPixmap(str(p))
                    if not pm.isNull():
                        thumb = pm.scaledToWidth(240, Qt.SmoothTransformation)
                        lab = QLabel(); lab.setPixmap(thumb); lab.setToolTip(str(p))
                        row.addWidget(lab); cnt += 1
                        if cnt % 5 == 0:
                            flush_row(row); row = QHBoxLayout(); row.setSpacing(12)
                except Exception:
                    pass
            if row.count():
                flush_row(row)
            self.pdf_images_layout.addSpacing(8)
        self.pdf_images_layout.addStretch(1)

    # --- 신규: PDF RAG 기능 메서드들 ---
    def on_upload_pdf(self):
        files, _ = QFileDialog.getOpenFileNames(self, "PDF 파일 선택", "", "PDF Files (*.pdf)")
        if files:
            self.handle_pdf_paths(files)

    def handle_pdf_paths(self, paths: list[str]):
        from threading import Event
        prog = QProgressDialog("PDF 처리 및 인덱싱 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)
        cancel_event = Event()
        try:
            prog.canceled.connect(cancel_event.set)
        except Exception:
            pass

        def _task(file_paths, progress_callback, cancel_event):
            ok, fail, results = 0, [], []
            for i, p_str in enumerate(file_paths, start=1):
                if cancel_event.is_set():
                    break
                path = Path(p_str)
                progress_callback(i - 1, f"{path.name} 텍스트/이미지 추출 중...")
                try:
                    # --- 1) 페이지 단위 텍스트 추출 ---
                    chunks_all: List[Tuple[int,int,str]] = []  # (page, chunk_index, text)
                    pages_count = 0

                    if fitz is not None:
                        doc = fitz.open(str(path))
                        pages_count = doc.page_count
                        for pi in range(pages_count):
                            if cancel_event.is_set(): break
                            txt = doc.load_page(pi).get_text("text") or ""
                            if not txt.strip():
                                continue
                            parts = self.text_splitter.split_text(txt)
                            parts = [f"[{path.name} | p.{pi+1}]\n{c}" for c in parts]
                            chunks_all.extend((pi+1, idx, t) for idx, t in enumerate(parts))
                        doc.close()
                    else:
                        reader = PdfReader(path)
                        pages_count = len(reader.pages)
                        for pi, page in enumerate(reader.pages, start=1):
                            if cancel_event.is_set(): break
                            txt = page.extract_text() or ""
                            if not txt.strip():
                                continue
                            parts = self.text_splitter.split_text(txt)
                            parts = [f"[{path.name} | p.{pi}]\n{c}" for c in parts]
                            chunks_all.extend((pi, idx, t) for idx, t in enumerate(parts))

                    if cancel_event.is_set(): break
                    if not chunks_all:
                        raise ValueError("텍스트를 추출할 수 없거나 문서가 비어 있습니다.")

                    # --- 2) DB 저장 ---
                    doc_id = insert_pdf(self.engine, path.name, pages_count)
                    chunk_ids = insert_chunks(self.engine, doc_id, chunks_all)  # -> [chunk_id...]

                    # --- 3) 벡터 인덱싱(문서 전용 chroma_docs): id=chunk_id ---
                    texts = [t for (_pg, _idx, t) in chunks_all]
                    metas = [{"chunk_id": cid, "doc_id": doc_id, "filename": path.name, "page": pg}
                             for (pg, _idx, _t), cid in zip(chunks_all, chunk_ids)]
                    ids = [str(cid) for cid in chunk_ids]
                    self.chroma_docs.add_texts(texts=texts, ids=ids, metadatas=metas)

                    # --- 4) 이미지 추출 (PyMuPDF 있을 때) ---
                    image_paths: List[str] = []
                    if fitz is not None:
                        try:
                            doc = fitz.open(str(path))
                            max_images = 40
                            for pi in range(len(doc)):
                                if len(image_paths) >= max_images: break
                                for img in doc.get_page_images(pi):
                                    xref = img[0]
                                    w, h = img[2], img[3]
                                    if (w or 0) * (h or 0) < 20000:  # 작은 아이콘/배경 스킵
                                        continue
                                    try:
                                        pix = doc.extract_image(xref)
                                        ext = pix.get("ext", "png")
                                        data = pix.get("image")
                                        out = self.images_dir / f"{doc_id}_p{pi+1}_{len(image_paths)}.{ext}"
                                        with open(out, "wb") as f:
                                            f.write(data)
                                        image_paths.append(str(out))
                                        if len(image_paths) >= max_images:
                                            break
                                    except Exception:
                                        continue
                            doc.close()
                        except Exception:
                            pass

                    results.append((path.name, doc_id, len(chunk_ids), image_paths))
                    ok += 1
                except Exception as e:
                    fail.append(f"{path.name}: {e}")
            return ok, fail, results

        def _done(res, err):
            prog.close()
            if err:
                QMessageBox.critical(self, "PDF 처리 오류", str(err)); return

            ok, fail, results = res

            # 신규 목록에 먼저 반영
            for name, doc_id, n_chunks in results:
                it = QListWidgetItem(f"{name}   (chunks={n_chunks})")
                it.setData(Qt.UserRole, name)
                it.setCheckState(Qt.Unchecked)
                self.pdf_new_list.addItem(it)
                self.pdf_chat.add_bot(f"업로드 완료: {name} (chunks={n_chunks})")

            # 저장됨(아래) 목록은 DB 기준으로 전체 새로고침
            self.refresh_pdf_saved_list()

            summary = f"성공: {ok}건"
            if fail:
                summary += f"\n실패: {len(fail)}건\n" + "\n".join(fail)
            QMessageBox.information(self, "PDF 처리 완료", summary)


        worker, thread = run_in_thread(self, _task, _done, paths, progress_callback=None, cancel_event=cancel_event)
        worker.progress.connect(lambda i, msg: (prog.setValue(i), prog.setLabelText(msg)))

    def on_delete_pdf(self, target: str = "saved"):
        """
        target: 'new' | 'saved'
        - 'new' 리스트에서 선택 삭제해도 실제 DB/Chroma까지 삭제.
        - 'saved' 리스트는 당연히 DB/Chroma 삭제.
        """
        widget = self.pdf_new_list if target == "new" else self.pdf_saved_list
        items = [widget.item(i) for i in range(widget.count())
                if widget.item(i).checkState() == Qt.Checked]
        if not items:
            return QMessageBox.information(self, "알림", "체크된 문서가 없습니다.")

        if QMessageBox.question(self, "삭제 확인",
                                f"{len(items)}개 문서를 삭제합니다. 계속할까요?") != QMessageBox.Yes:
            return

        errors = []
        for it in items:
            label = it.text()
            fname = it.data(Qt.UserRole) or label.split("   ")[0]  # 저장 시 UserRole에 fname 넣음
            doc_id = self.pdf_rag_files.get(fname)
            try:
                if doc_id:
                    # 1) Chroma 삭제
                    cids = list_chunk_ids_by_doc(self.engine, doc_id)
                    if cids:
                        self.chroma_docs.delete(ids=[str(x) for x in cids])
                    # 2) DB 삭제
                    delete_doc(self.engine, doc_id)
            except Exception as e:
                errors.append(f"{fname}: {e}")

        # UI 양쪽에서 모두 제거
        def _remove_from(lst: QListWidget):
            for i in reversed(range(lst.count())):
                it = lst.item(i)
                fname_i = it.data(Qt.UserRole) or it.text().split("   ")[0]
                if any((fname_i == (itm.data(Qt.UserRole) or itm.text().split('   ')[0])) for itm in items):
                    lst.takeItem(i)

        _remove_from(self.pdf_new_list)
        self.refresh_pdf_saved_list()  # 아래 리스트는 DB 기준으로 재로딩

        if errors:
            self.pdf_chat.add_bot("일부 삭제 실패:\n" + "\n".join(errors))
        else:
            self.pdf_chat.add_bot("선택 문서 삭제 완료")


    def on_ask_pdf(self):
        q = self.pdf_inp.text().strip()
        if not q:
            return

        self.pdf_inp.clear()
        self.pdf_chat.add_user(q)
        self.set_pdf_busy(True)

        def _task():
            # 1) 문서 전용 컬렉션에서 벡터 검색 → chunk_id 수집
            chunk_ids: List[int] = []
            try:
                docs = self.chroma_docs.similarity_search(q, k=6)
            except Exception:
                docs = []

            for d in docs:
                cid = None
                md = getattr(d, "metadata", {}) or {}
                if "chunk_id" in md:
                    cid = int(md["chunk_id"])
                else:
                    try:
                        cid = int(getattr(d, "id", None) or md.get("id"))
                    except Exception:
                        cid = None
                if cid:
                    chunk_ids.append(cid)

            # 2) DB에서 정확 텍스트 로드
            context_texts = fetch_chunks_by_ids(self.engine, chunk_ids[:6]) if chunk_ids else []

            # 3) 백업: LIKE 키워드 검색
            if not context_texts:
                found = keyword_search_chunks(self.engine, q, limit=6)
                chunk_ids = [cid for cid, _ in found]
                context_texts = [txt for _, txt in found]

            context = "\n\n---\n\n".join(context_texts)
            answer = llm_pdf_rag_answer(self.llm, q, context)

            # Evidence
            ev_lines = ["## Retrieved Evidence"]
            for i, txt in enumerate(context_texts[:6], 1):
                first = (txt.splitlines() or [""])[0][:200]
                ev_lines.append(f"{i}. {first}")
            return (q, answer, "\n".join(ev_lines))

        def _done(res, err):
            self.set_pdf_busy(False)
            if err:
                QMessageBox.critical(self, "PDF 질의 오류", str(err))
                self.pdf_chat.add_bot(f"오류가 발생했습니다: {err}")
                return
            q, answer, evidence = res
            self.pdf_chat.add_bot(answer)
            self.pdf_rag_history.append((q, answer))
            self.pdf_evidence.setPlainText(evidence)

        run_in_thread(self, _task, _done)

    # --- 기존 메서드들 ---
    def update_report_summary(self):
        if not self.csv_files:
            self.report.setPlainText("업로드된 데이터가 없습니다.")
            return
        lines = ["# 자동 분석 리포트(데이터 요약)\n"]
        for fname, df in self.csv_files:
            lines += [f"## 파일: {fname}", f"- 행: {len(df)}, 열: {df.shape[1]}"]
            for c in df.select_dtypes(include="number").columns[:10]:
                s = df[c].dropna()
                if not s.empty:
                    lines.append(f"· {c}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
            lines.append("")
        self.report.setPlainText("\n".join(lines))

    def render_all(self, df: pd.DataFrame, sql: str | None):
        view = df.head(self.MAX_ROWS_TABLE)
        plot_df = view.iloc[::max(1, len(view)//self.MAX_POINTS_PLOT)] if len(view) > self.MAX_POINTS_PLOT else view
        df_to_table(self.tbl, view)
        plot_df_line(self.ax, self.canvas, plot_df)
        self.last_df = df

    # ===== 시각화/시뮬 =====
    def load_csv_for_viz(self):
        self._stop_simulation_if_running()
        self.sim_controls_widget.hide()
        fileName, _ = QFileDialog.getOpenFileName(self, "CSV 파일 열기", "", "CSV Files (*.csv)")
        if fileName:
            try:
                self.df_viz = pd.read_csv(fileName)
                self.visualizer = AnalysisVisualizer(self.df_viz)
                self.viz_stack.setCurrentIndex(0)
                self.viz_fig.clear()
                ax = self.viz_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"'{Path(fileName).name}' loaded.\nPlease select an analysis.", ha='center', va='center')
                ax.axis('off'); self.viz_canvas.draw()
                self.btn_run_stability.setEnabled(True)
                self.btn_run_correlation.setEnabled(True)
                self.btn_run_3d_path.setEnabled(True)
                self.btn_run_simulation.setEnabled(True)
                self.btn_run_integrated.setEnabled(True)
                self.btn_run_aw_volume.setEnabled(True)
                self.btn_ask_llm_about_viz.setEnabled(False)
            except Exception as e:
                self.viz_fig.clear()
                ax = self.viz_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"File Load Error:\n{e}", ha='center', wrap=True); ax.axis('off')
                self.viz_canvas.draw()

    def _run_visualization(self, plot_function_name: str, context_text: str):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(0)
        self.sim_controls_widget.hide()
        if not self.visualizer:
            return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            plot_function = getattr(self.visualizer, plot_function_name)
            plot_function(self.viz_fig)
            self.viz_canvas.draw()
            self.viz_context = context_text
            self.btn_ask_llm_about_viz.setEnabled(True)
        except Exception as e:
            self.viz_fig.clear()
            ax = self.viz_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Analysis Error:\n{e}", ha='center', wrap=True); ax.axis('off')
            self.viz_canvas.draw()
            print(traceback.format_exc())

    def run_integrated_dashboard(self):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(2)  # 통합 뷰
        self.sim_controls_widget.show()
        if not self.visualizer:
            return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            self.sim_handles = self.visualizer.prepare_integrated_dashboard(self.pv_plotter_integrated, self.viz_fig_sync)
            if self.sim_handles is None:
                QMessageBox.information(self, "Info", "No data for simulation.")
                self.viz_stack.setCurrentIndex(0); self.sim_controls_widget.hide()
                self.color_by.setEnabled(False)
                return
            self.viz_canvas_sync.draw()
            self.active_plotter = self.pv_plotter_integrated
            self.reset_simulation()
            self.viz_context = "integrated analysis dashboard with 3D simulation and synchronized 2D graphs."
            self.btn_ask_llm_about_viz.setEnabled(True)
            self.color_by.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Dashboard Error", str(e))
            print(traceback.format_exc())
            self.viz_stack.setCurrentIndex(0)
            self.sim_controls_widget.hide()
            self.color_by.setEnabled(False)

    def run_stability_analysis(self): self._run_visualization('plot_stability', "stability analysis graph.")
    def run_correlation_dashboard(self): self._run_visualization('plot_correlation_dashboard', "correlation analysis dashboard.")
    def run_3d_path_analysis(self): self._run_visualization('plot_3d_path', "static 3D process path visualization.")

    def run_process_simulation(self):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(1)
        self.sim_controls_widget.show()
        if not self.visualizer:
            return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            self.sim_handles = self.visualizer.prepare_pyvista_simulation(self.pv_plotter)
            if self.sim_handles is None:
                QMessageBox.information(self, "시뮬레이션 불가", "시뮬레이션을 생성하기에 유효한 데이터가 부족합니다.")
                self.viz_stack.setCurrentIndex(0)
                self.sim_controls_widget.hide()
                self.color_by.setEnabled(False)
                return
            self.active_plotter = self.pv_plotter
            self.reset_simulation()
            self.viz_context = "time-synchronized 3D animation of the process."
            self.btn_ask_llm_about_viz.setEnabled(True)
            self.color_by.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            print(traceback.format_exc())
            self.viz_stack.setCurrentIndex(0)
            self.sim_controls_widget.hide()
            self.color_by.setEnabled(False)

    def run_aw_volume_dashboard(self):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(2)  # 통합 뷰
        self.sim_controls_widget.show()
        if not self.visualizer:
            return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            self.sim_handles = self.visualizer.prepare_aw_volume_dashboard(self.pv_plotter_integrated, self.viz_fig_sync)
            if self.sim_handles is None:
                QMessageBox.information(self, "Info", "A*W 부피 계산에 필요한 컬럼이 부족합니다.")
                self.viz_stack.setCurrentIndex(0); self.sim_controls_widget.hide()
                self.color_by.setEnabled(False)
                return
            self.viz_canvas_sync.draw()
            self.active_plotter = self.pv_plotter_integrated
            self.reset_simulation()
            self.viz_context = "A*W fused dashboard with real-time bead thickness/volume."
            self.btn_ask_llm_about_viz.setEnabled(True)
            self.color_by.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "AW Dashboard Error", str(e))
            print(traceback.format_exc())
            self.viz_stack.setCurrentIndex(0)
            self.sim_controls_widget.hide()
            self.color_by.setEnabled(False)

    def toggle_simulation_playback(self):
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.btn_toggle_playback.setText("▶ 재생")
        else:
            if self.sim_handles is not None and self.sim_frame_index >= len(self.sim_handles["sim_df"]):
                self.reset_simulation()
            self.simulation_timer.start(0)  # 첫 틱 즉시
            self.btn_toggle_playback.setText("⏸ 일시정지")

    def reset_simulation(self):
        self._stop_simulation_if_running()
        self.sim_frame_index = 0
        if self.sim_handles:
            base = self.sim_handles["base_rgba"]
            cur = base.copy(); cur[:, 3] = 0
            self.sim_handles["rgba_current"] = cur
            self.sim_handles["path_poly"]["RGBA"] = cur
            self.sim_handles["head_actor"].SetPosition(self.sim_handles["points"][0])
            if "aw_norm" in self.sim_handles:
                try:
                    s = float(self.sim_handles.get("head_scale_min", 0.6))
                    self.sim_handles["head_actor"].SetScale(s)
                except Exception:
                    pass
            if self.active_plotter:
                self.active_plotter.render()

    def _update_simulation_frame(self):
        if (not self.sim_handles) or (self.sim_frame_index >= len(self.sim_handles["sim_df"])):
            self._stop_simulation_if_running()
            return

        i = self.sim_frame_index
        sim_df = self.sim_handles["sim_df"]

        cur = self.sim_handles["rgba_current"]
        base = self.sim_handles["base_rgba"]
        end = min(i + 1, len(cur))
        if end > 0:
            cur[:end, :] = base[:end, :]
            self.sim_handles["path_poly"]["RGBA"] = cur

        self.sim_handles["head_actor"].SetPosition(self.sim_handles["points"][i])

        if "aw_norm" in self.sim_handles:
            aw_norm = self.sim_handles["aw_norm"]
            smin = float(self.sim_handles.get("head_scale_min", 0.6))
            smax = float(self.sim_handles.get("head_scale_max", 2.2))
            scale = smin + (smax - smin) * float(aw_norm[i])
            try:
                self.sim_handles["head_actor"].SetScale(scale)
            except Exception:
                pass

        if getattr(self, "active_plotter", None):
            self.active_plotter.render()

        vlines = self.sim_handles.get("vlines")
        if vlines:
            tvec = self.sim_handles.get("time_vec")
            x_val = tvec[i] if tvec is not None else (sim_df["time"].iloc[i] if "time" in sim_df.columns else i)
            for v in vlines.values():
                v.set_xdata([x_val, x_val])

            if "cumV_vec" in self.sim_handles:
                cumV = float(self.sim_handles["cumV_vec"][i])
                ax_bottom = self.sim_handles.get("ax_cumv")
                if ax_bottom is not None:
                    ax_bottom.set_title(f'Cumulative Volume (est.): {cumV:.3f}  [unit^3]', fontsize=10)
                fig = self.sim_handles.get("mpl_fig")
                canvas = getattr(fig, "canvas", None) or getattr(self, "viz_canvas_sync", None)
                if canvas:
                    canvas.draw_idle()

        if i + 1 < len(sim_df):
            try:
                interval_ms = float(sim_df["time_delta_ms"].iloc[i + 1])
            except Exception:
                interval_ms = 0.0
            playback_speed = max(0.1, self.speed_slider.value() / 10.0)
            adjusted = max(15, int(interval_ms / playback_speed)) if interval_ms > 0 else 15
            self.simulation_timer.setInterval(adjusted)
        else:
            self._stop_simulation_if_running()

        self.sim_frame_index += 1

    def on_change_color_by(self, name: str):
        if not self.visualizer or not self.sim_handles:
            return
        key = "A*W" if name.startswith("A*W") else name
        ok = False
        try:
            ok = self.visualizer.recolor_simulation(self.sim_handles, color_by=key)
        except Exception as e:
            print("[recolor_simulation error]", e)
        if not ok:
            return
        base = self.sim_handles["base_rgba"].copy()
        cur = base.copy(); cur[:,3] = 0
        end = min(self.sim_frame_index, len(cur))
        if end > 0:
            cur[:end, :] = base[:end, :]
        self.sim_handles["rgba_current"] = cur
        self.sim_handles["path_poly"]["RGBA"] = cur
        if self.active_plotter:
            self.active_plotter.render()

    def ask_llm_about_viz(self):
        if self.viz_context:
            self.tab_widget.setCurrentIndex(0)
            self.inp.setText("이 분석 결과가 의미하는 바를 해석하고, 공정 개선을 위한 제안 3가지를 해줘.")
            self.inp.setFocus()

        # Ensure dataset is loaded from selection if not already
        # (used when the user directly asks to analyze or interpret current visualization)
        try:
            self.ensure_dataset_loaded()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 전문가 코멘트 관리 메서드들
    def load_expert_comments(self):
        """
        Load expert comments from the database and populate the expert comments tab.
        Each comment will be displayed with a delete (X) button.
        """
        # Clear existing UI items
        layout = self.expert_container_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # Fetch comments from DB
        comments = []
        try:
            with self.engine.begin() as c:
                rows = c.exec_driver_sql("SELECT id, content FROM expert_comments ORDER BY id").fetchall()
                comments = [(int(r[0]), r[1]) for r in rows]
        except Exception as e:
            print(f"[load_expert_comments error] {e}")
        # Populate UI
        for cid, text in comments:
            fr = QFrame()
            fr.setFrameShape(QFrame.StyledPanel)
            h_layout = QHBoxLayout(fr)
            h_layout.setContentsMargins(4, 4, 4, 4)
            lab = QLabel(text)
            lab.setWordWrap(True)
            h_layout.addWidget(lab, 1)
            btn_del = QPushButton("✕")
            btn_del.setFixedWidth(24)
            btn_del.setStyleSheet("color: red; font-weight: bold;")
            # Use lambda to capture comment id
            btn_del.clicked.connect(lambda _, cid=cid: self.on_delete_expert_comment(cid))
            h_layout.addWidget(btn_del)
            layout.addWidget(fr)
        # Add stretch to push items to top
        layout.addStretch(1)

    def on_save_expert_comment(self):
        """
        Save the content of expert_input to the database as a new expert comment,
        if it is not empty.
        """
        content = self.expert_input.toPlainText().strip()
        if not content:
            QMessageBox.information(self, "입력 필요", "코멘트를 입력한 뒤 저장하세요.")
            return
        try:
            from datetime import datetime
            now = datetime.utcnow().isoformat()
        except Exception:
            now = None
        try:
            with self.engine.begin() as c:
                # psycopg2 (PostgreSQL) doesn't accept ':' parameters; choose param style based on dialect
                dialect = getattr(self.engine, "dialect", None)
                # Default to SQLite/qmark style
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    # Use Postgres pyformat (positional) placeholders
                    stmt = "INSERT INTO expert_comments (content, created_at) VALUES (%s, %s)"
                    params = (content, now)
                else:
                    # Use qmark style placeholders (e.g. SQLite)
                    stmt = "INSERT INTO expert_comments (content, created_at) VALUES (?, ?)"
                    params = (content, now)
                c.exec_driver_sql(stmt, params)
            # Clear input after saving
            self.expert_input.clear()
            # Refresh list
            self.load_expert_comments()
            QMessageBox.information(self, "저장", "코멘트가 저장되었습니다.")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"코멘트를 저장할 수 없습니다: {e}")

    def on_delete_expert_comment(self, comment_id: int):
        """
        Delete an expert comment from the database and update the UI.
        """
        # Confirm deletion
        resp = QMessageBox.question(
            self, "삭제 확인", "정말 이 코멘트를 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return
        try:
            with self.engine.begin() as c:
                dialect = getattr(self.engine, "dialect", None)
                # Build deletion statement based on DB dialect
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    stmt = "DELETE FROM expert_comments WHERE id = %s"
                    params = (comment_id,)
                else:
                    stmt = "DELETE FROM expert_comments WHERE id = ?"
                    params = (comment_id,)
                c.exec_driver_sql(stmt, params)
            # Refresh list
            self.load_expert_comments()
        except Exception as e:
            QMessageBox.warning(self, "삭제 실패", f"코멘트를 삭제할 수 없습니다: {e}")

    # ------------------------------------------------------------------
    # 선택된 CSV 파일 명단을 반환
    def get_selected_filenames(self) -> List[str]:
        """
        Return a list of filenames selected by the user. An item is considered selected if its checkbox
        is ticked or if the row is highlighted (selected) in the list. This makes it more convenient
        for the user to simply click a row without checking the box to select a dataset.
        """
        names: List[str] = []
        # Helper to extract names from a given QListWidget
        def collect_names(lst: QListWidget | None):
            if not lst:
                return
            for i in range(lst.count()):
                it = lst.item(i)
                if not it:
                    continue
                # Determine if the item is checked
                try:
                    checked = (it.checkState() == Qt.Checked)
                except Exception:
                    checked = False
                # Determine if the item is selected (highlighted)
                selected = False
                try:
                    selected = it.isSelected()
                except Exception:
                    selected = False
                # If either checked or selected, include this filename
                if checked or selected:
                    try:
                        fname = it.data(Qt.UserRole) or it.text().split(' ')[0]
                    except Exception:
                        fname = None
                    if fname:
                        names.append(fname)
        # Collect from new and saved lists
        collect_names(getattr(self, 'csv_new_list', None))
        collect_names(getattr(self, 'csv_saved_list', None))
        # If no names found via checked/selected items, fall back to the current item in each list
        if not names:
            for lst in (getattr(self, 'csv_new_list', None), getattr(self, 'csv_saved_list', None)):
                try:
                    current = lst.currentItem() if lst else None
                except Exception:
                    current = None
                if current:
                    try:
                        fname = current.data(Qt.UserRole) or current.text().split(' ')[0]
                    except Exception:
                        fname = None
                    if fname:
                        names.append(fname)
                        break
        return names

    def ensure_dataset_loaded(self) -> bool:
        """
        Ensure that a dataset is loaded into current_df and visualizer. If none are loaded yet,
        attempt to load the first selected dataset from the registry or database. Returns
        True if a dataset is loaded after this call, False otherwise.
        """
        # If we already have a dataset loaded and a visualizer, nothing to do
        if getattr(self, "current_df", None) is not None and getattr(self, "visualizer", None) is not None:
            return True
        # If there is no loaded dataset but there is a previously rendered DataFrame (last_df),
        # use it as the current dataset.  This allows advanced visualizations to operate on
        # the most recent LLM query results even if the user has not explicitly selected a file.
        try:
            if getattr(self, "last_df", None) is not None and isinstance(self.last_df, pd.DataFrame) and not self.last_df.empty:
                # Set current_df and df_viz to the last result
                self.current_df = self.last_df
                self.df_viz = self.last_df
                try:
                    self.visualizer = AnalysisVisualizer(self.last_df)
                except Exception:
                    self.visualizer = None
                # Record a placeholder name for prompts
                if not getattr(self, "selected_dataset_name", None):
                    self.selected_dataset_name = "recent_query"
                return True
        except Exception:
            pass
        # Try to load from selected filenames (checkboxes)
        try:
            selected_names = self.get_selected_filenames()
        except Exception:
            selected_names = []
        # No selected data, cannot load
        if not selected_names:
            return False
        # Loop through selected names and load the first valid dataset
        for fname in selected_names:
            path_str = None
            # Determine file path from registry via file_id
            try:
                file_id = self.file_ids.get(fname)
            except Exception:
                file_id = None
            if file_id:
                try:
                    entries = load_registry()
                    entry = entries.get(file_id)
                    if entry and entry.get("path"):
                        path_str = entry["path"]
                except Exception:
                    pass
            df_loaded = None
            # Try to load from file
            if path_str and os.path.exists(path_str):
                try:
                    # Use load_and_meta for consistency (standardize columns)
                    df_loaded, meta, _ = load_and_meta(Path(path_str), self.s.meta_json_dir)
                except Exception:
                    try:
                        df_loaded = pd.read_csv(path_str)
                    except Exception:
                        df_loaded = None
            # Fallback: load from SQL table
            if df_loaded is None:
                table = table_name_from_file(fname)
                try:
                    df_loaded = run_sql(self.engine, f'SELECT * FROM "{table}"')
                except Exception:
                    df_loaded = None
            if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty:
                # Set current_df, df_viz and visualizer
                try:
                    self.current_df = df_loaded
                    self.df_viz = df_loaded
                    try:
                        self.visualizer = AnalysisVisualizer(df_loaded)
                    except Exception:
                        self.visualizer = None
                    # Also update selected_dataset_name for prompts
                    self.selected_dataset_name = fname
                except Exception:
                    pass
                break
        # Return True if loaded
        return getattr(self, "current_df", None) is not None and getattr(self, "visualizer", None) is not None

    def get_selected_files_summary(self) -> str:
        """
        Build a human-readable summary of the selected CSV files using their
        metadata (rows and columns) from the registry. If registry data is
        unavailable, uses a placeholder.
        """
        names = self.get_selected_filenames()
        if not names:
            return ""
        summary_lines: List[str] = []
        try:
            entries = load_registry()
        except Exception:
            entries = {}
        for fname in names:
            file_id = self.file_ids.get(fname)
            rows = cols = None
            if file_id and entries:
                data = entries.get(file_id)
                if data:
                    rows = data.get("rows")
                    cols = data.get("cols")
            if rows is not None and cols is not None:
                summary_lines.append(f"- {fname}: {rows}행 × {cols}열")
            else:
                summary_lines.append(f"- {fname}: (요약 정보 없음)")
        return "\n".join(summary_lines)

    # ------------------------------------------------------------------
    # Parse user query for custom visualization requests
    def parse_visual_request(self, query: str) -> Tuple[str, List[str]] | None:
        """
        Analyze the user's query and determine if it requests a custom visualization using
        multiple variables. Returns a tuple (viz_type, variables) where viz_type is one of:

          - 'custom_3d': for 3D scatter/line plots with three variables (x, y, z)
          - 'custom_time': for multi-series line charts vs. time (first variable should be time)

        If the query does not match a custom visual request, returns None.

        Example queries:
            '시간, MPT, r_LP 3차원 시각화' -> ('custom_3d', ['time', 'mpt', 'r_lp'])
            '전체 시간에 대해 MPT, R_RS 그래프를 그려줘' -> ('custom_time', ['time','mpt','r_rs'])
        """
        if not isinstance(query, str) or not query.strip():
            return None
        lower = query.lower()
        # Determine if the user requested a 3D visualization
        is_3d = False
        for kw in ["3d", "3차원", "3d", "three-dimensional", "3차원 시각화", "3d plot", "3d 시각화"]:
            if kw in lower:
                is_3d = True
                break
        # Tokenize the query into potential variable names (alphanumerics, Hangul, and underscores)
        import re
        tokens = re.findall(r"[a-zA-Z0-9가-힣_]+", query)
        # Build mapping of lower-case standardized column names to original
        if getattr(self, "current_df", None) is None:
            return None
        col_map = {c.lower(): c for c in self.current_df.columns}
        # Define simple synonym mapping for commonly used Korean terms
        # to their corresponding column names (in lower-case) when present.
        synonyms = {
            "시간": "time",  # time column
            "타임": "time",
            "시각": "time",
            "타임스탬프": "time",
            "mpt": "mpt",  # melt pool temperature (if written in lower-case)
            "mpa": "mpa",
            "mpw": "mpw",
        }
        variables: List[str] = []
        # Helper to add variable if it matches a column
        def try_add_var(tok: str):
            """
            Attempt to map a token to an existing DataFrame column.  The search
            considers exact matches, normalized matches (underscores removed),
            substring matches, and user-defined synonyms for Korean terms.  The
            result is appended to the variables list if found.
            """
            raw = tok.strip()
            if not raw:
                return
            # Check for synonym mapping (case-sensitive for Korean terms)
            if raw in synonyms:
                mapped = synonyms[raw]
                # Use lower-case version of the mapped column name to find in col_map
                lc = mapped.lower()
                if lc in col_map:
                    variables.append(col_map[lc])
                    return
            t = raw.lower()
            # Try exact match
            if t in col_map:
                variables.append(col_map[t])
                return
            # Try removing underscores or hyphens for both token and columns
            t_mod = t.replace("_", "").replace("-", "")
            for key in col_map:
                key_mod = key.replace("_", "").replace("-", "")
                if key_mod == t_mod:
                    variables.append(col_map[key])
                    return
            # As a last resort, check if token is a substring of any column name
            for key in col_map:
                if t in key:
                    variables.append(col_map[key])
                    return
        # Extract potential variable names from tokens
        for tok in tokens:
            try_add_var(tok)
        # Remove duplicates while preserving order
        seen = set()
        variables = [x for x in variables if not (x.lower() in seen or seen.add(x.lower()))]
        # Determine type based on presence of variables and 3D flag
        if is_3d and len(variables) >= 3:
            # Use the first three variables
            return ("custom_3d", variables[:3])
        # For multi-series line vs time: require at least one time-like and one other variable
        # Identify time-like columns among variables
        time_like = [v for v in variables if any(k in v.lower() for k in ["time", "date", "datetime", "ts"])]
        if time_like and len(variables) >= 2:
            # Ensure the time variable is first in the list
            time_col = time_like[0]
            # Put time_col first and the rest afterwards (preserve order)
            ordered = [time_col] + [v for v in variables if v != time_col]
            return ("custom_time", ordered)
        # No custom visualization pattern detected
        return None

    # ------------------------------------------------------------------
    # 새로운 기능: 저장된 CSV 더블클릭으로 로딩
    def on_load_saved_csv(self, item):
        """
        Load a saved CSV from the registry into the visualizer and set it
        as the current dataset for advanced visualizations and analysis.
        """
        try:
            fname = item.data(Qt.UserRole) or item.text().split()[0]
        except Exception:
            fname = item.text() if item else None
        if not fname:
            return
        # Determine file path from registry
        path_str = None
        file_id = self.file_ids.get(fname)
        if file_id:
            try:
                entries = load_registry()
                entry = entries.get(file_id)
                if entry and entry.get("path"):
                    path_str = entry["path"]
            except Exception as e:
                print(f"[on_load_saved_csv] registry read error: {e}")
        # Attempt to load CSV from file system
        df = None
        if path_str and os.path.exists(path_str):
            try:
                # Use load_and_meta to keep consistency with uploads (column renaming etc.)
                df, meta, _ = load_and_meta(Path(path_str), self.s.meta_json_dir)
            except Exception:
                try:
                    df = pd.read_csv(path_str)
                except Exception as e:
                    print(f"[on_load_saved_csv] CSV load error: {e}")
                    df = None
        # Fallback: read from SQL table
        if df is None:
            table = table_name_from_file(fname)
            try:
                df = run_sql(self.engine, f'SELECT * FROM "{table}"')
            except Exception as e:
                print(f"[on_load_saved_csv] DB load error: {e}")
                df = None
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            QMessageBox.information(self, "로드 실패", f"{fname}을(를) 불러올 수 없습니다.")
            return
        # Set current dataset and visualizer
        self.current_df = df
        try:
            self.visualizer = AnalysisVisualizer(df)
        except Exception as e:
            print(f"[on_load_saved_csv] visualizer init error: {e}")
            self.visualizer = None
        # Also assign to viz context used by viz tab
        self.df_viz = df
        # Reset simulation and hide controls
        self._stop_simulation_if_running()
        self.sim_controls_widget.hide()
        # Update viz figure to prompt user to select analysis
        self.viz_stack.setCurrentIndex(0)
        self.viz_fig.clear()
        ax = self.viz_fig.add_subplot(111)
        ax.text(0.5, 0.5, f"'{fname}' loaded.\nPlease select an analysis.", ha='center', va='center')
        ax.axis('off')
        self.viz_canvas.draw()
        # Enable analysis buttons
        for btn in [self.btn_run_stability, self.btn_run_correlation, self.btn_run_3d_path,
                    self.btn_run_simulation, self.btn_run_integrated, self.btn_run_aw_volume]:
            btn.setEnabled(True)
        # Disable ask LLM about viz until a specific viz is generated
        self.btn_ask_llm_about_viz.setEnabled(False)
        # Update chat
        self.chat.add_bot(f"✅ '{fname}' 파일을 불러왔습니다. 분석을 선택하세요.")

    # 시작하기: 심층 리포트 모드 진입
    def start_deep_report(self):
        """
        Initiate deep report mode. Prompt the user to provide context for an in-depth report.
        """
        if self.current_df is None:
            # If no dataset is loaded, attempt to load all selected files
            try:
                selected = self.get_selected_filenames()
            except Exception:
                selected = []
            if selected:
                # Reset current_dfs list
                self.current_dfs = []
                for fname in selected:
                    # Attempt to load DataFrame (similar to on_load_saved_csv logic)
                    path_str = None
                    file_id = self.file_ids.get(fname)
                    if file_id:
                        try:
                            entries = load_registry()
                            entry = entries.get(file_id)
                            if entry and entry.get("path"):
                                path_str = entry["path"]
                        except Exception:
                            pass
                    df_loaded = None
                    if path_str and os.path.exists(path_str):
                        try:
                            df_loaded, meta, _ = load_and_meta(Path(path_str), self.s.meta_json_dir)
                        except Exception:
                            try:
                                df_loaded = pd.read_csv(path_str)
                            except Exception:
                                df_loaded = None
                    if df_loaded is None:
                        # Fallback to DB table
                        try:
                            table = table_name_from_file(fname)
                            df_loaded = run_sql(self.engine, f'SELECT * FROM "{table}"')
                        except Exception:
                            df_loaded = None
                    if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty:
                        self.current_dfs.append((df_loaded, fname))
                # If at least one dataset loaded, set current_df to the first and update visualizer
                if self.current_dfs:
                    # Use the first loaded dataset as current_df for visualization
                    first_df, first_name = self.current_dfs[0]
                    self.current_df = first_df
                    self.df_viz = self.current_df
                    try:
                        self.visualizer = AnalysisVisualizer(self.current_df)
                    except Exception:
                        self.visualizer = None
                    # Update viz tab UI to indicate the first file loaded
                    try:
                        self._stop_simulation_if_running()
                        self.sim_controls_widget.hide()
                        self.viz_stack.setCurrentIndex(0)
                        self.viz_fig.clear()
                        ax = self.viz_fig.add_subplot(111)
                        ax.text(0.5, 0.5, f"'{first_name}' loaded.\nPlease select an analysis.", ha='center', va='center')
                        ax.axis('off')
                        self.viz_canvas.draw()
                        for btn in [self.btn_run_stability, self.btn_run_correlation, self.btn_run_3d_path,
                                    self.btn_run_simulation, self.btn_run_integrated, self.btn_run_aw_volume]:
                            btn.setEnabled(True)
                        self.btn_ask_llm_about_viz.setEnabled(False)
                    except Exception:
                        pass
            # If still no dataset loaded, notify and abort
            if self.current_df is None:
                QMessageBox.information(
                    self, "심층 리포트", "데이터가 로드되어 있지 않습니다. 먼저 CSV 파일을 선택하거나 업로드하세요.")
                return
        # Activate deep report mode
        self.in_deep_report = True
        self.deep_report_inputs = []
        # Let user know
        self.chat.add_bot("심층 리포트 작성을 시작합니다. 보고서에 포함할 정보나 관점을 말씀해주세요.\n완료 시 '끝' 또는 '완료'라고 입력해주세요.")
        # Switch to report tab for clarity
        idx = self.tabs.indexOf(self.report_tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    # 완료: 심층 리포트 생성
    def generate_deep_report(self):
        """
        Compile a deep report from the current dataset and user-provided context using the LLM.
        """
        # Exit deep report mode
        self.in_deep_report = False
        # If no current dataset, cannot generate deep report
        if self.current_df is None:
            QMessageBox.information(self, "심층 리포트", "데이터가 로드되어 있지 않습니다.")
            return
        # Prepare summaries for one or multiple datasets
        dataset_summaries: List[str] = []
        # If multiple datasets loaded, iterate over them; otherwise use current_df
        if self.current_dfs:
            for df_i, name_i in self.current_dfs:
                # Summarise each dataset individually
                try:
                    missing_total_i = int(df_i.isna().sum().sum())
                except Exception:
                    missing_total_i = 0
                try:
                    desc_df_i = df_i.describe(include="all")
                    desc_str_i = desc_df_i.to_string(max_cols=6, max_rows=20)
                except Exception:
                    desc_str_i = ""
                try:
                    preview_df_i = df_i.head(10)
                    preview_str_i = preview_df_i.to_string(index=False)
                except Exception:
                    preview_str_i = ""
                summary = (
                    f"[{name_i}]\n"
                    f"- 총 행수: {len(df_i)}, 총 열수: {df_i.shape[1]}, 결측치 총합: {missing_total_i}\n"
                    f"- 통계 요약:\n{desc_str_i}\n"
                    f"- 상위 10행 미리보기:\n{preview_str_i}\n"
                )
                dataset_summaries.append(summary)
            # If more than one dataset, compute comparative statistics across datasets
            if len(self.current_dfs) > 1:
                try:
                    import numpy as _np  # local import to avoid global dependency issues
                    # Determine common numeric columns across all datasets
                    numeric_sets = []
                    for df_i, _name_i in self.current_dfs:
                        try:
                            numeric_cols = set(df_i.select_dtypes(include="number").columns)
                        except Exception:
                            numeric_cols = set()
                        numeric_sets.append(numeric_cols)
                    common_cols = set.intersection(*numeric_sets) if numeric_sets else set()
                    comparison_lines: List[str] = []
                    # Limit number of columns to compare for brevity
                    max_cols_to_compare = 10
                    col_count = 0
                    for col in sorted(common_cols):
                        means = []
                        for df_i, name_i in self.current_dfs:
                            try:
                                mean_val = float(_np.nanmean(_np.asarray(df_i[col], dtype=float)))
                            except Exception:
                                mean_val = float('nan')
                            means.append((name_i, mean_val))
                        # Remove NaN entries
                        means_filtered = [(n, m) for n, m in means if m == m]
                        if len(means_filtered) < 2:
                            continue
                        # Sort by mean value descending
                        means_sorted = sorted(means_filtered, key=lambda x: x[1], reverse=True)
                        top_name, top_val = means_sorted[0]
                        bottom_name, bottom_val = means_sorted[-1]
                        diff_val = top_val - bottom_val
                        comparison_lines.append(
                            f"- '{col}' 컬럼 평균: {top_name}({top_val:.3g}) > {bottom_name}({bottom_val:.3g}), 차이 {diff_val:.3g}"
                        )
                        col_count += 1
                        if col_count >= max_cols_to_compare:
                            break
                    if comparison_lines:
                        comparison_text = "\n".join(comparison_lines)
                        dataset_summaries.append(
                            "[데이터 간 비교] (공통 숫자 컬럼 평균 비교)\n" + comparison_text + "\n"
                        )
                except Exception:
                    # On any error, silently ignore comparative summary
                    pass
        else:
            # Fallback: use current_df only
            df = self.current_df
            try:
                missing_total = int(df.isna().sum().sum())
            except Exception:
                missing_total = 0
            try:
                desc_df = df.describe(include="all")
                desc_str = desc_df.to_string(max_cols=6, max_rows=20)
            except Exception:
                desc_str = ""
            try:
                preview_df = df.head(10)
                preview_str = preview_df.to_string(index=False)
            except Exception:
                preview_str = ""
            summary = (
                f"[{getattr(self, 'selected_dataset_name', '데이터')}]\n"
                f"- 총 행수: {len(df)}, 총 열수: {df.shape[1]}, 결측치 총합: {missing_total}\n"
                f"- 통계 요약:\n{desc_str}\n"
                f"- 상위 10행 미리보기:\n{preview_str}\n"
            )
            dataset_summaries.append(summary)
        # Combine user context
        user_context = "\n".join(self.deep_report_inputs).strip()
        if not user_context:
            user_context = "(사용자가 추가 정보를 제공하지 않았습니다.)"
        # Build overall dataset summary block
        combined_summary = "\n".join(dataset_summaries)
        # Build prompt for the LLM to generate a deep report
        prompt = (
            "당신은 제조 공정 데이터 분석을 수행하는 전문 리포트 작성자입니다. "
            "아래 제공된 사용자 입력과 데이터 요약들을 바탕으로 심층 보고서를 작성하세요.\n"
            "보고서는 한국어로 작성하며, 각 섹션을 명확한 제목으로 구분하고, 데이터 기반 통찰과 해석, 제한 사항 및 추천을 포함해야 합니다.\n\n"
            f"[사용자 입력]\n{user_context}\n\n"
            f"[데이터 요약]\n{combined_summary}\n\n"
            "위 정보를 바탕으로 심층 분석 보고서를 작성해주세요."
        )
        # Invoke the LLM to generate the report
        try:
            deep_report_text = self.llm.invoke(prompt).content
        except Exception as e:
            deep_report_text = f"심층 리포트 생성 중 오류가 발생했습니다: {e}"
        # Display the report in the report tab
        try:
            self.report.setPlainText(deep_report_text)
        except Exception:
            self.report.setPlainText(str(deep_report_text))
        # Notify via chat
        self.chat.add_bot("📄 심층 리포트가 생성되었습니다. 보고서 탭에서 확인하세요.")
        # Ensure report tab is active
        idx = self.tabs.indexOf(self.report_tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    # 고급 시각화 처리
    def show_visualization(self, viz_type: str, variables: List[str] | None = None):
        """
        Generate and display advanced visualizations in the LLM results area based on the
        provided viz_type and optional variables list. Uses self.visualizer and self.current_df.

        Supported viz_type values:
          - 'correlation': 상관관계 대시보드
          - '3d': 3D 경로 (정적)
          - 'simulation': 공정 시뮬레이션 (메시지로 안내)
          - 'aw': A*W 적층 부피 (메시지로 안내)
          - 'mpt_time': 시간 대비 MPT 변화 그래프
          - 'custom_time': multi-series line chart with a time column and multiple numeric y variables (variables param)
          - 'custom_3d': 3D scatter plot using three variables for x, y, z axes (variables param)
          - other: 메시지로 안내
        """
        # Ensure there is a dataset to visualize. If none loaded, attempt to load from selected data.
        try:
            if not self.ensure_dataset_loaded():
                # Still no dataset; notify the user
                QMessageBox.information(self, "시각화", "먼저 CSV 파일을 선택하거나 업로드하여 분석을 진행하세요.")
                return
        except Exception:
            # On error, fallback to user message
            QMessageBox.information(self, "시각화", "먼저 CSV 파일을 선택하거나 업로드하여 분석을 진행하세요.")
            return
        # Clear existing figure
        try:
            self.adv_fig.clear()
        except Exception:
            self.adv_fig = plt.figure()
            self.adv_canvas = FigureCanvas(self.adv_fig)
        success = True
        try:
            if viz_type == 'correlation':
                self.visualizer.plot_correlation_dashboard(self.adv_fig)
            elif viz_type == '3d':
                self.visualizer.plot_3d_path(self.adv_fig)
            elif viz_type == 'simulation':
                # Provide guidance that interactive simulation is only available in the Viz tab
                ax = self.adv_fig.add_subplot(111)
                ax.text(0.5, 0.5, "공정 시뮬레이션은 시각화 탭에서 실행할 수 있습니다.", ha='center', va='center')
                ax.axis('off')
            elif viz_type == 'aw':
                # Provide guidance that AW dashboard is only available in the Viz tab
                ax = self.adv_fig.add_subplot(111)
                ax.text(0.5, 0.5, "A*W 대시보드는 시각화 탭에서 실행할 수 있습니다.", ha='center', va='center')
                ax.axis('off')
            elif viz_type == 'mpt_time':
                # Draw MPT vs Time line chart using relative time in seconds
                # Find columns for time and MPT using flexible substring matching
                # Build a lowercase mapping for quick lookup
                cols_lower = {c.lower(): c for c in self.current_df.columns}
                # Determine a time-like column: exact match or containing keywords
                time_col = None
                # First try common names
                for cand in ["time", "timestamp", "date", "datetime", "ts"]:
                    if cand in cols_lower:
                        time_col = cols_lower[cand]
                        break
                # If not found, search for any column containing 'time' or 'date'
                if time_col is None:
                    for c in self.current_df.columns:
                        cl = c.lower()
                        if 'time' in cl or 'date' in cl or 'ts' in cl:
                            time_col = c
                            break
                # Determine MPT-like column: exact or containing 'mpt'
                mpt_col = None
                # Try common names
                for cand in ["mpt", "mp_t", "mp.t", "m.p.t", "mpttemp"]:
                    if cand in cols_lower:
                        mpt_col = cols_lower.get(cand)
                        if mpt_col:
                            break
                # Search for any column containing 'mpt'
                if mpt_col is None:
                    for c in self.current_df.columns:
                        if 'mpt' in c.lower():
                            mpt_col = c
                            break
                if not time_col or not mpt_col:
                    ax = self.adv_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "'time' 또는 'MPT' 관련 컬럼을 찾을 수 없어 그래프를 그릴 수 없습니다.", ha='center', va='center')
                    ax.axis('off')
                else:
                    # Parse time column with custom logic: treat last 3 digits as ms for strings of pattern
                    def parse_custom_time(s: str):
                        # Handles strings like MM_DD_HH_MM_SS_MMM (ms)
                        if isinstance(s, str):
                            parts = s.split('_')
                            if len(parts) == 6 and len(parts[-1]) == 3 and parts[-1].isdigit():
                                s_mod = '_'.join(parts[:-1] + [parts[-1] + '000'])
                                try:
                                    return pd.to_datetime(s_mod, format="%m_%d_%H_%M_%S_%f")
                                except Exception:
                                    pass
                        try:
                            return pd.to_datetime(s)
                        except Exception:
                            return pd.NaT
                    t_series_raw = self.current_df[time_col]
                    # Use existing datetime if dtype is datetime, otherwise parse
                    if pd.api.types.is_datetime64_any_dtype(t_series_raw):
                        t_series = t_series_raw
                    else:
                        # Parse each element
                        t_series = t_series_raw.apply(parse_custom_time)
                    # Validate times and MPT values
                    valid_mask = t_series.notna()
                    mpt_series = pd.to_numeric(self.current_df.loc[valid_mask, mpt_col], errors="coerce")
                    t_series = t_series[valid_mask]
                    mpt_series = mpt_series[~mpt_series.isna()]
                    # Align time and mpt lengths (drop NaN mpt values)
                    if len(mpt_series) != len(t_series):
                        # Align index
                        common_index = t_series.index.intersection(mpt_series.index)
                        t_series = t_series.loc[common_index]
                        mpt_series = mpt_series.loc[common_index]
                    # Need at least two points
                    if len(t_series) < 2:
                        ax = self.adv_fig.add_subplot(111)
                        ax.text(0.5, 0.5, "유효한 시간 데이터가 충분하지 않습니다.", ha='center', va='center')
                        ax.axis('off')
                    else:
                        # Compute elapsed seconds relative to start
                        try:
                            elapsed = (t_series - t_series.iloc[0]).dt.total_seconds()
                        except Exception:
                            elapsed = None
                        if elapsed is None:
                            ax = self.adv_fig.add_subplot(111)
                            ax.text(0.5, 0.5, "시간 데이터를 해석할 수 없습니다.", ha='center', va='center')
                            ax.axis('off')
                        else:
                            ax = self.adv_fig.add_subplot(111)
                            ax.plot(elapsed, mpt_series.reset_index(drop=True), marker='o')
                            ax.set_xlabel("Elapsed time (s)")
                            ax.set_ylabel("MPT")
                            ax.set_title("MPT vs Time")
                            ax.grid(True)
            elif viz_type == 'custom_time':
                # Multi-series line chart versus time for user-selected variables
                # variables[0] should be time, rest are y variables
                if not variables or len(variables) < 2:
                    ax = self.adv_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "적절한 변수 목록이 없어 시각화할 수 없습니다.", ha='center', va='center')
                    ax.axis('off')
                else:
                    time_col = variables[0]
                    y_vars = variables[1:]
                    # Build time series using parse_custom_time similar to MPT
                    def parse_custom_time(s: str):
                        if isinstance(s, str):
                            parts = s.split('_')
                            if len(parts) == 6 and len(parts[-1]) == 3 and parts[-1].isdigit():
                                s_mod = '_'.join(parts[:-1] + [parts[-1] + '000'])
                                try:
                                    return pd.to_datetime(s_mod, format="%m_%d_%H_%M_%S_%f")
                                except Exception:
                                    pass
                        try:
                            return pd.to_datetime(s)
                        except Exception:
                            return pd.NaT
                    try:
                        t_series_raw = self.current_df[time_col]
                    except Exception:
                        t_series_raw = None
                    if t_series_raw is None:
                        ax = self.adv_fig.add_subplot(111)
                        ax.text(0.5, 0.5, "시간 변수를 찾을 수 없어 시각화할 수 없습니다.", ha='center', va='center')
                        ax.axis('off')
                    else:
                        if pd.api.types.is_datetime64_any_dtype(t_series_raw):
                            t_series = t_series_raw
                        else:
                            t_series = t_series_raw.apply(parse_custom_time)
                        # Use relative seconds if datetime, else numeric directly
                        elapsed = None
                        if pd.api.types.is_datetime64_any_dtype(t_series):
                            try:
                                elapsed = (t_series - t_series.iloc[0]).dt.total_seconds()
                            except Exception:
                                elapsed = None
                        if elapsed is None:
                            try:
                                elapsed = pd.to_numeric(t_series, errors='coerce')
                            except Exception:
                                elapsed = None
                        if elapsed is None or elapsed.dropna().size < 2:
                            ax = self.adv_fig.add_subplot(111)
                            ax.text(0.5, 0.5, "시간 변수를 해석할 수 없거나 데이터가 부족합니다.", ha='center', va='center')
                            ax.axis('off')
                        else:
                            ax = self.adv_fig.add_subplot(111)
                            for var in y_vars:
                                try:
                                    y = pd.to_numeric(self.current_df[var], errors='coerce')
                                except Exception:
                                    y = None
                                if y is None:
                                    continue
                                # Align length
                                common_index = elapsed.dropna().index.intersection(y.dropna().index)
                                if len(common_index) < 2:
                                    continue
                                ax.plot(elapsed.loc[common_index], y.loc[common_index], marker='o', label=var)
                            if not ax.lines:
                                ax.text(0.5, 0.5, "유효한 y 변수가 없어 그래프를 그릴 수 없습니다.", ha='center', va='center')
                                ax.axis('off')
                            else:
                                ax.set_xlabel("Elapsed time (s)")
                                # Y-axis label omitted to avoid font issues; individual lines have legends
                                title = " vs Time: " + ", ".join(y_vars)
                                ax.set_title(title)
                                ax.legend()
                                ax.grid(True)
            elif viz_type == 'custom_3d':
                # 3D surface and contour visualizations using three variables (x, y, z)
                if not variables or len(variables) < 3:
                    ax = self.adv_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "3개 이상의 변수가 필요합니다.", ha='center', va='center')
                    ax.axis('off')
                else:
                    x_var, y_var, z_var = variables[:3]
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)
                    from matplotlib.tri import Triangulation
                    try:
                        x_raw = self.current_df[x_var]
                        y_raw = self.current_df[y_var]
                        z_raw = self.current_df[z_var]
                    except Exception:
                        x_raw = y_raw = z_raw = None
                    if x_raw is None or y_raw is None or z_raw is None:
                        ax = self.adv_fig.add_subplot(111)
                        ax.text(0.5, 0.5, "지정한 변수를 찾을 수 없습니다.", ha='center', va='center')
                        ax.axis('off')
                    else:
                        # Convert variables to numeric values; treat x as time-like if needed
                        def convert_series(s, is_time=False):
                            # Helper to convert series to numeric; if time-like, convert datetime to elapsed seconds
                            if is_time:
                                if pd.api.types.is_datetime64_any_dtype(s):
                                    try:
                                        return (s - s.iloc[0]).dt.total_seconds()
                                    except Exception:
                                        return pd.to_numeric(s, errors='coerce')
                                # Try parsing time strings
                                parsed = s.apply(lambda x: pd.to_datetime(x) if isinstance(x, str) else pd.NaT)
                                if parsed.notna().mean() > 0.5:
                                    try:
                                        return (parsed - parsed.iloc[0]).dt.total_seconds()
                                    except Exception:
                                        return pd.to_numeric(s, errors='coerce')
                                return pd.to_numeric(s, errors='coerce')
                            else:
                                return pd.to_numeric(s, errors='coerce')
                        is_time_like = any(k in x_var.lower() for k in ["time", "date", "datetime", "ts"])
                        x_num = convert_series(x_raw, is_time_like)
                        y_num = convert_series(y_raw, any(k in y_var.lower() for k in ["time", "date", "datetime", "ts"]))
                        z_num = pd.to_numeric(z_raw, errors='coerce')
                        # Create mask for valid rows
                        mask = (~x_num.isna()) & (~y_num.isna()) & (~z_num.isna())
                        x_vals = x_num[mask]
                        y_vals = y_num[mask]
                        z_vals = z_num[mask]
                        if len(x_vals) < 3:
                            ax = self.adv_fig.add_subplot(111)
                            ax.text(0.5, 0.5, "유효한 데이터가 충분하지 않습니다.", ha='center', va='center')
                            ax.axis('off')
                        else:
                            # Create triangulation for irregular data
                            tri = Triangulation(x_vals, y_vals)
                            # Prepare 3 subplots
                            self.adv_fig.clear()
                            # Wireframe / surface 1
                            ax1 = self.adv_fig.add_subplot(1, 3, 1, projection='3d')
                            ax1.plot_trisurf(tri, z_vals, linewidth=0.2, edgecolor='black', antialiased=True)
                            ax1.set_xlabel(x_var)
                            ax1.set_ylabel(y_var)
                            ax1.set_zlabel(z_var)
                            ax1.set_title('Wireframe')
                            # Contour-like surface 2 (colored)
                            ax2 = self.adv_fig.add_subplot(1, 3, 2, projection='3d')
                            # Use contour-like effect by drawing contour lines along z-axis with a colormap
                            try:
                                ax2.plot_trisurf(tri, z_vals, cmap='viridis', linewidth=0.0, antialiased=True)
                            except Exception:
                                ax2.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=5)
                            ax2.set_xlabel(x_var)
                            ax2.set_ylabel(y_var)
                            ax2.set_zlabel(z_var)
                            ax2.set_title('Surface')
                            # Color-coded surface 3
                            ax3 = self.adv_fig.add_subplot(1, 3, 3, projection='3d')
                            try:
                                ax3.plot_trisurf(tri, z_vals, cmap='plasma', linewidth=0.0, antialiased=True)
                            except Exception:
                                ax3.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='plasma', s=5)
                            ax3.set_xlabel(x_var)
                            ax3.set_ylabel(y_var)
                            ax3.set_zlabel(z_var)
                            ax3.set_title('Surface (Alt)')
            else:
                # Fallback for unsupported visualization types
                ax = self.adv_fig.add_subplot(111)
                ax.text(0.5, 0.5, "해당 시각화는 지원되지 않습니다.", ha='center', va='center')
                ax.axis('off')
        except Exception as e:
            success = False
            self.adv_fig.clear()
            ax = self.adv_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"시각화 오류: {e}", ha='center', va='center')
            ax.axis('off')
        # Redraw the canvas
        try:
            self.adv_canvas.draw()
        except Exception:
            pass
        # Show the advanced tab
        idx = self.tabs.indexOf(self.adv_canvas)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
        # Mark that an advanced visualization was shown for the current query
        self._advanced_triggered = True
        # 창 닫을 때 정리
    def closeEvent(self, event):
        try:
            # 1) 시뮬레이션 타이머 정지
            self._stop_simulation_if_running()

            # 2) PyVista 렌더러 안전 종료
            if self.active_plotter:
                try:
                    self.active_plotter.close()
                    self.active_plotter.deleteLater()
                except Exception:
                    pass
                self.active_plotter = None

            # 3) Matplotlib Figure 닫기 (메모리 릭 방지)
            plt.close('all')

            # 4) 히스토리 저장
            self.save_history()

        except Exception as e:
            print(f"[closeEvent 경고] {e}")

        super().closeEvent(event)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
