# app/main10_pdf_rag.py
# 최종 업데이트 : 2025-08-20
# PyQt5 기반 공정 데이터 LLM 분석기 (V4.0)
# - V3.1 기능 전체 포함
# - [신규] 3번째 탭: PDF 문서 대상 RAG 챗봇 기능 추가
# - 문서 전용 벡터DB 분리(vector_db_dir/docs), 드래그&드롭/진행률/취소, 삭제, Evidence 패널

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
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem, QSlider, QStackedWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor

# --- PDF RAG 기능 추가에 필요한 라이브러리 ---
from pypdf import PdfReader
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
from core.files_registry import upsert_entry
from core.analysis_visualizer import AnalysisVisualizer

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
            # kw에 progress_callback 키가 있으면 signal.emit로 치환
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

# ---------------- LLM prompt helpers (fixed & balanced) ----------------
def _tone_style(tone: str) -> str:
    return (
        "말투는 친근하고 공감 있게, 군더더기 없이 자연스럽게."
        if tone == "친근"
        else "말투는 단정하고 간결하며, 근거 중심으로 정확하게 설명한다. 불필요한 수식은 피한다."
    )

def llm_final_only(llm, question: str, df_snip: str, meta_snip: str, tone: str) -> str:
    """
    균형 버전:
    - 자료가 '전혀' 없을 때만 불가 문구를 출력.
    - 표/메타가 '조금이라도' 있으면 반드시 한 단락(3~5문장)으로 요약→근거→해석→한계 순서로 답함.
    - 영어 문서를 읽어도 출력은 항상 한국어. 기술 용어/약어/단위/코드는 원문 보존, 최초 1회 (원문) 병기 권장.
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
        "출력 규칙(데이터가 일부라도 있으면 반드시 따름):\n"
        "- 한 단락(3~5문장)으로 구성하고, 다음 흐름을 포함한다: 요약 → 핵심 근거(수치/항목 언급) → 짧은 해석 → 마지막 문장에 제한/가정.\n"
        "- 제목/불릿/섹션/이모지/Markdown 금지.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n\n"
        "출력: 위 규칙을 따르는 한국어 한 단락."
    )
    return llm.invoke(prompt).content

def llm_checks_only(llm, question: str, df_snip: str, meta_snip: str) -> str:
    """
    체크리스트는 너무 보수적으로 막지 않고, 항상 3~6개를 제안.
    - 영어 문서라도 출력은 한국어, 용어/단위는 원문 유지(최초 1회 (원문) 병기).
    - 실행 가능하고 구체적으로.
    """
    prompt = (
        "역할: 제조 공정 데이터 분석 점검관.\n"
        "언어 규칙:\n"
        "- 자료가 영어여도 이해는 영어 맥락으로 하되, 결과는 한국어로 작성한다.\n"
        "- 기술 용어·제품명·약어·코드·단위·수치는 원문을 보존하고, 최초 1회만 한국어 뒤 (원문)을 병기한다.\n"
        "출력 규칙:\n"
        "- 하이픈('- ') 불릿 리스트로 3~6개 항목을 제시한다. 머리말/제목/이모지/Markdown 금지.\n"
        "- 각 항목은 1~2문장, 데이터/메타에 기반한 실행 가능한 검증 방법을 제안한다.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n\n"
        "출력 예시 형식:\n"
        "- …\n- …\n- …"
    )
    return llm.invoke(prompt).content


# main10_pdf_rag.py 파일에서 이 함수를 찾아 통째로 교체하세요.

# --- 신규: PDF RAG 답변 생성을 위한 LLM 프롬프트 헬퍼 (개선 버전) ---
def llm_pdf_rag_answer(llm, question: str, context: str) -> str:
    """
    요약과 상세 설명을 명확히 분리하고, 구조화된 답변을 생성하는 프롬프트.
    - [요약]: 질문에 대한 가장 핵심적인 결론을 1~2 문장으로 압축.
    - [상세 설명]: 핵심 포인트를 바탕으로 컨텍스트를 인용하여 구체적으로 서술.
    - 출처는 컨텍스트에 포함된 '[출처: 파일명 | p.N]' 패턴을 수집하여 정리.
    """
    prompt = (
        "역할: 당신은 기술 문서를 기반으로 질문에 답변하는 전문 분석가입니다.\n"
        "언어 규칙:\n"
        "1. 최종 답변은 반드시 한국어로 작성해야 합니다.\n"
        "2. 기술 용어, 제품명, 코드, 단위 등은 번역하지 말고 원문을 그대로 사용하세요.\n"
        "3. 용어가 처음 나올 때만 '한국어 설명 (원문)' 형식으로 표기하고, 이후에는 한국어나 약어만 사용하세요.\n"
        "출력 규칙:\n"
        "1. 반드시 아래에 명시된 템플릿 구조와 섹션 제목([요약], [상세 설명] 등)을 정확히 따라야 합니다.\n"
        "2. 각 섹션의 지시사항을 철저히 이행하여 내용을 작성해야 합니다.\n"
        "3. 컨텍스트에 근거가 없는 내용은 절대로 추측해서 작성하지 마세요.\n"
        "4. 컨텍스트 각 부분의 끝에 있는 '[출처: 파일명 | p.페이지]'를 수집하여 [출처] 섹션에 중복 없이 기입하세요.\n\n"
        "--- 템플릿 및 섹션별 지시사항 ---\n"
        "[요약]\n"
        "# 지시: 질문에 대한 가장 핵심적인 결론을 1~2 문장으로 간결하게 압축하여 여기에 작성하세요.\n\n"
        
        "[상세 설명]\n"
        "# 지시: 위 요약 내용을 뒷받침하는 구체적인 근거를 컨텍스트에서 찾아 2~4개의 핵심 항목으로 나누어 상세하게 설명하세요. 각 항목은 번호(1., 2., ...)를 붙이고, 관련된 컨텍스트 내용을 직접 인용하여 논리적으로 서술하세요.\n"
        "1. (첫 번째 핵심 설명)\n"
        "2. (두 번째 핵심 설명)\n\n"

        "[관련 정보 / 제한 사항]\n"
        "# 지시: 질문과 직접적인 관련은 없지만 알아두면 좋은 추가 정보나, 컨텍스트에서 언급된 한계점/주의사항을 불릿(•)으로 간략히 정리하세요. 내용이 없다면 '해당 없음'이라고 작성하세요.\n"
        "• \n\n"

        "[출처]\n"
        "# 지시: 답변의 근거가 된 컨텍스트의 '[출처: ...]' 부분을 모두 수집하여 중복을 제거한 후, 불릿(•)으로 여기에 나열하세요.\n"
        "• \n"
        "-------------------------------------\n\n"
        f"[사용자 질문]\n{question}\n\n"
        f"[참고 문서 내용 (컨텍스트)]\n{context}\n"
    )
    return llm.invoke(prompt).content


# # --- 신규: PDF RAG 답변 생성을 위한 LLM 프롬프트 헬퍼 ---
# def llm_pdf_rag_answer(llm, question: str, context: str) -> str:
#     """
#     다국어 문서를 읽되(영문은 영문 그대로 이해/추론), 결과는 항상 한국어로 구조화.
#     - 기술 용어/제품명/약어/코드/단위는 원문 유지.
#     - 최초 1회만 한국어 뒤에 (원문) 병기. 예) 산업용 로봇 컨트롤러(KUKA System Software, KSS)
#     - HTML/Markdown/이모지 금지(현재 UI 라벨은 순수 텍스트가 가장 깔끔함).
#     - 섹션/불릿/번호 형식을 강제하여 가독성 향상.
#     - 컨텍스트에 근거 없으면 요약 섹션에 불가 문구만.
#     - 출처는 컨텍스트에 포함된 [파일명 | p.N] 패턴을 모아 중복 제거(최대 6개).
#     """
#     prompt = (
#         "역할: 기술 문서를 근거로 한국어 요약을 생성하는 분석 보조자.\n"
#         "언어 규칙:\n"
#         "- 문서가 영어여도 내용 이해와 추론은 영어 맥락을 유지하라. 하지만 최종 출력은 반드시 한국어로 작성한다.\n"
#         "- 기술 용어/제품명/약어/코드/파일명/단위/수치는 원문을 보존한다.\n"
#         "- 용어의 최초 등장 시에만 한국어 설명 뒤에 (원문)을 병기한다. 이후에는 한국어 또는 약어만 사용해도 된다.\n"
#         "- 코드/식별자/에러메시지는 번역하지 말고 원문 그대로 둔다.\n"
#         "출력 형식 규칙:\n"
#         "- HTML/Markdown/이모지 사용 금지. 섹션 제목과 불릿/번호만 사용.\n"
#         "- 각 불릿은 1~2문장, 군더더기 없이 간결하게.\n"
#         "- 컨텍스트에 없는 내용은 쓰지 말고, 없으면 '제공된 문서 내용으로는 답변할 수 없습니다.'라고만 적는다.\n"
#         "- 컨텍스트에 포함된 '[파일명 | p.N]' 표식을 찾아 출처 목록을 만들고 중복을 제거한다(최대 6개).\n\n"
#         "반드시 아래 템플릿을 그대로 따를 것:\n"
#         "[요약]\n"
#         "…한 줄 요약…\n\n"
#         "[핵심 포인트]\n"
#         "• …\n"
#         "• …\n"
#         "• …\n\n"
#         "[세부 설명]\n"
#         "1) …\n"
#         "2) …\n"
#         "3) …\n\n"
#         "[주의/제한]\n"
#         "• …  (없으면 '해당 문맥에서 명확히 언급되지 않음')\n\n"
#         "[출처]\n"
#         "• 파일명 | p.N\n"
#         "• 파일명 | p.N\n\n"
#         f"[질문]\n{question}\n\n"
#         f"[컨텍스트]\n{context}\n"
#     )
#     return llm.invoke(prompt).content




# ---------------- main window ----------------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def __init__(self):
        super().__init__()
        self.history: List[Tuple[str, str]] = []

        # 시뮬 상태/타이머
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setTimerType(Qt.PreciseTimer)
        self.simulation_timer.timeout.connect(self._update_simulation_frame)
        self.sim_handles = None
        self.sim_frame_index = 0
        self.active_plotter: QtInteractor | None = None  # 현재 렌더 대상(일반/통합)

        # PDF RAG 상태
        self.pdf_rag_history: List[Tuple[str, str]] = []
        self.pdf_rag_files = {}          # {filename: file_id}
        self.pdf_chunk_counts = {}       # {filename: n_chunks}

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

    def setup_llm_tab(self):
        layout = QHBoxLayout(self.llm_tab)
        left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        layout.addLayout(left, 2); layout.addLayout(center, 5); layout.addLayout(right, 3)

        left.addWidget(QLabel("📁 소스 파일 (RAG 및 SQL 대상)"))
        self.drop = DropArea(file_type="CSV"); self.drop.filesDropped.connect(self.handle_csv_paths); left.addWidget(self.drop)
        self.btn_upload = QPushButton("CSV 업로드"); self.btn_upload.clicked.connect(self.on_upload); left.addWidget(self.btn_upload)
        left.addWidget(QLabel("저장된 파일"))
        self.file_list = QListWidget(); left.addWidget(self.file_list, 1)
        self.btn_del = QPushButton("선택 삭제"); self.btn_del.clicked.connect(self.on_delete_files); left.addWidget(self.btn_del)

        center.addWidget(QLabel("💬 LLM 질의"))
        tone_row = QHBoxLayout(); tone_row.addWidget(QLabel("톤"))
        self.tone = QComboBox(); self.tone.addItems(["전문", "친근"]); tone_row.addWidget(self.tone)
        tone_row.addStretch(1); center.addLayout(tone_row)
        self.chat = ChatView(); center.addWidget(self.chat, 1)
        self.btn_clear_history = QPushButton("채팅 로그 초기화"); self.btn_clear_history.clicked.connect(self.on_clear_history); center.addWidget(self.btn_clear_history)
        send_row = QHBoxLayout()
        self.inp = QLineEdit(); self.inp.setPlaceholderText("질문을 입력하고 Enter…"); self.inp.returnPressed.connect(self.on_ask)
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask); self.status = QLabel("")
        send_row.addWidget(self.inp, 1); send_row.addWidget(self.btn_send); send_row.addWidget(self.status)
        center.addLayout(send_row)

        right.addWidget(QLabel("📊 LLM 결과/리포트"))
        self.tabs = QTabWidget(); right.addWidget(self.tabs, 1)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.tabs.addTab(self.tbl, "표(Table)")
        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig); self.tabs.addTab(self.canvas, "그래프(Chart)")
        self.evidence = QTextEdit(); self.evidence.setReadOnly(True); self.tabs.addTab(self.evidence, "근거(Evidence)")
        self.report = QTextEdit(); self.report.setReadOnly(True); self.tabs.addTab(self.report, "보고서(Report)")

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

    # --- 신규: PDF RAG 챗봇 탭 UI 설정 ---
    def setup_pdf_tab(self):
        layout = QHBoxLayout(self.pdf_tab)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 8)

        # 좌측: 파일 관리
        left.addWidget(QLabel("📂 PDF 문서 라이브러리"))

        # 드래그&드롭
        self.drop_pdf = DropArea(file_type="pdf")
        self.drop_pdf.filesDropped.connect(self.handle_pdf_paths)
        left.addWidget(self.drop_pdf)

        # 업로드/삭제
        self.btn_upload_pdf = QPushButton("PDF 업로드")
        self.btn_upload_pdf.clicked.connect(self.on_upload_pdf)
        left.addWidget(self.btn_upload_pdf)

        self.pdf_file_list = QListWidget()
        left.addWidget(self.pdf_file_list, 1)

        self.btn_del_pdf = QPushButton("선택 삭제")
        self.btn_del_pdf.clicked.connect(self.on_delete_pdf)
        left.addWidget(self.btn_del_pdf)

        # 우측: 채팅 + Evidence
        right.addWidget(QLabel("💬 PDF 내용에 대해 질문하기"))
        self.pdf_chat = ChatView()
        self.pdf_chat.add_bot("PDF 파일을 업로드하고 내용에 대해 질문해 주세요.")
        right.addWidget(self.pdf_chat, 1)

        # Evidence 탭
        self.pdf_tabs = QTabWidget()
        self.pdf_evidence = QTextEdit(); self.pdf_evidence.setReadOnly(True)
        self.pdf_tabs.addTab(self.pdf_evidence, "Evidence")
        right.addWidget(self.pdf_tabs, 1)

        # 상태/입력
        self.pdf_status = QLabel("")
        right.addWidget(self.pdf_status)

        send_row = QHBoxLayout()
        self.pdf_inp = QLineEdit()
        self.pdf_inp.setPlaceholderText("질문을 입력하고 Enter…")
        self.pdf_inp.returnPressed.connect(self.on_ask_pdf)
        self.btn_send_pdf = QPushButton("▶")
        self.btn_send_pdf.clicked.connect(self.on_ask_pdf)
        send_row.addWidget(self.pdf_inp, 1)
        send_row.addWidget(self.btn_send_pdf)
        right.addLayout(send_row)

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

        # [추가] 문서 전용 벡터DB (vector_db_dir/docs)
        docs_dir = Path(s.vector_db_dir) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_docs = build_chroma(self.emb, str(docs_dir))

        # PDF 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # LLM 보조
    def build_prompt(self, question: str) -> str:
        full_question = question
        if self.viz_context:
            full_question = f"[Current Analysis Context]\n{self.viz_context}\n\n[User's Question]\n{question}"
            self.viz_context = None
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

    # 파일 업로드/인덱싱 (CSV)
    def on_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
        if files:
            self.handle_csv_paths(files)

    def handle_csv_paths(self, paths: list[str]):
        prog = QProgressDialog("CSV 처리 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        ok, fail = 0, 0
        for i, p in enumerate(paths, 1):
            prog.setValue(i - 1); QApplication.processEvents()
            if prog.wasCanceled(): break
            try:
                df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)
                entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
                upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                self.file_ids[Path(p).name] = entry.file_id
                table = table_name_from_file(Path(p).name)
                ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)
                try:
                    if _build_meta_for_table:
                        sessions = _build_meta_for_table(self.s.db_url, table)
                        if _index_sessions: _index_sessions(self.s.db_url, str(self.s.vector_db_dir), sessions)
                except Exception as _e:
                    self.chat.add_bot(f"⚠️ 메타/인덱싱 경고: {Path(p).name}\n{_e}")
                self.csv_files.append((Path(p).name, df))
                it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
                self.chat.add_bot(f"✅ 업로드 완료: {Path(p).name}\n(table={table})"); ok += 1
            except Exception as e:
                self.chat.add_bot(f"❌ 업로드 실패: {p}\n{e}"); fail += 1
        prog.setValue(len(paths))
        QMessageBox.information(self, "완료", f"성공 {ok} / 실패 {fail}")
        self.update_report_summary()

    def on_delete_files(self):
        items = [self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).checkState() == Qt.Checked]
        if not items:
            return QMessageBox.information(self, "알림", "체크된 파일이 없습니다.")
        if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 파일을 삭제합니다. 계속할까요?") != QMessageBox.Yes:
            return
        for it in items:
            fname = it.text()
            self.csv_files = [(f, df) for f, df in self.csv_files if f != fname]
            self.file_list.takeItem(self.file_list.row(it))
            table = table_name_from_file(fname)
            try:
                with self.engine.begin() as c:
                    c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
            except Exception as e:
                self.chat.add_bot(f"⚠️ DB 테이블 삭제 경고: {table} / {e}")
            fid = self.file_ids.pop(fname, None)
            if fid:
                try: self.chroma.delete(ids=[f"{fid}:{i:04d}" for i in range(2000)])
                except Exception as e: self.chat.add_bot(f"⚠️ 임베딩 삭제 경고: {fname} / {e}")
        self.update_report_summary(); self.chat.add_bot("🗑️ 선택 파일 삭제 완료")

    # LLM 질의 (CSV/DB + 메타)
    def on_ask(self):
        q = self.inp.text().strip()
        if not q:
            return
        self.inp.clear(); self.chat.add_user(q)
        tone = self.tone.currentText(); self.set_busy(True)

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

            ev_lines = ["## 사용 근거"]
            if sql: ev_lines += ["### 사용 SQL", "```sql", sql.strip(), "```"]
            if isinstance(df, pd.DataFrame): ev_lines += ["### SQL 결과 개요", f"- 행 수: {len(df)}", f"- 열 수: {df.shape[1]}"]
            if docs:
                ev_lines.append("### RAG 근거(상위 문서 첫 줄)")
                for i, d in enumerate(docs[:5], 1):
                    ev_lines.append(f"{i}. {getattr(d, 'page_content', str(d)).splitlines()[0][:200]}")
            if checks_list: ev_lines += ["", "## 추가 확인 항목", checks_list]
            if err_sql and not sql: ev_lines += ["", "### SQL 생성/실행 참고", err_sql]
            return (q, final_text, df, sql, "\n".join(ev_lines))

        def _done(res, err):
            self.set_busy(False)
            if err:
                return QMessageBox.critical(self, "질의 오류", str(err))
            q, final_text, df, sql, evidence_text = res
            self.chat.add_bot(final_text)
            self.history.append((q, final_text)); self.save_history()
            if isinstance(df, pd.DataFrame):
                self.render_all(df, sql)
            self.evidence.setPlainText(evidence_text)

        run_in_thread(self, _task, _done)

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
                progress_callback(i - 1, f"{path.name} 텍스트 추출 중...")
                try:
                    reader = PdfReader(path)
                    chunks_all = []
                    for pi, page in enumerate(reader.pages, start=1):
                        if cancel_event.is_set():
                            break
                        txt = page.extract_text() or ""
                        if not txt.strip():
                            continue
                        # 페이지 텍스트 → 분할 → [파일명 | p.N] 프리픽스 부착
                        chunks = self.text_splitter.split_text(txt)
                        chunks = [f"[{path.name} | p.{pi}]\n{c}" for c in chunks]
                        chunks_all.extend(chunks)

                    if cancel_event.is_set():
                        break
                    if not chunks_all:
                        raise ValueError("텍스트를 추출할 수 없거나 문서가 비어 있습니다.")

                    # 파일 ID 발급 및 업서트 (문서 전용 Chroma)
                    entry = upsert_entry(path, rows=len(reader.pages), cols=0, status="indexed")
                    file_id = entry.file_id
                    ids = [f"{file_id}:{idx:04d}" for idx in range(len(chunks_all))]
                    self.chroma_docs.add_texts(texts=chunks_all, ids=ids)

                    results.append((path.name, file_id, len(chunks_all)))
                    ok += 1
                except Exception as e:
                    fail.append(f"{path.name}: {e}")
            return ok, fail, results

        def _done(res, err):
            prog.close()
            if err:
                QMessageBox.critical(self, "PDF 처리 오류", str(err))
                return

            ok, fail, results = res
            # UI 반영
            existing = [self.pdf_file_list.item(i).text() for i in range(self.pdf_file_list.count())]
            for name, fid, n_chunks in results:
                self.pdf_rag_files[name] = fid
                self.pdf_chunk_counts[name] = n_chunks
                if name not in existing:
                    it = QListWidgetItem(name)
                    it.setCheckState(Qt.Unchecked)
                    self.pdf_file_list.addItem(it)
                self.pdf_chat.add_bot(f"업로드 완료: {name} (chunks={n_chunks})")

            summary = f"성공: {ok}건"
            if fail:
                summary += f"\n실패: {len(fail)}건\n" + "\n".join(fail)
            QMessageBox.information(self, "PDF 처리 완료", summary)

        worker, thread = run_in_thread(self, _task, _done, paths, progress_callback=None, cancel_event=cancel_event)
        worker.progress.connect(lambda i, msg: (prog.setValue(i), prog.setLabelText(msg)))

    def on_delete_pdf(self):
        items = [self.pdf_file_list.item(i) for i in range(self.pdf_file_list.count())
                 if self.pdf_file_list.item(i).checkState() == Qt.Checked]
        if not items:
            return QMessageBox.information(self, "알림", "체크된 문서가 없습니다.")
        if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 문서를 삭제합니다. 계속할까요?") != QMessageBox.Yes:
            return

        for it in items:
            fname = it.text()
            row = self.pdf_file_list.row(it)
            self.pdf_file_list.takeItem(row)

            fid = self.pdf_rag_files.pop(fname, None)
            n_chunks = self.pdf_chunk_counts.pop(fname, 0)
            if fid and n_chunks > 0:
                try:
                    ids = [f"{fid}:{i:04d}" for i in range(n_chunks)]
                    self.chroma_docs.delete(ids=ids)
                except Exception as e:
                    self.pdf_chat.add_bot(f"임베딩 삭제 경고: {fname} / {e}")

        self.pdf_chat.add_bot("선택 문서 삭제 완료")

    def on_ask_pdf(self):
        q = self.pdf_inp.text().strip()
        if not q:
            return

        self.pdf_inp.clear()
        self.pdf_chat.add_user(q)
        self.set_pdf_busy(True)

        def _task():
            # 1) 문서 전용 컬렉션에서 RAG 검색
            try:
                docs = self.chroma_docs.similarity_search(q, k=6)
            except Exception:
                docs = []
            # 2) 컨텍스트 구성 (상위 4~6개)
            context = "\n\n---\n\n".join(getattr(d, "page_content", str(d)) for d in docs[:6])
            # 3) LLM 호출
            answer = llm_pdf_rag_answer(self.llm, q, context)
            # Evidence 출력용
            ev_lines = ["## Retrieved Evidence"]
            for i, d in enumerate(docs[:6], 1):
                first_line = getattr(d, "page_content", str(d)).splitlines()[0][:200]
                ev_lines.append(f"{i}. {first_line}")
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

    def closeEvent(self, event):
        self._stop_simulation_if_running()
        self.save_history()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
