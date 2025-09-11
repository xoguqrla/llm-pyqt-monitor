# main10_csv_only.py - 이었던 것. 지금은 LLM_main.py
# 최종 업데이트 : 2025-09-05
# PyQt5 기반 공정 데이터 LLM 분석기 (V1.0 - CSV 전용)
# - 기존 V4.0에서 2, 3번째 탭(시각화, PDF RAG) 관련 기능이 모두 제거된 버전입니다.
# - 오직 LLM 기반 CSV 분석 기능에만 집중합니다.

from __future__ import annotations
import sys, traceback, html
from pathlib import Path
import json
import os
from typing import List, Tuple

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem, QStackedWidget,
    QSplitter
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

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
#.

# ---------------- global excepthook ---------------- 
# 예외 처리 훅
    # 프로그램 전역에서 발생하는 에러를 PyQt 메시지 박스로 표시
    # GUI 프로그램 안정성을 위한 기본 장치.
def _excepthook(et, ev, tb):
    msg = "".join(traceback.format_exception(et, ev, tb))[-4000:]
    print(msg, file=sys.stderr)
    try:
        QMessageBox.critical(None, "Unhandled Error", msg)
    except Exception:
        pass
sys.excepthook = _excepthook
#.

# ---------------- threading helper ----------------
# 스레딩 유틸
    # PyQt5의 Worker-Thread 패턴.
    # LLM 호출, DB 질의, 임베딩 빌드 등 시간이 긴 작업을 메인 스레드(UI)와 분리해서 실행.
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
            
    # run_in_thread 함수로 간편하게 비동기 작업을 실행 가능.
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
#.

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
#.

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
#.

# ---------------- LLM prompt helpers (CSV/DB용) ----------------
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
#.

# ---------------- main window ----------------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def __init__(self):
        super().__init__()
        self.history: List[Tuple[str, str]] = []

        # Deep report state and current DataFrame
        self.in_deep_report: bool = False
        self.deep_report_inputs: List[str] = []
        self.current_df: pd.DataFrame | None = None
        # For deep report: allow multiple datasets to be loaded
        self.current_dfs: List[Tuple[pd.DataFrame, str]] = []

        self.setupUi()
        self.init_backend()
        self.load_history()
        self.repopulate_chat()

    def setupUi(self):
        self.setWindowTitle("공정 데이터 LLM 분석기 (CSV 전용)")
        self.resize(1700, 900)

        # Main layout will be the content of the former "LLM Tab"
        self.setup_llm_tab()

    def setup_llm_tab(self):
        """
        Construct the UI for the LLM-based CSV analysis page.
        """
        # Main horizontal layout for the window
        main_layout = QHBoxLayout(self)
        
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

        self.csv_new_list = QListWidget()
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

        self.csv_saved_list = QListWidget()
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
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.addTab(self.canvas, "그래프(Chart)")
        # (3) Evidence: SQL 및 RAG 근거, 프리뷰 출력
        self.evidence = QTextEdit()
        self.evidence.setReadOnly(True)
        self.tabs.addTab(self.evidence, "근거(Evidence)")
        # (4) 보고서: 기본 보고서 + 심층 리포트 요청 버튼을 포함하는 탭
        self.report_tab = QWidget()
        report_layout = QVBoxLayout(self.report_tab)
        self.btn_deep_report = QPushButton("심층 리포트 요청")
        self.btn_deep_report.clicked.connect(self.start_deep_report)
        report_layout.addWidget(self.btn_deep_report)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        report_layout.addWidget(self.report)
        self.tabs.addTab(self.report_tab, "보고서(Report)")
        # (5) 전문가 코멘트: 도메인 지식 메모장 탭
        self.expert_tab = QWidget()
        expert_layout = QVBoxLayout(self.expert_tab)
        self.expert_input = QTextEdit()
        self.expert_input.setPlaceholderText("전문가의 코멘트를 입력하세요...")
        expert_layout.addWidget(self.expert_input)
        self.btn_save_expert = QPushButton("저장")
        self.btn_save_expert.clicked.connect(self.on_save_expert_comment)
        expert_layout.addWidget(self.btn_save_expert)
        self.expert_scroll = QScrollArea()
        self.expert_scroll.setWidgetResizable(True)
        self.expert_container = QWidget()
        self.expert_container_layout = QVBoxLayout(self.expert_container)
        self.expert_container_layout.setContentsMargins(0, 0, 0, 0)
        self.expert_container_layout.setSpacing(4)
        self.expert_scroll.setWidget(self.expert_container)
        expert_layout.addWidget(self.expert_scroll, 1)
        self.expert_tab_index = self.tabs.addTab(self.expert_tab, "전문가 코멘트")
        QTimer.singleShot(0, self.load_expert_comments)

        # 시작 시 저장된 CSV 목록 채우기
        QTimer.singleShot(0, self.refresh_csv_saved_list)

        # ----------------------------------------------------------------------
        # Combine centre and right layouts into widgets and wrap in a splitter
        # ----------------------------------------------------------------------
        center_widget = QWidget(); center_widget.setLayout(center_layout)
        right_widget = QWidget(); right_widget.setLayout(right_layout)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(2, 3)
        left_widget = QWidget(); left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(260)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(splitter, 1)
#.

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
        self.chroma = build_chroma(self.emb, s.vector_db_dir)

        # 전문가 코멘트 테이블 보장
        try:
            with self.engine.begin() as c:
                dialect = getattr(self.engine, "dialect", None)
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    ddl = (
                        "\n"
                        "CREATE TABLE IF NOT EXISTS expert_comments (\n"
                        "    id SERIAL PRIMARY KEY,\n"
                        "    content TEXT NOT NULL,\n"
                        "    created_at TEXT\n"
                        ")\n"
                    )
                else:
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
#.

    # LLM 보조
    def build_prompt(self, question: str) -> str:
        full_question = question
        try:
            selected = self.get_selected_filenames()
        except Exception:
            selected = []
        if selected:
            try:
                summary = self.get_selected_files_summary()
            except Exception:
                summary = ""
            selected_context = "[선택된 데이터 파일 요약]\n" + summary + "\n\n"
            full_question = selected_context + full_question
        
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
        self.btn_send.setEnabled(not busy)
        self.inp.setReadOnly(busy)
        self.status.setText("🤖 답변 생성 중…" if busy else "")

    def refresh_csv_saved_list(self):
        """files_registry.json을 읽어 저장된 CSV 목록을 갱신."""
        try:
            entries = load_registry()
        except Exception as e:
            QMessageBox.critical(self, "CSV 레지스트리 로딩 오류", str(e))
            return

        self.file_ids = {} # {filename: file_id}
        self.csv_saved_list.clear()
        for file_id, data in entries.items():
            path_str = data.get("path")
            if not path_str: continue
            fname = Path(path_str).name
            
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
        from threading import Event
        prog = QProgressDialog("CSV 처리 및 인덱싱 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        cancel_event = Event()
        try: prog.canceled.connect(cancel_event.set)
        except: pass

        def _task(file_paths, progress_callback, cancel_event):
            ok, fail, results = 0, [], []
            for i, p_str in enumerate(file_paths, 1):
                if cancel_event.is_set():
                    break
                path = Path(p_str)
                progress_callback(i - 1, f"{path.name} 처리 중...")
                try:
                    df, meta, _ = load_and_meta(path, self.s.meta_json_dir)
                    entry = upsert_entry(path, rows=meta["rows"], cols=meta["cols"], status="indexed")
                    upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                    table = table_name_from_file(path.name)
                    ingest_df(self.engine, df, table)
                    ensure_indexes(self.engine, table)
                    
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
            for (name, rows, cols, missing_total, desc_str, preview_str, df) in results:
                it = QListWidgetItem(f"{name} ({rows}x{cols})")
                it.setData(Qt.UserRole, name)
                it.setCheckState(Qt.Unchecked)
                self.csv_new_list.addItem(it)
                self.chat.add_bot(f"✅ 업로드 완료: {name}")
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
                    self.report.append(summary_text)
                except Exception:
                    current_text = self.report.toPlainText()
                    self.report.setPlainText(current_text + "\n" + summary_text)
                last_df = df
            
            self.refresh_csv_saved_list()
            
            if last_df is not None:
                self.current_df = last_df
                try:
                    self.render_all(last_df, sql=None)
                except Exception:
                    pass
            
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
                table = table_name_from_file(fname)
                with self.engine.begin() as c:
                    c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
                
                fid = self.file_ids.get(fname)
                if fid:
                    self.chroma.delete(ids=[f"{fid}:{i:04d}" for i in range(2000)])
                
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

        self.refresh_csv_saved_list()
        for i in reversed(range(self.csv_new_list.count())):
            it = self.csv_new_list.item(i)
            if it.data(Qt.UserRole) in fnames_to_delete:
                self.csv_new_list.takeItem(i)

        if errors:
            self.chat.add_bot("일부 삭제 실패:\n" + "\n".join(errors))
        else:
            self.chat.add_bot("🗑️ 선택 파일 삭제 완료")
#.
    #   on_ask 함수 시작~~
    # LLM 질의 (CSV/DB + 메타)
    def on_ask(self):
        q = self.inp.text().strip()
        if not q:
            return
        
        self.inp.clear()
        self.chat.add_user(q)
        
        if self.in_deep_report:
            kw = q.strip().lower()
            if kw in ["끝", "완료", "finish", "done"]:
                self.generate_deep_report()
            else:
                self.deep_report_inputs.append(q)
                self.chat.add_bot("계속 입력해주세요. 보고서를 완료하려면 '끝' 또는 '완료'라고 입력하세요.")
            return

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

            ev_lines = ["## 사용 근거"]
            if sql:
                ev_lines += ["### 사용 SQL", "```sql", sql.strip(), "```"]
            if isinstance(df, pd.DataFrame):
                ev_lines += [
                    "### SQL 결과 개요",
                    f"- 행 수: {len(df)}",
                    f"- 열 수: {df.shape[1]}"
                ]
                try:
                    preview = df.head(10).to_string(index=False)
                    ev_lines += ["", "### SQL 결과 미리보기", preview]
                except Exception:
                    pass
            
            if docs:
                ev_lines.append("### RAG 근거(상위 문서 첫 줄)")
                for i, d in enumerate(docs[:5], 1):
                    ev_lines.append(f"{i}. {getattr(d, 'page_content', str(d)).splitlines()[0][:200]}")
            
            if checks_list:
                ev_lines += ["", "## 추가 확인 항목", checks_list]
            
            if err_sql and not sql:
                ev_lines += ["", "### SQL 생성/실행 참고", err_sql]
            
            return (q, final_text, df, sql, "\n".join(ev_lines))

        def _done(res, err):
            self.set_busy(False)
            if err:
                return QMessageBox.critical(self, "질의 오류", str(err))
            
            q, final_text, df, sql, evidence_text = res
            self.chat.add_bot(final_text)
            self.history.append((q, final_text))
            self.save_history()

            if isinstance(df, pd.DataFrame):
                self.current_df = df # Store the result for potential deep reports
                try:
                    self.render_all(df, sql)
                except Exception as e:
                    print(f"Render error: {e}")

            self.evidence.setPlainText(evidence_text)

            if isinstance(df, pd.DataFrame):
                idx = self.tabs.indexOf(self.tbl)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            else:
                idx = self.tabs.indexOf(self.evidence)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)

        run_in_thread(self, _task, _done)

    def render_all(self, df: pd.DataFrame, sql: str | None):
        view = df.head(self.MAX_ROWS_TABLE)
        plot_df = view.iloc[::max(1, len(view)//self.MAX_POINTS_PLOT)] if len(view) > self.MAX_POINTS_PLOT else view
        df_to_table(self.tbl, view)
        plot_df_line(self.ax, self.canvas, plot_df)
        self.last_df = df
#.

    # ------------------------------------------------------------------
    # 전문가 코멘트 관리 메서드들
    def load_expert_comments(self):
        layout = self.expert_container_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        comments = []
        try:
            with self.engine.begin() as c:
                rows = c.exec_driver_sql("SELECT id, content FROM expert_comments ORDER BY id").fetchall()
                comments = [(int(r[0]), r[1]) for r in rows]
        except Exception as e:
            print(f"[load_expert_comments error] {e}")
        
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
            btn_del.clicked.connect(lambda _, cid=cid: self.on_delete_expert_comment(cid))
            h_layout.addWidget(btn_del)
            layout.addWidget(fr)
        
        layout.addStretch(1)

    def on_save_expert_comment(self):
        content = self.expert_input.toPlainText().strip()
        if not content:
            QMessageBox.information(self, "입력 필요", "코멘트를 입력한 뒤 저장하세요.")
            return
        
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        
        try:
            with self.engine.begin() as c:
                dialect = getattr(self.engine, "dialect", None)
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    stmt = "INSERT INTO expert_comments (content, created_at) VALUES (%s, %s)"
                    params = (content, now)
                else:
                    stmt = "INSERT INTO expert_comments (content, created_at) VALUES (?, ?)"
                    params = (content, now)
                c.exec_driver_sql(stmt, params)
            
            self.expert_input.clear()
            self.load_expert_comments()
            QMessageBox.information(self, "저장", "코멘트가 저장되었습니다.")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"코멘트를 저장할 수 없습니다: {e}")

    def on_delete_expert_comment(self, comment_id: int):
        resp = QMessageBox.question(
            self, "삭제 확인", "정말 이 코멘트를 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return
        
        try:
            with self.engine.begin() as c:
                dialect = getattr(self.engine, "dialect", None)
                if dialect and getattr(dialect, "name", "").lower() == "postgresql":
                    stmt = "DELETE FROM expert_comments WHERE id = %s"
                    params = (comment_id,)
                else:
                    stmt = "DELETE FROM expert_comments WHERE id = ?"
                    params = (comment_id,)
                c.exec_driver_sql(stmt, params)
            
            self.load_expert_comments()
        except Exception as e:
            QMessageBox.warning(self, "삭제 실패", f"코멘트를 삭제할 수 없습니다: {e}")
#.

    # ------------------------------------------------------------------
    # 선택된 CSV 파일 명단을 반환
    def get_selected_filenames(self) -> List[str]:
        names: List[str] = []
        def collect_names(lst: QListWidget | None):
            if not lst: return
            for i in range(lst.count()):
                it = lst.item(i)
                if not it: continue
                checked = (it.checkState() == Qt.Checked)
                selected = it.isSelected()
                if checked or selected:
                    fname = it.data(Qt.UserRole) or it.text().split(' ')[0]
                    if fname:
                        names.append(fname)

        collect_names(getattr(self, 'csv_new_list', None))
        collect_names(getattr(self, 'csv_saved_list', None))
        
        if not names:
            for lst in (getattr(self, 'csv_new_list', None), getattr(self, 'csv_saved_list', None)):
                current = lst.currentItem() if lst else None
                if current:
                    fname = current.data(Qt.UserRole) or current.text().split(' ')[0]
                    if fname:
                        names.append(fname)
                        break
        return list(set(names)) # Return unique names

    def get_selected_files_summary(self) -> str:
        names = self.get_selected_filenames()
        if not names: return ""
        
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
#.
    
    # 심층 리포트
    def start_deep_report(self):
        if self.current_df is None:
            selected = self.get_selected_filenames()
            if not selected:
                QMessageBox.information(self, "심층 리포트", "데이터가 로드되어 있지 않습니다. 먼저 CSV 파일을 선택하거나 업로드하세요.")
                return
            
            self.current_dfs = []
            for fname in selected:
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
                        df_loaded, _, _ = load_and_meta(Path(path_str), self.s.meta_json_dir)
                    except Exception:
                        df_loaded = None
                if df_loaded is None:
                    try:
                        table = table_name_from_file(fname)
                        df_loaded = run_sql(self.engine, f'SELECT * FROM "{table}"')
                    except Exception:
                        df_loaded = None
                if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty:
                    self.current_dfs.append((df_loaded, fname))
            
            if self.current_dfs:
                self.current_df, _ = self.current_dfs[0]
            else:
                QMessageBox.information(self, "심층 리포트", "선택한 파일을 로드할 수 없습니다.")
                return

        self.in_deep_report = True
        self.deep_report_inputs = []
        self.chat.add_bot("심층 리포트 작성을 시작합니다. 보고서에 포함할 정보나 관점을 말씀해주세요.\n완료 시 '끝' 또는 '완료'라고 입력해주세요.")
        idx = self.tabs.indexOf(self.report_tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    def generate_deep_report(self):
        self.in_deep_report = False
        if self.current_df is None:
            QMessageBox.information(self, "심층 리포트", "데이터가 로드되어 있지 않습니다.")
            return

        dataset_summaries: List[str] = []
        datasets_to_summarize = self.current_dfs if self.current_dfs else [(self.current_df, "current_data")]
        
        for df_i, name_i in datasets_to_summarize:
            try:
                missing_total_i = int(df_i.isna().sum().sum())
                desc_df_i = df_i.describe(include="all")
                desc_str_i = desc_df_i.to_string(max_cols=6, max_rows=20)
                preview_df_i = df_i.head(10)
                preview_str_i = preview_df_i.to_string(index=False)
                summary = (
                    f"[{name_i}]\n"
                    f"- 총 행수: {len(df_i)}, 총 열수: {df_i.shape[1]}, 결측치 총합: {missing_total_i}\n"
                    f"- 통계 요약:\n{desc_str_i}\n"
                    f"- 상위 10행 미리보기:\n{preview_str_i}\n"
                )
                dataset_summaries.append(summary)
            except Exception as e:
                dataset_summaries.append(f"[{name_i}] 요약 생성 오류: {e}")

        user_context = "\n".join(self.deep_report_inputs).strip() or "(사용자가 추가 정보를 제공하지 않았습니다.)"
        combined_summary = "\n".join(dataset_summaries)
        
        prompt = (
            "당신은 제조 공정 데이터 분석을 수행하는 전문 리포트 작성자입니다. "
            "아래 제공된 사용자 입력과 데이터 요약들을 바탕으로 심층 보고서를 작성하세요.\n"
            "보고서는 한국어로 작성하며, 각 섹션을 명확한 제목으로 구분하고, 데이터 기반 통찰과 해석, 제한 사항 및 추천을 포함해야 합니다.\n\n"
            f"[사용자 입력]\n{user_context}\n\n"
            f"[데이터 요약]\n{combined_summary}\n\n"
            "위 정보를 바탕으로 심층 분석 보고서를 작성해주세요."
        )
        
        try:
            deep_report_text = self.llm.invoke(prompt).content
        except Exception as e:
            deep_report_text = f"심층 리포트 생성 중 오류가 발생했습니다: {e}"
            
        self.report.setPlainText(deep_report_text)
        self.chat.add_bot("📄 심층 리포트가 생성되었습니다. 보고서 탭에서 확인하세요.")
        idx = self.tabs.indexOf(self.report_tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    def closeEvent(self, event):
        try:
            plt.close('all')
            self.save_history()
        except Exception as e:
            print(f"[closeEvent 경고] {e}")
        super().closeEvent(event)
#.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())