# app/main9.py
# 최종 업데이트 : 2025-08-13
# PyQt5 기반 공정 데이터 LLM 분석기 (V2.2)
# - PyVista 기반, 재생/일시정지/초기화 기능이 포함된 안정적인 시뮬레이터 적용

# 주요 수정 사항
# __init__ 메서드 수정: setupUi()가 호출되기 전에 UI 위젯(self.chat 등)에 접근하는 repopulate_chat()이 먼저 호출되는 문제를 해결하기 위해, __init__ 내의 함수 호출 순서를 setupUi() -> init_backend() -> load_history() -> repopulate_chat()으로 명확하게 정리했습니다.

# _update_simulation_frame 안정성 강화: 시뮬레이션이 시작되기 전이나 데이터가 없는 상태에서 타이머가 실수로 호출될 경우를 대비하여, 함수의 시작 부분에 self.sim_data is None 체크를 추가하여 AttributeError를 원천적으로 방지했습니다.

# UI 텍스트 명확화: 3D 경로 분석 버튼의 텍스트를 "(정적)"으로 변경하여, 애니메이션인 시뮬레이션 기능과 명확히 구분되도록 했습니다.


# PyQt5 기반 공정 데이터 LLM 분석기 (V2.2.2)
# - 시뮬레이션 데이터 부족 예외 처리 강화
# PyQt5 기반 공정 데이터 LLM 분석기 (V2.3)
# - 드웰 타임을 포함한 전체 로봇 경로 애니메이션 구현
# PyQt5 기반 공정 데이터 LLM 분석기 (V3.0)
# - 통합 분석 대시보드 기능 추가 (3D 시뮬레이션 + 2D 동기화 그래프)

from __future__ import annotations
import sys, traceback, html
from pathlib import Path
import json
import os
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import pyvista as pv

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem, QSlider, QStackedWidget, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor

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
from core.analysis_visualizer_v1 import AnalysisVisualizer

# --- optional: metadata build & indexing scripts (존재할 경우 자동 사용) ---
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

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn, self.a, self.kw = fn, a, kw

    def run(self):
        try:
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

# ---------------- drag & drop ----------------
class DropArea(QFrame):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setStyleSheet("""
        QFrame {
            border: 2px dashed #9ca3af; border-radius: 10px;
            background: #fafafa; color:#374151;
        }
        QFrame[drag='true'] { border-color:#2563eb; background:#eef2ff; }
        """)
        lay = QVBoxLayout(self)
        lab = QLabel("📥 여기에 CSV 파일을 드래그 & 드롭")
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("font-weight:600;")
        lay.addWidget(lab)

    def dragEnterEvent(self, e):
        ok = any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv") for u in e.mimeData().urls())
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
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")]
        if paths:
            self.filesDropped.emit(paths)
        e.acceptProposedAction()

# ---------------- chat bubbles (always pinned to bottom) ----------------
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
        self._bot_style = "QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;} QLabel {color:#0f5132; font-size:13px;}"
        self._container.installEventFilter(self)

    def _scroll_to_bottom_later(self):
        from PyQt5.QtCore import QTimer
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

# ---------------- LLM prompt helpers ----------------
def _tone_style(tone: str) -> str:
    return ("말투는 친근하고 공감 있게, 군더더기 없이 자연스럽게." if tone == "친근" else "말투는 단정하고 간결하게, 불필요한 수식은 피한다.")

def llm_final_only(llm, question: str, df_snip: str, meta_snip: str, tone: str) -> str:
    prompt = (
        "역할: 제조 공정 데이터 분석 파트너.\n"
        f"{_tone_style(tone)}\n"
        "아래 자료를 참고해 질문에 답하되, **최종 답변 한 단락**만 출력하라.\n"
        "금지: '근거', '추가 확인 항목', '결론:' 같은 제목이나 섹션을 쓰지 말 것.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n\n"
        "출력: 최종 답변 한 단락(한국어)."
    )
    return llm.invoke(prompt).content

def llm_checks_only(llm, question: str, df_snip: str, meta_snip: str) -> str:
    prompt = ("역할: 제조 공정 데이터 분석 점검관.\n"
        "다음 자료를 보고, 분석을 더 신뢰할 수 있게 만들 **추가 확인 항목** 3~6개를 제안하라.\n"
        "형식: 하이픈(- ) 불릿 리스트만 출력. 다른 문구/제목/서론 금지.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n"
    )
    return llm.invoke(prompt).content


# ----- #
# ----- #
# ----- #
# ---------------- main window ----------------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def __init__(self):
        super().__init__()
        self.history: List[Tuple[str, str]] = []
        
        # <<<< 핵심 수정: 시뮬레이터 상태 변수 일원화 및 정밀 타이머 >>>>
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setTimerType(Qt.PreciseTimer)
        self.simulation_timer.timeout.connect(self._update_simulation_frame)
        self.sim_handles = None
        self.sim_frame_index = 0
        
        self.setupUi()
        self.init_backend()
        self.load_history()
        self.repopulate_chat()

    def setupUi(self):
        self.setWindowTitle("공정 데이터 LLM 분석기 V2.2.2 (PyVista)")
        self.resize(1700, 900)
        main_layout = QHBoxLayout(self)
        self.tab_widget = QTabWidget(); main_layout.addWidget(self.tab_widget)
        self.llm_tab = QWidget(); self.tab_widget.addTab(self.llm_tab, "LLM 기반 분석")
        self.setup_llm_tab()
        self.viz_tab = QWidget(); self.tab_widget.addTab(self.viz_tab, "데이터 시각화 분석")
        self.setup_viz_tab()
        self.csv_files, self.file_ids = [], {}; self.last_df, self.df_viz = None, None
        self.visualizer, self.viz_context = None, None

    def setup_llm_tab(self):
        layout = QHBoxLayout(self.llm_tab)
        left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        layout.addLayout(left, 2); layout.addLayout(center, 5); layout.addLayout(right, 3)
        left.addWidget(QLabel("📁 소스 파일 (RAG 및 SQL 대상)")); self.drop = DropArea(); self.drop.filesDropped.connect(self.handle_csv_paths); left.addWidget(self.drop)
        self.btn_upload = QPushButton("CSV 업로드"); self.btn_upload.clicked.connect(self.on_upload); left.addWidget(self.btn_upload); left.addWidget(QLabel("저장된 파일"))
        self.file_list = QListWidget(); left.addWidget(self.file_list, 1); self.btn_del = QPushButton("선택 삭제"); self.btn_del.clicked.connect(self.on_delete_files); left.addWidget(self.btn_del)
        center.addWidget(QLabel("💬 LLM 질의")); tone_row = QHBoxLayout(); tone_row.addWidget(QLabel("톤")); self.tone = QComboBox(); self.tone.addItems(["전문", "친근"]); tone_row.addWidget(self.tone)
        tone_row.addStretch(1); center.addLayout(tone_row); self.chat = ChatView(); center.addWidget(self.chat, 1)
        self.btn_clear_history = QPushButton("채팅 로그 초기화"); self.btn_clear_history.clicked.connect(self.on_clear_history); center.addWidget(self.btn_clear_history)
        send_row = QHBoxLayout(); self.inp = QLineEdit(); self.inp.setPlaceholderText("질문을 입력하고 Enter…"); self.inp.returnPressed.connect(self.on_ask)
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask); self.status = QLabel("")
        send_row.addWidget(self.inp, 1); send_row.addWidget(self.btn_send); send_row.addWidget(self.status); center.addLayout(send_row)
        right.addWidget(QLabel("📊 LLM 결과/리포트")); self.tabs = QTabWidget(); right.addWidget(self.tabs, 1)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.tabs.addTab(self.tbl, "표(Table)")
        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig); self.tabs.addTab(self.canvas, "그래프(Chart)")
        self.evidence = QTextEdit(); self.evidence.setReadOnly(True); self.tabs.addTab(self.evidence, "근거(Evidence)")
        self.report = QTextEdit(); self.report.setReadOnly(True); self.tabs.addTab(self.report, "보고서(Report)")

# app/main9_1.py -> MainWindow 클래스 내부

    def setup_viz_tab(self):
        layout = QVBoxLayout(self.viz_tab)
        control_panel = QFrame(); control_panel.setFixedHeight(60); control_layout = QHBoxLayout(control_panel)
        
        # 1. 분석 실행 버튼들
        self.btn_load_csv_viz = QPushButton("CSV 파일 불러오기")
        self.btn_run_stability = QPushButton("1. 안정성 분석"); self.btn_run_stability.setEnabled(False)
        self.btn_run_correlation = QPushButton("2. 상관관계 대시보드"); self.btn_run_correlation.setEnabled(False)
        self.btn_run_3d_path = QPushButton("3. 3D 경로 (정적)"); self.btn_run_3d_path.setEnabled(False)
        self.btn_run_simulation = QPushButton("4. 공정 시뮬레이션"); self.btn_run_simulation.setEnabled(False)
        self.btn_ask_llm_about_viz = QPushButton("🤖 이 분석에 대해 질문하기"); self.btn_ask_llm_about_viz.setEnabled(False)

        control_layout.addWidget(self.btn_load_csv_viz)
        control_layout.addWidget(self.btn_run_stability)
        control_layout.addWidget(self.btn_run_correlation)
        control_layout.addWidget(self.btn_run_3d_path)
        control_layout.addWidget(self.btn_run_simulation)
        control_layout.addStretch(1)

        # 2. 시뮬레이션 제어 위젯 그룹 (기본은 숨김)
        self.sim_controls_widget = QWidget()
        sim_layout = QHBoxLayout(self.sim_controls_widget)
        sim_layout.setContentsMargins(0,0,0,0)
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
        
        control_layout.addWidget(self.sim_controls_widget)
        control_layout.addWidget(self.btn_ask_llm_about_viz)
        self.sim_controls_widget.hide()

        # 3. 뷰 전환을 위한 QStackedWidget 설정
        self.viz_stack = QStackedWidget()
        
        # 3.1. Matplotlib 뷰 (인덱스 0)
        self.mpl_widget = QWidget()
        mpl_layout = QVBoxLayout(self.mpl_widget)
        self.viz_fig = plt.figure(tight_layout=True)
        self.viz_canvas = FigureCanvas(self.viz_fig)
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, self)
        mpl_layout.addWidget(self.viz_toolbar)
        mpl_layout.addWidget(self.viz_canvas)
        self.viz_stack.addWidget(self.mpl_widget)

        # 3.2. PyVista 뷰 (인덱스 1)
        self.pv_plotter = QtInteractor(self.viz_tab, auto_update=False)
        self.viz_stack.addWidget(self.pv_plotter.interactor)
        
        # 초기 안내 메시지
        ax = self.viz_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Please load a CSV file to start analysis.", ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
        
        # 4. 최종 레이아웃에 위젯 추가
        layout.addWidget(control_panel)
        layout.addWidget(self.viz_stack, 1)

        # 5. 모든 버튼의 시그널-슬롯 연결
        self.btn_load_csv_viz.clicked.connect(self.load_csv_for_viz)
        self.btn_run_stability.clicked.connect(self.run_stability_analysis)
        self.btn_run_correlation.clicked.connect(self.run_correlation_dashboard)
        self.btn_run_3d_path.clicked.connect(self.run_3d_path_analysis)
        self.btn_run_simulation.clicked.connect(self.run_process_simulation)
        self.btn_ask_llm_about_viz.clicked.connect(self.ask_llm_about_viz)
        self.btn_toggle_playback.clicked.connect(self.toggle_simulation_playback)
        self.btn_reset_simulation.clicked.connect(self.reset_simulation)

    def _stop_simulation_if_running(self):
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.btn_toggle_playback.setText("▶ 재생")

    def init_backend(self):
        s = self.s = get_settings(); self.engine = make_engine(s.db_url)
        try:
            with self.engine.begin() as c: c.exec_driver_sql("SELECT 1")
        except Exception as e: QMessageBox.critical(self, "DB 연결 실패", str(e))
        self.llm = build_llm(s.openai_model, s.openai_key, 0)
        self.sql_chain = build_sql_chain(self.llm, s.db_url)
        self.emb = build_embeddings(s.openai_key, s.embed_model)
        self.chroma = build_chroma(self.emb, s.vector_db_dir)

    def build_prompt(self, question: str) -> str:
        full_question = question
        if self.viz_context:
            full_question = f"[Current Analysis Context]\n{self.viz_context}\n\n[User's Question]\n{question}"
            self.viz_context = None
        context = "".join(f"이전 Q: {q}\n이전 A: {a}\n" for q, a in self.history[-MAX_HISTORY_TURNS:])
        return context + f"질문: {full_question}"

    def save_history(self):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f: json.dump(self.history[-MAX_HISTORY_RECORDS:], f, ensure_ascii=False, indent=2)
        except Exception as e: print(f"[히스토리 저장 오류] {e}")

    def load_history(self):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f: self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): self.history = []
        except Exception as e: print(f"[히스토리 불러오기 오류] {e}"); self.history = []

    # <<<< ADD THIS FUNCTION BACK IN >>>>
    def repopulate_chat(self):
        """Reloads the chat window with messages from the loaded history."""
        self.chat.clear()
        if not self.history: 
            self.chat.add_bot("안녕하세요! 업로드 후 질문을 입력해 주세요.")
            return
        for q, a in self.history: 
            self.chat.add_user(q)
            self.chat.add_bot(a)
    # <<<< END OF FUNCTION TO ADD >>>>

    def on_clear_history(self):
        if QMessageBox.question(self, "확인", "정말 모든 채팅 로그를 삭제할까요?") == QMessageBox.Yes:
            self.history = []; self.chat.clear(); self.chat.add_bot("채팅 로그가 초기화되었습니다.")
            if os.path.exists(HISTORY_PATH):
                try: os.remove(HISTORY_PATH)
                except Exception as e: print(f"[히스토리 파일 삭제 오류] {e}")

    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy); self.inp.setReadOnly(busy)
        self.status.setText("🤖 답변 생성 중…" if busy else "")

    def on_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
        if files: self.handle_csv_paths(files)

    def handle_csv_paths(self, paths: list[str]):
        prog = QProgressDialog("CSV 처리 중...", "취소", 0, len(paths), self); prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
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
                except Exception as _e: self.chat.add_bot(f"⚠️ 메타/인덱싱 경고: {Path(p).name}\n{_e}")
                self.csv_files.append((Path(p).name, df)); it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
                self.chat.add_bot(f"✅ 업로드 완료: {Path(p).name}\n(table={table})"); ok += 1
            except Exception as e: self.chat.add_bot(f"❌ 업로드 실패: {p}\n{e}"); fail += 1
        prog.setValue(len(paths)); QMessageBox.information(self, "완료", f"성공 {ok} / 실패 {fail}"); self.update_report_summary()

    def on_delete_files(self):
        items = [self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).checkState() == Qt.Checked]
        if not items: return QMessageBox.information(self, "알림", "체크된 파일이 없습니다.")
        if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 파일을 삭제합니다. 계속할까요?") != QMessageBox.Yes: return
        for it in items:
            fname = it.text()
            self.csv_files = [(f, df) for f, df in self.csv_files if f != fname]
            self.file_list.takeItem(self.file_list.row(it))
            table = table_name_from_file(fname)
            try:
                with self.engine.begin() as c: c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
            except Exception as e: self.chat.add_bot(f"⚠️ DB 테이블 삭제 경고: {table} / {e}")
            fid = self.file_ids.pop(fname, None)
            if fid:
                try: self.chroma.delete(ids=[f"{fid}:{i:04d}" for i in range(2000)])
                except Exception as e: self.chat.add_bot(f"⚠️ 임베딩 삭제 경고: {fname} / {e}")
        self.update_report_summary(); self.chat.add_bot("🗑️ 선택 파일 삭제 완료")

    def on_ask(self):
        q = self.inp.text().strip();
        if not q: return
        self.inp.clear(); self.chat.add_user(q)
        tone = self.tone.currentText(); self.set_busy(True)
        def _task():
            sql, df, err_sql = "", None, ""
            prompt_for_llm = self.build_prompt(q)
            try:
                sql = generate_sql_from_nlq(self.sql_chain, prompt_for_llm, engine_or_url=self.engine)
                df = run_sql(self.engine, sql)
                if isinstance(df, pd.DataFrame) and df.empty: df = None
            except Exception as e: err_sql = str(e)
            try: docs = retrieve_meta(self.chroma, prompt_for_llm, 6)
            except Exception: docs = []
            df_snip = df.head(20).to_csv(index=False) if df is not None else ""
            meta_snip = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs[:4])
            final_text = llm_final_only(self.llm, prompt_for_llm, df_snip, meta_snip, tone)
            checks_list = llm_checks_only(self.llm, prompt_for_llm, df_snip, meta_snip)
            ev_lines = ["## 사용 근거"];
            if sql: ev_lines += ["### 사용 SQL", "```sql", sql.strip(), "```"]
            if isinstance(df, pd.DataFrame): ev_lines += ["### SQL 결과 개요", f"- 행 수: {len(df)}", f"- 열 수: {df.shape[1]}"]
            if docs:
                ev_lines.append("### RAG 근거(상위 문서 첫 줄)")
                for i, d in enumerate(docs[:5], 1): ev_lines.append(f"{i}. {getattr(d, 'page_content', str(d)).splitlines()[0][:200]}")
            if checks_list: ev_lines += ["", "## 추가 확인 항목", checks_list]
            if err_sql and not sql: ev_lines += ["", "### SQL 생성/실행 참고", err_sql]
            return (q, final_text, df, sql, "\n".join(ev_lines))

        def _done(res, err):
            self.set_busy(False)
            if err: return QMessageBox.critical(self, "질의 오류", str(err))
            q, final_text, df, sql, evidence_text = res
            self.chat.add_bot(final_text)
            self.history.append((q, final_text)); self.save_history()
            if isinstance(df, pd.DataFrame): self.render_all(df, sql)
            self.evidence.setPlainText(evidence_text)
        run_in_thread(self, _task, _done)

    def update_report_summary(self):
        if not self.csv_files: self.report.setPlainText("업로드된 데이터가 없습니다."); return
        lines = ["# 자동 분석 리포트(데이터 요약)\n"]
        for fname, df in self.csv_files:
            lines += [f"## 파일: {fname}", f"- 행: {len(df)}, 열: {df.shape[1]}"]
            for c in df.select_dtypes(include="number").columns[:10]:
                s = df[c].dropna()
                if not s.empty: lines.append(f"· {c}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
            lines.append("")
        self.report.setPlainText("\n".join(lines))

    def render_all(self, df: pd.DataFrame, sql: str | None):
        view = df.head(self.MAX_ROWS_TABLE)
        plot_df = view.iloc[::max(1, len(view)//self.MAX_POINTS_PLOT)] if len(view) > self.MAX_POINTS_PLOT else view
        df_to_table(self.tbl, view)
        plot_df_line(self.ax, self.canvas, plot_df)
        self.last_df = df
    
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
                ax = self.viz_fig.add_subplot(111); ax.text(0.5, 0.5, f"'{Path(fileName).name}' loaded.\nPlease select an analysis.", ha='center', va='center'); ax.axis('off')
                self.viz_canvas.draw()
                self.btn_run_stability.setEnabled(True); self.btn_run_correlation.setEnabled(True)
                self.btn_run_3d_path.setEnabled(True); self.btn_run_simulation.setEnabled(True)
                self.btn_ask_llm_about_viz.setEnabled(False)
            except Exception as e:
                self.viz_fig.clear(); ax = self.viz_fig.add_subplot(111); ax.text(0.5, 0.5, f"File Load Error:\n{e}", ha='center', wrap=True); ax.axis('off'); self.viz_canvas.draw()

    def _run_visualization(self, plot_function_name: str, context_text: str):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(0)
        self.sim_controls_widget.hide()
        if not self.visualizer: return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            plot_function = getattr(self.visualizer, plot_function_name)
            plot_function(self.viz_fig); self.viz_canvas.draw()
            self.viz_context = context_text; self.btn_ask_llm_about_viz.setEnabled(True)
        except Exception as e:
            self.viz_fig.clear(); ax = self.viz_fig.add_subplot(111); ax.text(0.5, 0.5, f"Analysis Error:\n{e}", ha='center', wrap=True); ax.axis('off'); self.viz_canvas.draw()
            print(traceback.format_exc())

    def run_integrated_dashboard(self):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(1)
        self.sim_controls_widget.show()
        if not self.visualizer: return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        try:
            self.sim_handles = self.visualizer.prepare_integrated_dashboard(self.pv_plotter, self.viz_fig_sync)
            
            if self.sim_handles is None:
                QMessageBox.information(self, "Info", "No data for simulation.")
                self.viz_stack.setCurrentIndex(0); self.sim_controls_widget.hide()
                return

            self.viz_canvas_sync.draw()
            self.reset_simulation()
            self.viz_context = "integrated analysis dashboard with 3D simulation and synchronized 2D graphs."
            self.btn_ask_llm_about_viz.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Dashboard Error", str(e)); print(traceback.format_exc())


    def run_stability_analysis(self): self._run_visualization('plot_stability', "stability analysis graph.")
    def run_correlation_dashboard(self): self._run_visualization('plot_correlation_dashboard', "correlation analysis dashboard.")
    def run_3d_path_analysis(self): self._run_visualization('plot_3d_path', "static 3D process path visualization.")
    
        
    def run_process_simulation(self):
        self._stop_simulation_if_running()
        self.viz_stack.setCurrentIndex(1)
        self.sim_controls_widget.show()
        if not self.visualizer: return QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
        
        try:
            self.sim_handles = self.visualizer.prepare_pyvista_simulation(self.pv_plotter)
            
            if self.sim_handles is None:
                QMessageBox.information(self, "시뮬레이션 불가", "시뮬레이션을 생성하기에 유효한 데이터(LASER_ON 상태 등)가 부족합니다.")
                self.viz_stack.setCurrentIndex(0)
                self.sim_controls_widget.hide()
                return

            self.reset_simulation()
            self.viz_context = "time-synchronized 3D animation of the process."
            self.btn_ask_llm_about_viz.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e)); print(traceback.format_exc())
            self.viz_stack.setCurrentIndex(0)
            self.sim_controls_widget.hide()

    def toggle_simulation_playback(self):
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.btn_toggle_playback.setText("▶ 재생")
        else:
            # 재생 전, 프레임이 끝까지 갔으면 처음부터
            if self.sim_handles is not None and self.sim_frame_index >= len(self.sim_handles["sim_df"]):
                self.reset_simulation()
            # 실제 간격은 _update_simulation_frame에서 다음 간격으로 동기화
            self.simulation_timer.start(0)
            self.btn_toggle_playback.setText("⏸ 일시정지")

    def reset_simulation(self):
        self._stop_simulation_if_running()
        self.sim_frame_index = 0
        if self.sim_handles:
            # 전체 숨김(알파 0)
            base = self.sim_handles["base_rgba"]
            cur = base.copy()
            cur[:, 3] = 0
            self.sim_handles["rgba_current"] = cur
            self.sim_handles["path_poly"]["RGBA"] = cur
            # 헤드 위치를 시작점으로
            self.sim_handles["head_actor"].SetPosition(self.sim_handles["points"][0])
            self.pv_plotter.render()

    
    def _update_simulation_frame(self):
        if (not self.sim_handles) or (self.sim_frame_index >= len(self.sim_handles["sim_df"])):
            self._stop_simulation_if_running(); return

        i = self.sim_frame_index
        sim_df = self.sim_handles["sim_df"]

        # 1) 누적(쌓이기): alpha만 앞부분 켜기
        cur = self.sim_handles["rgba_current"]
        base = self.sim_handles["base_rgba"]
        end = min(i + 1, len(cur))
        if end > 0:
            cur[:end, :] = base[:end, :]  # 앞부분을 최종 색으로 (alpha 포함)
            self.sim_handles["path_poly"]["RGBA"] = cur

        # 2) 헤드 이동
        self.sim_handles["head_actor"].SetPosition(self.sim_handles["points"][i])

        # 3) 렌더
        self.pv_plotter.render()

        # 4) 다음 프레임 간격(최소 15ms)
        if i + 1 < len(sim_df):
            interval_ms = float(sim_df['time_delta_ms'].iloc[i + 1])
            playback_speed = max(0.1, self.speed_slider.value() / 10.0)
            adjusted = max(15, int(interval_ms / playback_speed))
            self.simulation_timer.setInterval(adjusted)
        else:
            self._stop_simulation_if_running()

        self.sim_frame_index += 1


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
