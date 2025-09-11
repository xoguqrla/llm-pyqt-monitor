# # app/main.py
# # 최종 업데이트 : 2025-08-05
# # 한국어 고정 + 말풍선 리치텍스트 렌더링(간단 Markdown→HTML)

# from __future__ import annotations
# import sys, traceback, re, html
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import pandas as pd
# from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
#     QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
#     QTextEdit, QTabWidget, QHeaderView, QMessageBox, QFrame,
#     QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem
# )
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt

# from core.config import get_settings
# from core.csv_ops import load_and_meta
# from core.db_ops import make_engine, ingest_df, ensure_indexes, table_name_from_file
# from core.rag_ops import build_embeddings, build_chroma, build_embedding_texts_from_meta, upsert_texts
# from core.llm_ops import build_llm
# from core.plotting import df_to_table, plot_df_line
# from core.files_registry import upsert_entry

# from langchain.memory import ConversationBufferMemory
# from core.agent import build_agent

# try:
#     from scripts.build_metadata import build_for_table as _build_meta_for_table
# except Exception:
#     _build_meta_for_table = None
# try:
#     from scripts.index_metadata import index_for_sessions as _index_sessions
# except Exception:
#     _index_sessions = None


# # ---------- 예외 훅 ----------
# def global_excepthook(et, ev, tb):
#     msg = "".join(traceback.format_exception(et, ev, tb))[-4000:]
#     print(msg, file=sys.stderr)
#     try:
#         QMessageBox.critical(None, "Unhandled Error", msg)
#     except Exception:
#         pass
# sys.excepthook = global_excepthook


# # ---------- 워커 ----------
# class Worker(QObject):
#     finished = pyqtSignal(object, object)
#     def __init__(self, fn, *a, **kw):
#         super().__init__(); self.fn, self.a, self.kw = fn, a, kw
#     def run(self):
#         try:
#             self.finished.emit(self.fn(*self.a, **self.kw), None)
#         except Exception as e:
#             self.finished.emit(None, e)

# def run_in_thread(parent, fn, cb, *a, **kw):
#     th = QThread(parent); wk = Worker(fn, *a, **kw); wk.moveToThread(th)
#     def _done(r, e):
#         try: cb(r, e)
#         finally:
#             th.quit(); wk.deleteLater(); th.deleteLater()
#     wk.finished.connect(_done); th.started.connect(wk.run); th.start()


# # ---------- 간단 Markdown → HTML ----------
# def md_to_html_basic(text: str) -> str:
#     """
#     - ```lang ... ``` 코드펜스 → <pre><code>
#     - 선두 '- ' 목록 → 불릿
#     - 나머지 줄바꿈 유지
#     - 마크다운 표/긴 텍스트는 <pre>로 감싸 가독성 유지
#     """
#     if not text:
#         return ""

#     code_blocks: List[str] = []

#     # 1) 코드펜스 보존
#     def _save_code(m):
#         code = m.group(2)
#         code_html = (
#             '<pre style="background:#111827;color:#e5e7eb;'
#             'padding:10px;border-radius:8px;white-space:pre-wrap;">'
#             f'{html.escape(code)}</pre>'
#         )
#         code_blocks.append(code_html)
#         return f"@@CODEBLOCK{len(code_blocks)-1}@@"

#     s = re.sub(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", lambda m: _save_code(m), text)

#     # 2) 일반 텍스트 escape
#     s = html.escape(s)

#     # 3) 목록/줄바꿈 처리
#     s = re.sub(r"(?:^|[\n])-\s+([^\n]+)", r"\n• \1", s)
#     s = s.replace("\n", "<br>")

#     # 4) 코드펜스 복원
#     for i, block in enumerate(code_blocks):
#         s = s.replace(f"@@CODEBLOCK{i}@@", block)

#     # 5) Markdown 표(파이프 포함 다중행)는 <pre>로 감싸기(가독)
#     if "|" in text and "\n" in text:
#         # 아주 단순히, 파이프가 두 번 이상 등장하면 표가 있는 걸로 간주
#         pipe_cnt = text.count("|")
#         if pipe_cnt >= 2:
#             s = f'<pre style="background:#f8fafc;padding:10px;border-radius:8px;white-space:pre-wrap;">{s}</pre>'

#     return s


# # ---------- 드롭 영역 ----------
# class DropArea(QFrame):
#     filesDropped = pyqtSignal(list)
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAcceptDrops(True); self.setMinimumHeight(140)
#         self.setStyleSheet("""
#         QFrame { border:2px dashed #9ca3af; border-radius:10px; background:#fafafa; }
#         QFrame[drag='true'] { border-color:#2563eb; background:#eef2ff; }
#         """)
#         lay = QVBoxLayout(self)
#         lab = QLabel("📥 CSV 파일을 드래그 & 드롭")
#         lab.setAlignment(Qt.AlignCenter); lab.setStyleSheet("font-weight:600;")
#         lay.addWidget(lab)
#     def dragEnterEvent(self, e):
#         ok = any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv") for u in e.mimeData().urls())
#         if ok:
#             self.setProperty("drag", True); self.style().unpolish(self); self.style().polish(self)
#             e.acceptProposedAction()
#         else:
#             e.ignore()
#     def dragLeaveEvent(self, e):
#         self.setProperty("drag", False); self.style().unpolish(self); self.style().polish(self)
#         super().dragLeaveEvent(e)
#     def dropEvent(self, e):
#         self.setProperty("drag", False); self.style().unpolish(self); self.style().polish(self)
#         paths = [u.toLocalFile() for u in e.mimeData().urls()
#                  if u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")]
#         if paths: self.filesDropped.emit(paths)
#         e.acceptProposedAction()


# # ---------- 채팅 뷰 ----------
# class ChatView(QScrollArea):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWidgetResizable(True)
#         container = QWidget(); self.setWidget(container)
#         self.vbox = QVBoxLayout(container); self.vbox.setSpacing(8); self.vbox.setContentsMargins(8,8,8,8)
#         self.vbox.addItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))
#         self._user_css = "QFrame {background:#f3f4f6; border-radius:12px; padding:8px 10px;}"
#         self._bot_css  = "QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;}"

#     def _add(self, html_text: str, css: str, right: bool):
#         fr = QFrame(); fr.setStyleSheet(css)
#         lab = QLabel(); lab.setTextFormat(Qt.RichText); lab.setWordWrap(True)
#         lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
#         lab.setText(html_text)
#         hl = QHBoxLayout(fr); hl.setContentsMargins(10,6,10,6); hl.addWidget(lab)
#         row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
#         if right: rl.addStretch(); rl.addWidget(fr)
#         else: rl.addWidget(fr); rl.addStretch()
#         self.vbox.addWidget(row)
#         self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

#     def add_user(self, text: str):
#         self._add(md_to_html_basic(text), self._user_css, False)

#     def add_bot(self, text: str):
#         self._add(md_to_html_basic(text), self._bot_css, True)


# # ---------- 메인 윈도우 ----------
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("공정 데이터 LLM 분석 V1.7 Ultimate")
#         self.resize(1700, 900); self.setAcceptDrops(True)

#         self.s = get_settings()
#         self.engine = make_engine(self.s.db_url)
#         # 모델은 유지하되, 에이전트 프롬프트에서 한국어/형식 강제
#         self.llm = build_llm(self.s.openai_model, self.s.openai_key, 0.0)
#         self.emb = build_embeddings(self.s.openai_key, self.s.embed_model)
#         self.chroma = build_chroma(self.emb, self.s.vector_db_dir)

#         self.trace_logs: List[str] = []
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self.agent  = build_agent(self.llm, self.engine, self.chroma, self.memory, logs=self.trace_logs)

#         self.csv_files: List[Tuple[str, pd.DataFrame]] = []
#         self.file_ids: Dict[str, str] = {}

#         left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
#         main = QHBoxLayout(self); main.addLayout(left,2); main.addLayout(center,5); main.addLayout(right,3)

#         # left
#         left.addWidget(QLabel("📁 Data Files"))
#         drop = DropArea(); drop.filesDropped.connect(self.handle_csv_paths); left.addWidget(drop)
#         btn_up = QPushButton("Upload CSV"); btn_up.clicked.connect(self.on_upload); left.addWidget(btn_up)
#         left.addWidget(QLabel("Loaded Files"))
#         self.file_list = QListWidget(); left.addWidget(self.file_list,1)
#         btn_del = QPushButton("Delete Selected"); btn_del.clicked.connect(self.on_delete_files); left.addWidget(btn_del)

#         # center
#         center.addWidget(QLabel("💬 Ask Agent"))
#         self.chat = ChatView(); center.addWidget(self.chat,1)
#         row = QHBoxLayout()
#         self.input = QLineEdit(); self.input.setPlaceholderText("예: 001_1_data의 time과 mpt 산점도 그려줘")
#         self.input.returnPressed.connect(self.on_ask); row.addWidget(self.input,1)
#         send = QPushButton("▶"); send.clicked.connect(self.on_ask); row.addWidget(send)
#         center.addLayout(row)

#         # right
#         right.addWidget(QLabel("📊 Results"))
#         self.tabs = QTabWidget(); right.addWidget(self.tabs,1)
#         self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
#         self.tabs.addTab(self.tbl, "Table")
#         self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
#         self.tabs.addTab(self.canvas, "Chart")
#         self.evidence = QTextEdit(); self.evidence.setReadOnly(True); self.tabs.addTab(self.evidence, "Evidence")
#         self.report   = QTextEdit(); self.report.setReadOnly(True); self.tabs.addTab(self.report, "Report")

#         self.chat.add_bot("안녕하세요! CSV 업로드 후 질문해 주세요. 표/코드는 깔끔하게 보여 드릴게요 🙂")

#     # ---- 업로드 ----
#     def on_upload(self):
#         paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV", str(self.s.uploads_dir), "CSV (*.csv)")
#         if paths: self.handle_csv_paths(paths)

#     def handle_csv_paths(self, paths: List[str]):
#         ok = fail = 0
#         prog = QProgressDialog("Loading...", "Cancel", 0, len(paths), self)
#         prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
#         for i, p in enumerate(paths, 1):
#             prog.setValue(i-1); QApplication.processEvents()
#             if prog.wasCanceled(): break
#             try:
#                 df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)

#                 entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
#                 upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
#                 self.file_ids[Path(p).name] = entry.file_id

#                 table = table_name_from_file(Path(p).name)
#                 ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)

#                 try:
#                     if _build_meta_for_table is not None:
#                         sessions = _build_meta_for_table(self.s.db_url, table)
#                         if _index_sessions is not None:
#                             _index_sessions(self.s.db_url, str(self.s.vector_db_dir), sessions)
#                 except Exception as sube:
#                     self.chat.add_bot(f"⚠️ 메타/인덱싱 경고({Path(p).name}): {sube}")

#                 self.csv_files.append((Path(p).name, df))
#                 it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
#                 self.chat.add_bot(f"✅ Loaded: {Path(p).name} (table={table})")
#                 ok += 1
#             except Exception as e:
#                 self.chat.add_bot(f"❌ Load failed: {p}\n{e}"); fail += 1
#         prog.setValue(len(paths))
#         QMessageBox.information(self, "Done", f"Success {ok} / Fail {fail}")
#         self._update_report()

#     # ---- 삭제 ----
#     def on_delete_files(self):
#         items = [self.file_list.item(i) for i in range(self.file_list.count())
#                  if self.file_list.item(i).checkState() == Qt.Checked]
#         if not items:
#             QMessageBox.information(self, "Notice", "No files selected"); return
#         if QMessageBox.question(self, "Confirm", f"Delete {len(items)} files?", QMessageBox.Yes) != QMessageBox.Yes:
#             return
#         for it in items:
#             name = it.text()
#             self.csv_files = [(f, df) for f, df in self.csv_files if f != name]
#             self.file_list.takeItem(self.file_list.row(it))
#         QMessageBox.information(self, "Deleted", "Selected files removed")
#         self._update_report()

#     # ---- 질의 ----
#     def on_ask(self):
#         q = self.input.text().strip()
#         if not q: return
#         self.input.clear()
#         self.chat.add_user(q)
#         self.setEnabled(False)
#         self.trace_logs.clear()

#         def task():
#             try:
#                 return self.agent.run(q)
#             except Exception as e:
#                 return {"_error": str(e)}

#         def done(res, err):
#             self.setEnabled(True)
#             if err:
#                 QMessageBox.critical(self, "Error", str(err)); return

#             # Evidence 로그
#             self.evidence.setPlainText("\n".join(self.trace_logs))

#             # DataFrame
#             if isinstance(res, pd.DataFrame):
#                 self._render_table_chart(res); return

#             # dict (산점도/오류 등)
#             if isinstance(res, dict):
#                 if "_error" in res:
#                     self.chat.add_bot(f"⚠️ 처리 오류: {res['_error']}"); return
#                 if res.get("_plot") == "scatter" and all(k in res for k in ("x","y","data")):
#                     self._render_scatter(res); return
#                 if all(k in res for k in ("x","y","data")):
#                     payload = {"_plot": "scatter", **res}
#                     self._render_scatter(payload); return
#                 self.chat.add_bot(str(res)); return

#             # 문자열
#             self.chat.add_bot(str(res))

#         run_in_thread(self, task, done)

#     # ---- 렌더링 ----
#     def _render_table_chart(self, df: pd.DataFrame):
#         df_to_table(self.tbl, df)
#         plot_df_line(self.ax, self.canvas, df)
#         self.tabs.setCurrentWidget(self.tbl)

#     def _render_scatter(self, payload: Dict[str, Any]):
#         self.ax.clear()
#         xs = payload["data"]["x"]; ys = payload["data"]["y"]
#         self.ax.scatter(xs, ys, s=10)
#         self.ax.set_xlabel(payload["x"]); self.ax.set_ylabel(payload["y"])
#         self.ax.set_title("Scatter Plot")
#         self.canvas.draw()
#         self.tabs.setCurrentWidget(self.canvas)

#     def _update_report(self):
#         if not self.csv_files:
#             self.report.setPlainText("업로드된 데이터가 없습니다. 좌측에서 CSV를 추가하세요."); return
#         lines = ["# 자동 분석 리포트(데이터 요약)\n"]
#         for fname, df in self.csv_files:
#             lines += [f"## 파일: {fname}", f"- 행: {len(df)}, 열: {df.shape[1]}"]
#             for c in df.select_dtypes(include="number").columns[:10]:
#                 s = df[c].dropna()
#                 if s.empty: continue
#                 lines.append(f"· {c}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
#             lines.append("")
#         self.report.setPlainText("\n".join(lines))


# # ---------- 엔트리 ----------
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     w = MainWindow(); w.show()
#     sys.exit(app.exec_())


# app/main.py
# 한국어 출력 + .invoke() 기반, Evidence 로그 및 차트/표 렌더 안정화

from __future__ import annotations
import sys, traceback, re, html
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from core.config import get_settings
from core.csv_ops import load_and_meta
from core.db_ops import make_engine, ingest_df, ensure_indexes, table_name_from_file
from core.rag_ops import build_embeddings, build_chroma, build_embedding_texts_from_meta, upsert_texts
from core.llm_ops import build_llm
from core.plotting import df_to_table, plot_df_line
from core.files_registry import upsert_entry

from langchain.memory import ConversationBufferMemory
from core2.agent import build_agent

try:
    from scripts.build_metadata import build_for_table as _build_meta_for_table
except Exception:
    _build_meta_for_table = None
try:
    from scripts.index_metadata import index_for_sessions as _index_sessions
except Exception:
    _index_sessions = None


def global_excepthook(et, ev, tb):
    msg = "".join(traceback.format_exception(et, ev, tb))[-4000:]
    print(msg, file=sys.stderr)
    try:
        QMessageBox.critical(None, "Unhandled Error", msg)
    except Exception:
        pass
sys.excepthook = global_excepthook


class Worker(QObject):
    finished = pyqtSignal(object, object)
    def __init__(self, fn, *a, **kw):
        super().__init__(); self.fn, self.a, self.kw = fn, a, kw
    def run(self):
        try: self.finished.emit(self.fn(*self.a, **self.kw), None)
        except Exception as e: self.finished.emit(None, e)

def run_in_thread(parent, fn, cb, *a, **kw):
    th = QThread(parent); wk = Worker(fn, *a, **kw); wk.moveToThread(th)
    def _done(r, e):
        try: cb(r, e)
        finally: th.quit(); wk.deleteLater(); th.deleteLater()
    wk.finished.connect(_done); th.started.connect(wk.run); th.start()


def md_to_html_basic(text: str) -> str:
    if not text: return ""
    code_blocks: List[str] = []
    def _save_code(m):
        code = m.group(1)
        code_html = ('<pre style="background:#111827;color:#e5e7eb;'
                     'padding:10px;border-radius:8px;white-space:pre-wrap;">'
                     f'{html.escape(code)}</pre>')
        code_blocks.append(code_html)
        return f"@@CODE{len(code_blocks)-1}@@"
    s = re.sub(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", lambda m: _save_code(m), text)
    s = html.escape(s)
    s = re.sub(r"(?:^|[\n])-\s+([^\n]+)", r"\n• \1", s)
    s = s.replace("\n", "<br>")
    for i, block in enumerate(code_blocks):
        s = s.replace(f"@@CODE{i}@@", block)
    if "|" in text and "\n" in text and text.count("|") >= 2:
        s = f'<pre style="background:#f8fafc;padding:10px;border-radius:8px;white-space:pre-wrap;">{s}</pre>'
    return s


class DropArea(QFrame):
    filesDropped = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True); self.setMinimumHeight(140)
        self.setStyleSheet("""
        QFrame { border:2px dashed #9ca3af; border-radius:10px; background:#fafafa; }
        QFrame[drag='true'] { border-color:#2563eb; background:#eef2ff; }
        """)
        lay = QVBoxLayout(self)
        lab = QLabel("📥 CSV 파일을 드래그 & 드롭"); lab.setAlignment(Qt.AlignCenter); lab.setStyleSheet("font-weight:600;")
        lay.addWidget(lab)
    def dragEnterEvent(self, e):
        ok = any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv") for u in e.mimeData().urls())
        if ok:
            self.setProperty("drag", True); self.style().unpolish(self); self.style().polish(self)
            e.acceptProposedAction()
        else:
            e.ignore()
    def dragLeaveEvent(self, e):
        self.setProperty("drag", False); self.style().unpolish(self); self.style().polish(self)
        super().dragLeaveEvent(e)
    def dropEvent(self, e):
        self.setProperty("drag", False); self.style().unpolish(self); self.style().polish(self)
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")]
        if paths: self.filesDropped.emit(paths)
        e.acceptProposedAction()


class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        container = QWidget(); self.setWidget(container)
        self.vbox = QVBoxLayout(container); self.vbox.setSpacing(8); self.vbox.setContentsMargins(8,8,8,8)
        self.vbox.addItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))
        self._user_css = "QFrame {background:#f3f4f6; border-radius:12px; padding:8px 10px;}"
        self._bot_css  = "QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;}"
    def _add(self, html_text: str, css: str, right: bool):
        fr = QFrame(); fr.setStyleSheet(css)
        lab = QLabel(); lab.setTextFormat(Qt.RichText); lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse); lab.setText(html_text)
        hl = QHBoxLayout(fr); hl.setContentsMargins(10,6,10,6); hl.addWidget(lab)
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
        if right: rl.addStretch(); rl.addWidget(fr)
        else: rl.addWidget(fr); rl.addStretch()
        self.vbox.addWidget(row)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    def add_user(self, text: str): self._add(md_to_html_basic(text), self._user_css, False)
    def add_bot (self, text: str): self._add(md_to_html_basic(text), self._bot_css, True)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("공정 데이터 LLM 분석 V1.7 Ultimate")
        self.resize(1700, 900); self.setAcceptDrops(True)

        self.s = get_settings()
        self.engine = make_engine(self.s.db_url)
        self.llm    = build_llm(self.s.openai_model, self.s.openai_key, 0.0)
        self.emb    = build_embeddings(self.s.openai_key, self.s.embed_model)
        self.chroma = build_chroma(self.emb, self.s.vector_db_dir)

        self.trace_logs: List[str] = []
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent  = build_agent(self.llm, self.engine, self.chroma, self.memory, logs=self.trace_logs)

        self.csv_files: List[Tuple[str, pd.DataFrame]] = []
        self.file_ids: Dict[str, str] = {}

        left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        main = QHBoxLayout(self); main.addLayout(left,2); main.addLayout(center,5); main.addLayout(right,3)

        left.addWidget(QLabel("📁 Data Files"))
        drop = DropArea(); drop.filesDropped.connect(self.handle_csv_paths); left.addWidget(drop)
        btn_up = QPushButton("Upload CSV"); btn_up.clicked.connect(self.on_upload); left.addWidget(btn_up)
        left.addWidget(QLabel("Loaded Files"))
        self.file_list = QListWidget(); left.addWidget(self.file_list,1)
        btn_del = QPushButton("Delete Selected"); btn_del.clicked.connect(self.on_delete_files); left.addWidget(btn_del)

        center.addWidget(QLabel("💬 Ask Agent"))
        self.chat = ChatView(); center.addWidget(self.chat,1)
        row = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("예: 001_1_data에서 time~mpt 산점도")
        self.input.returnPressed.connect(self.on_ask); row.addWidget(self.input,1)
        send = QPushButton("▶"); send.clicked.connect(self.on_ask); row.addWidget(send)
        center.addLayout(row)

        right.addWidget(QLabel("📊 Results"))
        self.tabs = QTabWidget(); right.addWidget(self.tabs,1)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl,"Table")
        self.fig,self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas,"Chart")
        self.evidence = QTextEdit(); self.evidence.setReadOnly(True); self.tabs.addTab(self.evidence,"Evidence")
        self.report   = QTextEdit(); self.report.setReadOnly(True); self.tabs.addTab(self.report,"Report")

        self.chat.add_bot("안녕하세요! CSV 업로드 후 질문해 주세요. 결과는 한국어/표/코드로 깔끔히 보여드릴게요.")

    # ---- 업로드 ----
    def on_upload(self):
        paths,_ = QFileDialog.getOpenFileNames(self,"Select CSV",str(self.s.uploads_dir),"CSV (*.csv)")
        if paths: self.handle_csv_paths(paths)

    def handle_csv_paths(self, paths: List[str]):
        ok=fail=0
        prog=QProgressDialog("Loading...","Cancel",0,len(paths),self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        for i,p in enumerate(paths,1):
            prog.setValue(i-1); QApplication.processEvents()
            if prog.wasCanceled(): break
            try:
                df,meta,_ = load_and_meta(Path(p), self.s.meta_json_dir)
                entry = upsert_entry(Path(p), meta["rows"], meta["cols"], "indexed")
                upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                self.file_ids[Path(p).name] = entry.file_id
                table = table_name_from_file(Path(p).name)
                ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)

                try:
                    if _build_meta_for_table is not None:
                        sessions = _build_meta_for_table(self.s.db_url, table)
                        if _index_sessions is not None:
                            _index_sessions(self.s.db_url, str(self.s.vector_db_dir), sessions)
                except Exception as warn:
                    self.chat.add_bot(f"⚠️ 메타/색인 경고({Path(p).name}): {warn}")

                self.csv_files.append((Path(p).name, df))
                it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
                self.chat.add_bot(f"✅ Loaded: {Path(p).name} (table={table})")
                ok+=1
            except Exception as e:
                self.chat.add_bot(f"❌ Load failed: {p}\n{e}"); fail+=1
        prog.setValue(len(paths))
        QMessageBox.information(self,"Done",f"Success {ok} / Fail {fail}")
        self._update_report()

    # ---- 삭제 ----
    def on_delete_files(self):
        items=[self.file_list.item(i) for i in range(self.file_list.count())
               if self.file_list.item(i).checkState()==Qt.Checked]
        if not items:
            QMessageBox.information(self,"Notice","No files selected"); return
        if QMessageBox.question(self,"Confirm",f"Delete {len(items)} files?",QMessageBox.Yes)!=QMessageBox.Yes:
            return
        for it in items:
            name=it.text()
            self.csv_files = [(f,df) for f,df in self.csv_files if f!=name]
            self.file_list.takeItem(self.file_list.row(it))
        QMessageBox.information(self,"Deleted","Selected files removed")
        self._update_report()

    # ---- 질의 ----
    def on_ask(self):
        q = self.input.text().strip()
        print("[DEBUG][on_ask] 질문:", q)
        if not q:
            print("[DEBUG][on_ask] 질문 없음")
            return
        self.input.clear()
        self.chat.add_user(q)
        self.setEnabled(False)
        self.trace_logs.clear()

        def task():
            print("[DEBUG][task] agent.run 진입")
            try:
                result = self.agent.run(q)
                print("[DEBUG][task] agent.run 반환:", result)
                return result
            except Exception as e:
                print("[DEBUG][task] agent.run 예외:", e)
                return {"error": str(e)}

        def done(res, err):
            print("[DEBUG][done] 콜백 진입 res:", res, "err:", err)
            self.setEnabled(True)
            if err:
                self.chat.add_bot(f"⚠️ 오류: {err}")
                print("[DEBUG][done] err:", err)
                return

            if "trace_logs" in res:
                self.evidence.setPlainText("\n".join(res["trace_logs"]))
            if res.get("error"):
                self.chat.add_bot(f"⚠️ 처리 오류: {res['error']}")
                print("[DEBUG][done] res.error:", res['error'])
                return
            if res.get("df") is not None:
                self._render_table_chart(res["df"])
                print("[DEBUG][done] 표 출력 완료")
                return
            if res.get("rag"):
                self.chat.add_bot(str(res["rag"]))
                print("[DEBUG][done] RAG 출력 완료")
                return
            if res.get("hyb"):
                df, sql, final = res["hyb"]
                self._render_table_chart(df)
                self.chat.add_bot(final)
                print("[DEBUG][done] HYB 출력 완료")
                return
            if res.get("chat"):
                self.chat.add_bot(res["chat"])
                print("[DEBUG][done] CHAT 출력 완료")
                return
            self.chat.add_bot("알 수 없는 결과 형식입니다.")
            print("[DEBUG][done] 결과 미해석")

        # run_in_thread가 아니라면
        run_in_thread(self, task, done)



    # ---- 렌더링 ----
    def _render_table_chart(self, df: pd.DataFrame):
        df_to_table(self.tbl, df)
        plot_df_line(self.ax, self.canvas, df)
        self.tabs.setCurrentWidget(self.tbl)

    def _render_scatter(self, payload: Dict[str, Any]):
        self.ax.clear()
        xs = payload["data"]["x"]; ys = payload["data"]["y"]
        self.ax.scatter(xs, ys, s=10)
        self.ax.set_xlabel(payload["x"]); self.ax.set_ylabel(payload["y"])
        self.ax.set_title("Scatter Plot")
        self.canvas.draw()
        self.tabs.setCurrentWidget(self.canvas)

    def _update_report(self):
        if not self.csv_files:
            self.report.setPlainText("업로드된 데이터가 없습니다. 좌측에서 CSV를 추가하세요."); return
        lines = ["# 자동 분석 리포트(데이터 요약)\n"]
        for fname, df in self.csv_files:
            lines += [f"## 파일: {fname}", f"- 행: {len(df)}, 열: {df.shape[1]}"]
            for c in df.select_dtypes(include="number").columns[:10]:
                s = df[c].dropna()
                if s.empty: continue
                lines.append(f"· {c}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
            lines.append("")
        self.report.setPlainText("\n".join(lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
