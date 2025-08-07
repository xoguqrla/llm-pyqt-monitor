
# app/main6.py
# 최종 업데이트 : 2025-08-05
# PyQt5 기반 공정 데이터 LLM 분석기 (V1.7 Ultimate)
# - CSV 업로드, 대화형 Agent 질의, SQL·RAG·분석툴 자동 라우팅
# - Conversation Memory를 통한 연속질문 지원
# - scatter plot, stats summary, correlation 등 고급 시각화/분석 자동화

from __future__ import annotations
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# --- core modules ---
from core.config import get_settings
from core.csv_ops import load_and_meta
from core.db_ops import make_engine, ingest_df, ensure_indexes, run_sql, table_name_from_file
from core.rag_ops import build_embeddings, build_chroma, build_embedding_texts_from_meta, upsert_texts
from core.llm_ops import build_llm
from core.plotting import df_to_table, plot_df_line
from core.files_registry import upsert_entry

# --- Analysis & Agent ---
from langchain.memory import ConversationBufferMemory
from core.agent import build_agent

# Optional metadata scripts
try:
    from scripts.build_metadata import build_for_table as _build_meta_for_table
except ImportError:
    _build_meta_for_table = None
try:
    from scripts.index_metadata import index_for_sessions as _index_sessions
except ImportError:
    _index_sessions = None

# ---------------- global excepthook ----------------
def global_excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-4000:]
    print(msg, file=sys.stderr)
    QMessageBox.critical(None, "Unhandled Error", msg)
sys.excepthook = global_excepthook

# ---------------- threading helper ----------------
class Worker(QObject):
    finished = pyqtSignal(object, object)
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn; self.args = args; self.kwargs = kwargs
    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.finished.emit(res, None)
        except Exception as e:
            self.finished.emit(None, e)

def run_in_thread(parent, fn, callback, *args, **kwargs):
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    worker.finished.connect(lambda res, err: (callback(res, err), thread.quit(), worker.deleteLater(), thread.deleteLater()))
    thread.started.connect(worker.run)
    thread.start()

# ---------------- drag & drop ----------------
class DropArea(QFrame):
    filesDropped = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True); self.setMinimumHeight(140)
        self.setStyleSheet(
            "QFrame{border:2px dashed #9ca3af; border-radius:10px; background:#fafafa;}"
            "QFrame[drag='true']{border-color:#2563eb; background:#eef2ff;}"
        )
        lay = QVBoxLayout(self)
        lab = QLabel("📥 CSV 파일을 드래그 & 드롭")
        lab.setAlignment(Qt.AlignCenter); lab.setStyleSheet("font-weight:600;")
        lay.addWidget(lab)
    def dragEnterEvent(self, e):
        if any(u.isLocalFile() and u.toLocalFile().lower().endswith('.csv') for u in e.mimeData().urls()):
            self.setProperty('drag', True); self.style().unpolish(self); self.style().polish(self)
            e.acceptProposedAction()
        else: e.ignore()
    def dragLeaveEvent(self, e):
        self.setProperty('drag', False); self.style().unpolish(self); self.style().polish(self); super().dragLeaveEvent(e)
    def dropEvent(self, e):
        self.setProperty('drag', False); self.style().unpolish(self); self.style().polish(self)
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().lower().endswith('.csv')]
        if paths: self.filesDropped.emit(paths)
        e.acceptProposedAction()

# ---------------- chat bubbles ----------------
class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWidgetResizable(True)
        container = QWidget(); self.setWidget(container)
        self.vbox = QVBoxLayout(container); self.vbox.setSpacing(8); self.vbox.setContentsMargins(8,8,8,8)
        self.vbox.addItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))
        self.user_style = "QFrame{background:#f3f4f6;} QLabel{font-size:13px;}"
        self.bot_style  = "QFrame{background:#e8f5e9;} QLabel{font-size:13px;}"
    def add_user(self, text: str): self._add(text, self.user_style, 'left')
    def add_bot(self, text: str): self._add(text, self.bot_style, 'right')
    def _add(self, text, style, align):
        frame = QFrame(); frame.setStyleSheet(style)
        lab = QLabel(text); lab.setWordWrap(True)
        layout = QHBoxLayout(frame)
        if align=='right': layout.addStretch()
        layout.addWidget(lab)
        if align=='left': layout.addStretch()
        self.vbox.addWidget(frame); self._scroll_to_bottom()
    def _scroll_to_bottom(self): sb = self.verticalScrollBar(); sb.setValue(sb.maximum())

# ---------------- main window ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("공정 데이터 LLM 분석 V1.7 Ultimate")
        self.resize(1700,900); self.setAcceptDrops(True)
        # services
        s = get_settings(); self.engine = make_engine(s.db_url)
        self.llm = build_llm(s.openai_model, s.openai_key, 0)
        self.emb   = build_embeddings(s.openai_key, s.embed_model)
        self.chroma= build_chroma(self.emb, s.vector_db_dir)
        self.memory= ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.agent = build_agent(self.llm, self.engine, self.chroma, self.memory)
        # state
        self.csv_files: List[Tuple[str,pd.DataFrame]] = []
        self.file_ids: dict[str,str] = {}
        # layout
        left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        main = QHBoxLayout(self); main.addLayout(left,2); main.addLayout(center,5); main.addLayout(right,3)
        # left
        left.addWidget(QLabel("📁 Data Files"))
        drop = DropArea(); drop.filesDropped.connect(self.handle_csv_paths); left.addWidget(drop)
        btn_up = QPushButton("Upload CSV"); btn_up.clicked.connect(self.on_upload); left.addWidget(btn_up)
        left.addWidget(QLabel("Loaded Files"))
        self.file_list = QListWidget(); left.addWidget(self.file_list)
        btn_del = QPushButton("Delete Selected"); btn_del.clicked.connect(self.on_delete_files); left.addWidget(btn_del)
        # center
        center.addWidget(QLabel("💬 Ask Agent"))
        self.chat = ChatView(); center.addWidget(self.chat)
        row = QHBoxLayout();
        self.input = QLineEdit(); self.input.returnPressed.connect(self.on_ask); row.addWidget(self.input)
        send = QPushButton("▶"); send.clicked.connect(self.on_ask); row.addWidget(send)
        center.addLayout(row)
        # right
        right.addWidget(QLabel("📊 Results"))
        self.tabs = QTabWidget(); right.addWidget(self.tabs)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl,"Table")
        self.fig,self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas,"Chart")
        self.evidence = QTextEdit(); self.evidence.setReadOnly(True); self.tabs.addTab(self.evidence,"Evidence")
        self.report   = QTextEdit(); self.report.setReadOnly(True); self.tabs.addTab(self.report,"Report")
        self.chat.add_bot("안녕하세요! CSV 업로드 후 질문하세요.")
    # ---- file upload ----
    def on_upload(self):
        paths,_ = QFileDialog.getOpenFileNames(self,"Select CSV",str(get_settings().uploads_dir),"CSV (*.csv)")
        if paths: self.handle_csv_paths(paths)
    def handle_csv_paths(self, paths: list[str]):
        ok=fail=0; prog=QProgressDialog("Loading...","Cancel",0,len(paths),self); prog.setWindowModality(Qt.WindowModal)
        for i,p in enumerate(paths,1):
            prog.setValue(i-1); QApplication.processEvents()
            if prog.wasCanceled(): break
            try:
                df,meta,_=load_and_meta(Path(p),get_settings().meta_json_dir)
                entry=upsert_entry(Path(p),meta['rows'],meta['cols'],'indexed')
                upsert_texts(self.chroma,entry.file_id,build_embedding_texts_from_meta(meta))
                tbl=table_name_from_file(Path(p).name); ingest_df(self.engine,df,tbl); ensure_indexes(self.engine,tbl)
                if _build_meta_for_table: sess=_build_meta_for_table(get_settings().db_url,tbl)
                if _index_sessions: _index_sessions(get_settings().db_url,str(get_settings().vector_db_dir),sess)
                self.csv_files.append((Path(p).name,df)); itm=QListWidgetItem(Path(p).name); itm.setCheckState(Qt.Unchecked); self.file_list.addItem(itm)
                self.chat.add_bot(f"✅ Loaded: {Path(p).name}"); ok+=1
            except Exception as e:
                self.chat.add_bot(f"❌ Load failed: {p}\n{e}"); fail+=1
        prog.setValue(len(paths)); QMessageBox.information(self,"Done",f"Success {ok} / Fail {fail}")
    # ---- delete files ----
    def on_delete_files(self):
        items=[self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).checkState()==Qt.Checked]
        if not items: QMessageBox.information(self,"Notice","No files selected"); return
        if QMessageBox.question(self,"Confirm",f"Delete {len(items)} files?",QMessageBox.Yes)!=QMessageBox.Yes: return
        for it in items:
            name=it.text(); self.csv_files=[(f,df) for f,df in self.csv_files if f!=name]; self.file_list.takeItem(self.file_list.row(it))
        QMessageBox.information(self,"Deleted","Selected files removed")
    # ---- ask agent ----
    def on_ask(self):
        q=self.input.text().strip();
        if not q: return
        self.input.clear(); self.chat.add_user(q)
        self.setEnabled(False)
        def task(): return self.agent.run(q)
        def done(res,err):
            self.setEnabled(True)
            if err: QMessageBox.critical(self,"Error",str(err)); return
            # 이미지 반환
            if isinstance(res,str) and res.lower().endswith('.png'):
                pix=QPixmap(res); self.ax.clear(); self.ax.imshow(pix.toImage()); self.tabs.setCurrentWidget(self.canvas); self.canvas.draw()
            else:
                self.chat.add_bot(res)
        run_in_thread(self,task,done)

if __name__=='__main__':
    app=QApplication(sys.argv); w=MainWindow(); w.show(); sys.exit(app.exec_())

