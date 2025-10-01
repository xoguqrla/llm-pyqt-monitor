# app/main.py
from __future__ import annotations
import sys, traceback, html
from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy
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
from core.llm_ops import (
    build_llm, build_sql_chain, generate_sql_from_nlq,
    rag_answer, chat_answer
)
from core.hybrid import route, fuse_sql_and_rag
from core.plotting import df_to_table, plot_df_line, build_report_text
from core.files_registry import upsert_entry  # SHA256 기반 파일ID


# -------- global excepthook --------
def _excepthook(et, ev, tb):
    msg = "".join(traceback.format_exception(et, ev, tb))[-4000:]
    print(msg, file=sys.stderr)
    try:
        QMessageBox.critical(None, "Unhandled Error", msg)
    except Exception:
        pass
sys.excepthook = _excepthook


# -------- threading helper --------
class Worker(QObject):
    finished = pyqtSignal(object, object)  # (result, error)
    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn, self.a, self.kw = fn, a, kw
    def run(self):
        try:
            self.finished.emit(self.fn(*self.a, **self.kw), None)
        except Exception as e:
            self.finished.emit(None, e)

def run_in_thread(parent, fn, cb, *a, **kw):
    th = QThread(parent); wk = Worker(fn, *a, **kw); wk.moveToThread(th)
    wk.finished.connect(lambda r, e: (cb(r, e), th.quit(), wk.deleteLater(), th.deleteLater()))
    th.started.connect(wk.run)
    th.start()


# -------- drop area --------
class DropArea(QFrame):
    filesDropped = pyqtSignal(list)  # list[str]
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
        if paths:
            self.filesDropped.emit(paths)
        e.acceptProposedAction()


# ───────── Chat bubbles (Agent-only, right aligned) ─────────
class ChatView(QScrollArea):
    """에이전트 응답만 오른쪽 말풍선으로 누적 표시"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self.setWidget(self._container)

        self.vbox = QVBoxLayout(self._container)
        self.vbox.setSpacing(8)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self.vbox.addStretch()

        self._agent_style = """
            QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;}
            QLabel {color:#0f5132; font-size:13px;}
        """

    def add_agent(self, text: str):
        safe = html.escape(text).replace("\n", "<br>")

        label = QLabel(safe)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        bubble = QFrame()
        bubble.setStyleSheet(self._agent_style)
        bl = QHBoxLayout(bubble)
        bl.setContentsMargins(10, 6, 10, 6)
        bl.addWidget(label)

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addStretch()          # 왼쪽 여백
        hl.addWidget(bubble)     # 오른쪽 말풍선

        # stretch 위에 삽입
        self.vbox.insertWidget(self.vbox.count() - 1, row)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# -------- main window --------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("공정 데이터 LLM 분석 (PyQt)")
        self.resize(1700, 900)
        self.setAcceptDrops(True)

        # services
        s = self.s = get_settings()
        self.engine = make_engine(s.db_url)
        self.llm = build_llm(s.openai_model, s.openai_key, 0)
        self.sql_chain = build_sql_chain(self.llm, s.db_url)  # core/llm_ops에서 스키마 인스펙트
        self.emb = build_embeddings(s.openai_key, s.embed_model)
        self.chroma = build_chroma(self.emb, s.vector_db_dir)

        # state
        self.csv_files: list[tuple[str, pd.DataFrame]] = []
        self.file_ids: dict[str, str] = {}  # {filename: file_id}
        self.last_df: pd.DataFrame | None = None

        # layout
        left, center, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        main = QHBoxLayout(self); main.addLayout(left, 2); main.addLayout(center, 5); main.addLayout(right, 3)

        # left: drop + list + delete
        left.addWidget(QLabel("📁 소스"))
        self.drop = DropArea(); self.drop.filesDropped.connect(self.handle_csv_paths)
        left.addWidget(self.drop)
        self.btn_upload = QPushButton("CSV 업로드"); self.btn_upload.clicked.connect(self.on_upload)
        left.addWidget(self.btn_upload)

        left.addWidget(QLabel("저장된 파일"))
        self.file_list = QListWidget(); left.addWidget(self.file_list, 1)
        self.btn_del = QPushButton("선택 삭제"); self.btn_del.clicked.connect(self.on_delete_files)
        left.addWidget(self.btn_del)

        # center: mode + tone + chat
        center.addWidget(QLabel("💬 LLM 질의"))
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("모드"))
        self.mode = QComboBox(); self.mode.addItems(["Auto", "SQL", "Meta(RAG)", "Hybrid", "Chat"]); mode_row.addWidget(self.mode)
        mode_row.addWidget(QLabel("톤"))
        self.tone = QComboBox(); self.tone.addItems(["전문", "친근"]); mode_row.addWidget(self.tone)
        mode_row.addStretch(1)
        center.addLayout(mode_row)

        self.chat = ChatView()
        center.addWidget(self.chat, 1)

        send_row = QHBoxLayout()
        self.inp = QLineEdit(); self.inp.setPlaceholderText("예) mpt 추세 보여줘 / 이상치 구간 찾아줘 (Enter 전송)")
        self.inp.returnPressed.connect(self.on_ask)
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask)
        self.status = QLabel("")
        send_row.addWidget(self.inp, 1); send_row.addWidget(self.btn_send); send_row.addWidget(self.status)
        center.addLayout(send_row)

        # right: results
        right.addWidget(QLabel("📊 결과/리포트"))
        self.tabs = QTabWidget(); right.addWidget(self.tabs, 1)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl, "표(Table)")
        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas, "그래프(Chart)")
        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.tabs.addTab(self.report, "보고서(Report)")

        # 초기 안내(오른쪽 말풍선)
        self.append_agent("안녕하세요! 업로드 후 질문을 입력해 주세요. (예: mpt 시계열 추세 보여줘)")

    # ---- chat helper ----
    def append_agent(self, text: str):
        self.chat.add_agent(text)

    # ---- window-level dnd fallback ----
    def dragEnterEvent(self, e):
        if any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv") for u in e.mimeData().urls()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")]
        if paths:
            self.handle_csv_paths(paths)
        e.acceptProposedAction()

    # ---- status ----
    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy if False else not busy)  # guard for readability
        self.inp.setReadOnly(busy)
        self.status.setText("🤖 답변 생성 중…" if busy else "")

    # ---- upload ----
    def on_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
        if files:
            self.handle_csv_paths(files)

    def handle_csv_paths(self, paths: list[str]):
        ok = fail = 0
        prog = QProgressDialog("CSV 처리 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)
        for i, p in enumerate(paths, 1):
            prog.setValue(i - 1); QApplication.processEvents()
            if prog.wasCanceled():
                break
            try:
                # 1) CSV -> 메타 생성
                df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)

                # 2) RAG 업서트
                entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
                upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                self.file_ids[Path(p).name] = entry.file_id  # 저장

                # 3) DB 적재 + 인덱스
                table = table_name_from_file(Path(p).name)
                ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)

                # 4) UI
                self.csv_files.append((Path(p).name, df))
                it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
                self.append_agent(f"✅ 업로드 완료: {Path(p).name} (table={table})")
                ok += 1
            except Exception as e:
                self.append_agent(f"❌ 업로드 실패: {p}\n{e}")
                fail += 1

        prog.setValue(len(paths))
        QMessageBox.information(self, "완료", f"성공 {ok} / 실패 {fail}")
        self.update_report_summary()

    # ---- delete ----
    def on_delete_files(self):
        items = [self.file_list.item(i) for i in range(self.file_list.count())
                 if self.file_list.item(i).checkState() == Qt.Checked]
        if not items:
            QMessageBox.information(self, "알림", "체크된 파일이 없습니다.")
            return
        if QMessageBox.question(self, "삭제 확인", f"{len(items)}개 파일을 삭제합니다. 계속할까요?") != QMessageBox.Yes:
            return

        for it in items:
            fname = it.text()

            # 1) 메모리/리스트에서 제거
            self.csv_files = [(f, df) for f, df in self.csv_files if f != fname]
            self.file_list.takeItem(self.file_list.row(it))

            # 2) DB 테이블 삭제 (업로드와 동일 규칙)
            table = table_name_from_file(fname)
            try:
                with self.engine.begin() as c:
                    c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
            except Exception as e:
                self.append_agent(f"⚠️ DB 테이블 삭제 경고: {table} / {e}")

            # 3) Chroma 삭제 (file_id 기반)
            fid = self.file_ids.get(fname)
            if fid:
                ids = [f"{fid}:{i:04d}" for i in range(2000)]
                try:
                    # langchain_chroma 래퍼 아래 실제 컬렉션 접근
                    self.chroma._collection.delete(ids=ids)
                except Exception:
                    try:
                        self.chroma.delete(ids=ids)
                    except Exception as e:
                        self.append_agent(f"⚠️ 임베딩 삭제 경고: {fname} / {e}")
                self.file_ids.pop(fname, None)

        self.update_report_summary()
        self.append_agent("🗑️ 선택 파일 삭제 완료")

    # ---- ask ----
    def on_ask(self):
        q = self.inp.text().strip()
        if not q:
            return
        self.inp.clear()

        mode = self.mode.currentText()
        tone = self.tone.currentText()
        chosen = route(q) if mode == "Auto" else ("RAG" if mode == "Meta(RAG)" else mode)
        self.set_busy(True)

        def _task():
            if chosen == "SQL":
                sql = generate_sql_from_nlq(self.sql_chain, q)
                df = run_sql(self.engine, sql)
                return ("SQL", df, sql, tone)

            if chosen == "RAG":
                docs = retrieve_meta(self.chroma, q, 6)
                ans = rag_answer(self.llm, q, docs, tone=tone)
                return ("RAG", ans, None, tone)

            if chosen == "Chat":
                ans = chat_answer(self.llm, q, tone=tone)
                return ("CHAT", ans, None, tone)

            # Hybrid
            df, sql = None, ""
            try:
                sql = generate_sql_from_nlq(self.sql_chain, q)
                df = run_sql(self.engine, sql)
                df_snip = df.head(20).to_csv(index=False)
            except Exception:
                df_snip = ""
            docs = retrieve_meta(self.chroma, q, 6)
            meta_snip = "\n\n".join(d.page_content for d in docs[:4])
            final = fuse_sql_and_rag(self.llm, q, df_snip, meta_snip, tone=tone)
            return ("HYB", (df, sql, final), None, tone)

        def _done(res, err):
            self.set_busy(False)
            if err:
                QMessageBox.critical(self, "질의 오류", str(err))
                return
            kind, a, b, tone_ = res
            if kind == "SQL":
                df, sql = a, b
                self.render_all(df, sql)
                self.append_agent("SQL 결과를 우측 탭에 표시했어요.")
            elif kind == "RAG":
                self.report.setPlainText(a)
                self.append_agent("요약을 보고서 탭에 넣어두었어요.")
            elif kind == "CHAT":
                self.append_agent(a)
            else:  # HYB
                df, sql, text = a
                self.report.setPlainText(text)
                if df is not None:
                    self.render_all(df, sql)
                self.append_agent("Hybrid 결과를 반영했어요.")

        run_in_thread(self, _task, _done)

    # ---- report ----
    def update_report_summary(self):
        if not self.csv_files:
            self.report.setPlainText("업로드된 데이터가 없습니다. 좌측에서 CSV를 추가하세요.")
            return
        lines = ["# 자동 분석 리포트(데이터 요약)\n"]
        for fname, df in self.csv_files:
            lines += [f"## 파일: {fname}", f"- 행: {len(df)}, 열: {df.shape[1]}"]
            for c in df.select_dtypes(include="number").columns[:10]:
                s = df[c].dropna()
                if s.empty:
                    continue
                lines.append(f"· {c}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
            lines.append("")
        self.report.setPlainText("\n".join(lines))

    # ---- render ----
    def render_all(self, df: pd.DataFrame, sql: str | None):
        view = df.head(self.MAX_ROWS_TABLE)
        step = max(1, len(view)//self.MAX_POINTS_PLOT)
        plot_df = view.iloc[::step] if len(view) > self.MAX_POINTS_PLOT else view
        df_to_table(self.tbl, view)
        plot_df_line(self.ax, self.canvas, plot_df)
        self.report.setPlainText(build_report_text(view, sql))


# entry
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
