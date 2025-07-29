# app/main.py
from __future__ import annotations

import sys, traceback, shutil
from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QTextEdit, QTabWidget, QComboBox,
    QHeaderView, QMessageBox, QFrame, QProgressDialog
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ─── core modules ───
from core.config import get_settings
from core.csv_ops import load_and_meta
from core.db_ops import make_engine, ingest_df, ensure_indexes, run_sql, table_name_from_file
from core.rag_ops import (
    build_embeddings, build_chroma, retrieve_meta,
    build_embedding_texts_from_meta, upsert_texts
)
from core.llm_ops import build_llm, build_sql_chain, generate_sql_from_nlq, rag_answer
from core.hybrid import route, fuse_sql_and_rag
from core.plotting import df_to_table, plot_df_line, build_report_text
from core.files_registry import upsert_entry  # SHA256 기반 파일ID


# ───────── Global excepthook ─────────
def _excepthook(exctype, value, tb):
    msg = "".join(traceback.format_exception(exctype, value, tb))[-4000:]
    print(msg, file=sys.stderr)
    try:
        QMessageBox.critical(None, "Unhandled Error", msg)
    except Exception:
        pass
sys.excepthook = _excepthook


# ───────── 간단 워커(비동기) ─────────
class Worker(QObject):
    finished = pyqtSignal(object, object)  # (result, error)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            res = self.fn(*self.args, **self.kw)
            self.finished.emit(res, None)
        except Exception as e:
            self.finished.emit(None, e)


def run_in_thread(parent, fn, cb, *args, **kw):
    th = QThread(parent)
    wk = Worker(fn, *args, **kw)
    wk.moveToThread(th)
    wk.finished.connect(lambda r, e: (cb(r, e), th.quit(), wk.deleteLater(), th.deleteLater()))
    th.started.connect(wk.run)
    th.start()


# ───────── Drag & Drop ─────────
class DropArea(QFrame):
    filesDropped = pyqtSignal(list)  # list[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame { border: 2px dashed #9ca3af; border-radius: 10px; background: #fafafa; }
            QFrame[drag='true'] { border-color: #2563eb; background: #eef2ff; }
        """)
        lay = QVBoxLayout(self)
        lab = QLabel("여기로 CSV 파일을 드래그 앤 드롭"); lab.setAlignment(Qt.AlignCenter)
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


# ───────── Main Window ─────────
class MainWindow(QWidget):
    MAX_ROWS_TABLE = 5000
    MAX_POINTS_PLOT = 5000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("공정 데이터 LLM 분석 (PyQt)")
        self.resize(1700, 900)

        # services ------------------------------------------------------------
        self.s = get_settings()
        if not self.s.db_url:
            QMessageBox.critical(self, "오류", ".env의 DATABASE_URL이 비었습니다.")
        self.engine = make_engine(self.s.db_url)
        try:
            with self.engine.begin() as c: c.exec_driver_sql("SELECT 1")
        except Exception as e:
            QMessageBox.critical(self, "DB 연결 실패", str(e))

        self.llm = build_llm(self.s.openai_model, self.s.openai_key, temperature=0.0)
        self.sql_chain = build_sql_chain(self.llm, self.s.db_url)
        self.embeddings = build_embeddings(self.s.openai_key, self.s.embed_model)
        self.chroma = build_chroma(self.embeddings, self.s.vector_db_dir)

        # state ---------------------------------------------------------------
        self.csv_files: list[tuple[str, pd.DataFrame]] = []
        self.last_df: pd.DataFrame | None = None

        # layout --------------------------------------------------------------
        main = QHBoxLayout(self)

        # left ----------------------------------------------------------------
        left = QVBoxLayout()
        left.addWidget(QLabel("📁 소스"))

        self.drop = DropArea(); self.drop.filesDropped.connect(self.handle_csv_paths)
        left.addWidget(self.drop)

        self.btn_upload = QPushButton("CSV 업로드"); self.btn_upload.clicked.connect(self.on_upload)
        left.addWidget(self.btn_upload)

        left.addWidget(QLabel("저장된 파일"))
        self.file_list = QTextEdit(); self.file_list.setReadOnly(True)
        left.addWidget(self.file_list)
        left.addStretch()

        # center --------------------------------------------------------------
        center = QVBoxLayout()
        center.addWidget(QLabel("💬 LLM 질의"))
        center.addWidget(QLabel("모드 선택"))

        self.mode = QComboBox(); self.mode.addItems(["Auto","SQL","Meta(RAG)","Hybrid","Chat"])
        center.addWidget(self.mode)

        self.chat = QTextEdit(); self.chat.setReadOnly(True)
        center.addWidget(self.chat, 1)

        row = QHBoxLayout()
        self.inp = QLineEdit(); self.inp.setPlaceholderText("질문을 입력하세요 (Enter로 전송)")
        self.inp.returnPressed.connect(self.on_ask)                     # NEW: Enter 전송
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask)
        self.btn_send.setAutoDefault(True); self.btn_send.setDefault(True)
        self.status_lbl = QLabel("")                                    # NEW: 상태 표시

        row.addWidget(self.inp, 1); row.addWidget(self.btn_send); row.addWidget(self.status_lbl)
        center.addLayout(row)

        # right --------------------------------------------------------------
        right = QVBoxLayout()
        right.addWidget(QLabel("📊 결과/리포트"))
        self.tabs = QTabWidget()

        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl, "표(Table)")

        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas, "그래프(Chart)")

        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.tabs.addTab(self.report, "보고서(Report)")
        right.addWidget(self.tabs)

        main.addLayout(left, 2); main.addLayout(center, 5); main.addLayout(right, 3)

    # -------------- 상태 표시 --------------
    def set_busy(self, flag: bool):
        if flag:
            self.btn_send.setEnabled(False)
            self.inp.setReadOnly(True)
            self.status_lbl.setText("🤖 답변 생성 중…")
        else:
            self.btn_send.setEnabled(True)
            self.inp.setReadOnly(False)
            self.status_lbl.setText("")

    # -------------- 업로드 --------------
    def on_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", str(self.s.uploads_dir), "CSV Files (*.csv)")
        if files: self.handle_csv_paths(files)

    def handle_csv_paths(self, paths: list[str]):
        ok = fail = 0
        prog = QProgressDialog("CSV 처리 중...", "취소", 0, len(paths), self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(300)

        for i, p in enumerate(paths, 1):
            prog.setValue(i - 1); QApplication.processEvents()
            if prog.wasCanceled(): break
            try:
                df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)

                # RAG 인덱싱
                try:
                    texts_meta = build_embedding_texts_from_meta(meta)
                    entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
                    upsert_texts(self.chroma, entry.file_id, texts_meta)
                except Exception as e:
                    self.chat.append(f"⚠️ 임베딩 인덱싱 경고: {e}")

                # DB 적재
                table = table_name_from_file(Path(p).name)
                ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)

                # UI
                self.csv_files.append((Path(p).name, df))
                self.file_list.append(Path(p).name)
                self.chat.append(f"✅ 업로드 & DB 적재 성공: {p} (table={table})")
                ok += 1
            except Exception as e:
                fail += 1
                self.chat.append(f"❌ 업로드/DB 실패: {p}<br/>{e}")
                continue

        prog.setValue(len(paths))
        if fail: QMessageBox.warning(self, "업로드 완료(일부 실패)", f"성공 {ok}개 / 실패 {fail}개")
        else: QMessageBox.information(self, "업로드 완료", f"성공 {ok}개")

        # NEW: 최신 요약 리포트 갱신
        self.update_report_summary()

    # -------------- 질의 --------------
    def on_ask(self):
        q = self.inp.text().strip()
        if not q: return
        self.inp.clear()
        self.chat.append(f"<b>질문:</b> {q}")
        mode = self.mode.currentText()
        chosen = route(q) if mode == "Auto" else ("RAG" if mode=="Meta(RAG)" else mode)

        # 비동기 시작
        self.set_busy(True)

        def _task():
            if chosen == "SQL":
                sql = generate_sql_from_nlq(self.sql_chain, q); df = run_sql(self.engine, sql)
                return ("SQL", df, sql)
            if chosen == "RAG":
                docs = retrieve_meta(self.chroma, q, k=6); ans = rag_answer(self.llm, q, docs)
                return ("RAG", ans, None)
            if chosen == "Chat":
                ans = self.llm.invoke(q).content
                return ("CHAT", ans, None)
            # Hybrid
            df, sql = None, ""
            try:
                sql = generate_sql_from_nlq(self.sql_chain, q); df = run_sql(self.engine, sql)
                df_snip = df.head(20).to_csv(index=False)
            except Exception: df_snip=""
            docs = retrieve_meta(self.chroma, q, k=6)
            meta_snip = "\n\n".join(d.page_content for d in docs[:4])
            final = fuse_sql_and_rag(self.llm, q, df_snip, meta_snip)
            return ("HYB", (df, sql, final), None)

        def _done(res, err):
            self.set_busy(False)
            if err:
                QMessageBox.critical(self, "질의 오류", str(err)); return
            kind, a, b = res
            if kind == "SQL":
                df, sql = a, b; self.render_all(df, sql)
                self.chat.append("<b>응답:</b> SQL 결과를 우측 탭에 표시했습니다.")
            elif kind == "RAG":
                self.report.setPlainText(a); self.chat.append("<b>응답:</b> RAG 요약을 보고서 탭에 표시했습니다.")
            elif kind == "CHAT":
                self.chat.append(f"<b>어시스턴트:</b> {a}")
            else:  # HYB
                df, sql, final = a
                self.report.setPlainText(final)
                if df is not None: self.render_all(df, sql)
                self.chat.append("<b>응답:</b> Hybrid 결과를 우측 탭에 표시했습니다.")

        run_in_thread(self, _task, _done)

    # -------------- 리포트 생성 --------------
    def update_report_summary(self):
        if not self.csv_files:
            self.report.setPlainText("업로드된 데이터가 없습니다. 좌측에서 CSV를 추가하세요.")
            return
        lines = ["# 자동 분석 리포트(데이터 요약)\n"]
        for fname, df in self.csv_files:
            lines.append(f"## 파일: {fname}")
            lines.append(f"- 행: {len(df)}, 열: {df.shape[1]}")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                lines.append("### 수치형 컬럼 통계")
                for c in num_cols[:10]:
                    s = df[c].dropna()
                    if s.empty: continue
                    lines.append(f"- **{c}**: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}")
            else:
                lines.append("- 수치형 컬럼이 없습니다.")
            lines.append("")
        self.report.setPlainText("\n".join(lines))

    # -------------- 표/그래프 렌더 --------------
    def render_all(self, df: pd.DataFrame, sql: str|None):
        df_view = df.head(self.MAX_ROWS_TABLE).copy() if len(df)>self.MAX_ROWS_TABLE else df.copy()
        step = max(1, len(df_view)//self.MAX_POINTS_PLOT)
        df_plot = df_view.iloc[::step].copy() if len(df_view)>self.MAX_POINTS_PLOT else df_view

        self.last_df = df
        df_to_table(self.tbl, df_view)
        plot_df_line(self.ax, self.canvas, df_plot)
        self.report.setPlainText(build_report_text(df_view, sql))


# ───────── entry ─────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
