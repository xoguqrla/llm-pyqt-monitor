# app/main8_1.py
# 최종 업데이트 : 2025-08-06
# PyQt5 기반 공정 데이터 LLM 분석기 (V1.7+)
# - 대화 메모리, 로그 저장/불러오기/초기화까지 완전 반영

from __future__ import annotations
import sys, traceback, html
from pathlib import Path
import json
import os
from typing import List, Tuple, Optional

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QComboBox, QHeaderView, QMessageBox, QFrame,
    QProgressDialog, QScrollArea, QSizePolicy, QSpacerItem
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
)
from core.plotting import df_to_table, plot_df_line
from core.files_registry import upsert_entry

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
HISTORY_PATH = "history.json"      # 로그 파일명
MAX_HISTORY_TURNS = 3              # 프롬프트 포함 턴 수
MAX_HISTORY_RECORDS = 50           # 파일 최대 기록 수

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
        ok = any(u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")
                 for u in e.mimeData().urls())
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
                 if u.isLocalFile() and u.toLocalFile().lower().endswith(".csv")]
        if paths:
            self.filesDropped.emit(paths)
        e.acceptProposedAction()

# ---------------- chat bubbles (always pinned to bottom) ----------------
class ChatView(QScrollArea):
    """유저=왼쪽, 봇=오른쪽 말풍선 채팅 (최신 하단 고정)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self.setWidget(self._container)
        self.vbox = QVBoxLayout(self._container)
        self.vbox.setSpacing(8)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self._top_spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vbox.addItem(self._top_spacer)
        self._user_style = """
            QFrame {background:#f3f4f6; border-radius:12px; padding:8px 10px;}
            QLabel {color:#111827; font-size:13px;}
        """
        self._bot_style = """
            QFrame {background:#e8f5e9; border-radius:12px; padding:8px 10px;}
            QLabel {color:#0f5132; font-size:13px;}
        """
        self._container.installEventFilter(self)

    def _scroll_to_bottom_now(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _scroll_to_bottom_later(self):
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._scroll_to_bottom_now)

    def _bubble(self, text: str, role: str) -> QWidget:
        import html as _html
        safe = _html.escape(text).replace("\n", "<br>")
        lab = QLabel(safe)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        fr = QFrame()
        fr.setStyleSheet(self._bot_style if role == "bot" else self._user_style)
        fl = QHBoxLayout(fr); fl.setContentsMargins(10, 6, 10, 6)
        fl.addWidget(lab)
        row = QWidget()
        hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        if role == "bot":
            hl.addStretch(); hl.addWidget(fr)
        else:
            hl.addWidget(fr); hl.addStretch()
        return row

    def add_user(self, text: str):
        self.vbox.addWidget(self._bubble(text, "user"))
        self._scroll_to_bottom_now()
        self._scroll_to_bottom_later()

    def add_bot(self, text: str):
        self.vbox.addWidget(self._bubble(text, "bot"))
        self._scroll_to_bottom_now()
        self._scroll_to_bottom_later()

    def clear(self):
        # 전체 채팅 버블 지우기 (초기화)
        for i in reversed(range(self.vbox.count())):
            item = self.vbox.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            self.vbox.removeItem(item)
        self._top_spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vbox.addItem(self._top_spacer)
        self._scroll_to_bottom_now()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._scroll_to_bottom_later()

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj is self._container and event.type() in (QEvent.LayoutRequest, QEvent.Show):
            self._scroll_to_bottom_later()
        return super().eventFilter(obj, event)

# ---------------- LLM prompt helpers (final-only / checks-only) ----------------
def _tone_style(tone: str) -> str:
    return ("말투는 친근하고 공감 있게, 군더더기 없이 자연스럽게."
            if tone == "친근"
            else "말투는 단정하고 간결하게, 불필요한 수식은 피한다.")

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
    prompt = (
        "역할: 제조 공정 데이터 분석 점검관.\n"
        "다음 자료를 보고, 분석을 더 신뢰할 수 있게 만들 **추가 확인 항목** 3~6개를 제안하라.\n"
        "형식: 하이픈(- ) 불릿 리스트만 출력. 다른 문구/제목/서론 금지.\n\n"
        f"[질문]\n{question}\n\n"
        f"[SQL 미리보기(표 일부)]\n{df_snip or '(없음)'}\n\n"
        f"[메타 요약 일부]\n{meta_snip or '(없음)'}\n"
    )
    return llm.invoke(prompt).content

# ---------------- main window ----------------
class MainWindow(QWidget):
    MAX_ROWS_TABLE, MAX_POINTS_PLOT = 5000, 5000

    def build_prompt(self, question: str) -> str:
        context = ""
        for q, a in self.history[-MAX_HISTORY_TURNS:]:
            context += f"이전 Q: {q}\n이전 A: {a}\n"
        prompt = context + f"질문: {question}"
        return prompt

    def save_history(self):
        try:
            hist = self.history[-MAX_HISTORY_RECORDS:]
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(hist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[히스토리 저장 오류] {e}")

    def load_history(self):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = []
        except Exception as e:
            print(f"[히스토리 불러오기 오류] {e}")
            self.history = []

    def clear_history(self):
        self.history = []
        try:
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
        except Exception as e:
            print(f"[히스토리 파일 삭제 오류] {e}")
        # UI 채팅창도 완전 초기화
        if hasattr(self, "chat"):
            self.chat.clear()
            self.chat.add_bot("채팅 로그가 초기화되었습니다.")

    def on_clear_history(self):
        if QMessageBox.question(self, "확인", "정말 모든 채팅 로그를 삭제할까요?") == QMessageBox.Yes:
            self.clear_history()

    def __init__(self):
        self.history: List[Tuple[str, str]] = []
        super().__init__()
        self.setWindowTitle("공정 데이터 LLM 분석 (PyQt) V1.8_1 Ultimate")
        self.resize(1700, 900)
        self.setAcceptDrops(True)

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

        self.csv_files: List[Tuple[str, pd.DataFrame]] = []
        self.file_ids: dict[str, str] = {}
        self.last_df: Optional[pd.DataFrame] = None

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

        # center: tone + chat + input + 초기화 버튼
        center.addWidget(QLabel("💬 LLM 질의"))
        tone_row = QHBoxLayout()
        tone_row.addWidget(QLabel("톤"))
        self.tone = QComboBox(); self.tone.addItems(["전문", "친근"]); tone_row.addWidget(self.tone)
        tone_row.addStretch(1)
        center.addLayout(tone_row)
        self.chat = ChatView(); center.addWidget(self.chat, 1)
        self.btn_clear_history = QPushButton("채팅 로그 초기화")
        self.btn_clear_history.clicked.connect(self.on_clear_history)
        center.addWidget(self.btn_clear_history)
        send_row = QHBoxLayout()
        self.inp = QLineEdit(); self.inp.setPlaceholderText("질문을 입력하고 Enter를 눌러 전송…  (예: mpt 시계열 추세 보여줘)")
        self.inp.returnPressed.connect(self.on_ask)
        self.btn_send = QPushButton("▶"); self.btn_send.clicked.connect(self.on_ask)
        self.status = QLabel("")
        send_row.addWidget(self.inp, 1); send_row.addWidget(self.btn_send); send_row.addWidget(self.status)
        center.addLayout(send_row)

        # right: Table / Chart / Evidence / Report(auto-summary)
        right.addWidget(QLabel("📊 결과/리포트"))
        self.tabs = QTabWidget(); right.addWidget(self.tabs, 1)
        self.tbl = QTableWidget(); self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl, "표(Table)")
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas, "그래프(Chart)")
        self.evidence = QTextEdit(); self.evidence.setReadOnly(True)
        self.tabs.addTab(self.evidence, "근거(Evidence)")
        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.tabs.addTab(self.report, "보고서(Report)")
        self.chat.add_bot("안녕하세요! 업로드 후 질문을 입력해 주세요. (예: mpt 시계열 추세 보여줘)")

        # ----------- 앱 시작시 로그 불러오기 -----------
        self.load_history()
        for q, a in self.history[-MAX_HISTORY_TURNS:]:
            self.chat.add_user(q)
            self.chat.add_bot(a)

    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy)
        self.inp.setReadOnly(busy)
        self.status.setText("🤖 답변 생성 중…" if busy else "")

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
            if prog.wasCanceled(): break
            try:
                df, meta, _ = load_and_meta(Path(p), self.s.meta_json_dir)
                entry = upsert_entry(Path(p), rows=meta["rows"], cols=meta["cols"], status="indexed")
                upsert_texts(self.chroma, entry.file_id, build_embedding_texts_from_meta(meta))
                self.file_ids[Path(p).name] = entry.file_id
                table = table_name_from_file(Path(p).name)
                ingest_df(self.engine, df, table); ensure_indexes(self.engine, table)
                try:
                    if _build_meta_for_table is not None:
                        sessions = _build_meta_for_table(self.s.db_url, table)
                        if _index_sessions is not None:
                            _index_sessions(self.s.db_url, str(self.s.vector_db_dir), sessions)
                except Exception as _e:
                    self.chat.add_bot(f"⚠️ 메타/인덱싱 경고: {Path(p).name}\n{_e}")
                self.csv_files.append((Path(p).name, df))
                it = QListWidgetItem(Path(p).name); it.setCheckState(Qt.Unchecked); self.file_list.addItem(it)
                self.chat.add_bot(f"✅ 업로드 완료: {Path(p).name}\n(table={table})")
                ok += 1
            except Exception as e:
                self.chat.add_bot(f"❌ 업로드 실패: {p}\n{e}")
                fail += 1
        prog.setValue(len(paths))
        QMessageBox.information(self, "완료", f"성공 {ok} / 실패 {fail}")
        self.update_report_summary()

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
            self.csv_files = [(f, df) for f, df in self.csv_files if f != fname]
            self.file_list.takeItem(self.file_list.row(it))
            table = table_name_from_file(fname)
            try:
                with self.engine.begin() as c:
                    c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
            except Exception as e:
                self.chat.add_bot(f"⚠️ DB 테이블 삭제 경고: {table} / {e}")
            fid = self.file_ids.get(fname)
            if fid:
                ids = [f"{fid}:{i:04d}" for i in range(2000)]
                try:
                    self.chroma._collection.delete(ids=ids)
                except Exception:
                    try:
                        self.chroma.delete(ids=ids)
                    except Exception as e:
                        self.chat.add_bot(f"⚠️ 임베딩 삭제 경고: {fname} / {e}")
                self.file_ids.pop(fname, None)
        self.update_report_summary()
        self.chat.add_bot("🗑️ 선택 파일 삭제 완료")

    def on_ask(self):
        q = self.inp.text().strip()
        if not q:
            return
        self.inp.clear()
        self.chat.add_user(q)
        tone = self.tone.currentText()
        self.set_busy(True)
        def _task():
            df, sql, err_sql = None, "", ""
            try:
                sql = generate_sql_from_nlq(self.sql_chain, q, engine_or_url=self.engine)
                df = run_sql(self.engine, sql)
                if isinstance(df, pd.DataFrame) and df.empty:
                    df = None
            except Exception as e:
                err_sql = str(e)
            try:
                docs = retrieve_meta(self.chroma, q, 6)
            except Exception:
                docs = []
            df_snip = ""
            if df is not None:
                try:
                    df_snip = df.head(20).to_csv(index=False)
                except Exception:
                    df_snip = ""
            meta_snip = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs[:4])
            prompt = self.build_prompt(q)
            final_text  = llm_final_only(self.llm, prompt, df_snip, meta_snip, tone)
            checks_list = llm_checks_only(self.llm, prompt, df_snip, meta_snip)
            ev_lines = ["## 사용 근거"]
            if sql:
                ev_lines += ["### 사용 SQL", "```sql", sql.strip(), "```"]
            if isinstance(df, pd.DataFrame):
                ev_lines += ["### SQL 결과 개요", f"- 행 수: {len(df)}", f"- 열 수: {df.shape[1]}"]
            if docs:
                ev_lines.append("### RAG 근거(상위 문서 첫 줄)")
                for i, d in enumerate(docs[:5], 1):
                    first = getattr(d, "page_content", str(d)).splitlines()[0][:200]
                    ev_lines.append(f"{i}. {first}")
            if checks_list:
                ev_lines += ["", "## 추가 확인 항목", checks_list]
            if err_sql and not sql:
                ev_lines += ["", "### SQL 생성/실행 참고", err_sql]
            evidence_text = "\n".join(ev_lines)
            return (q, final_text, df, sql, evidence_text)
        def _done(res, err):
            self.set_busy(False)
            if err:
                QMessageBox.critical(self, "질의 오류", str(err))
                return
            q, final_text, df, sql, evidence_text = res
            self.chat.add_bot(final_text)
            self.history.append((q, final_text))
            # 최신 기록만 저장
            self.save_history()
            if isinstance(df, pd.DataFrame):
                self.render_all(df, sql)
            self.evidence.setPlainText(evidence_text)
        run_in_thread(self, _task, _done)

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

    def render_all(self, df: pd.DataFrame, sql: str | None):
        view = df.head(self.MAX_ROWS_TABLE)
        step = max(1, len(view)//self.MAX_POINTS_PLOT)
        plot_df = view.iloc[::step] if len(view) > self.MAX_POINTS_PLOT else view
        df_to_table(self.tbl, view)
        plot_df_line(self.ax, self.canvas, plot_df)
        self.last_df = df

    def closeEvent(self, event):
        self.save_history()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
