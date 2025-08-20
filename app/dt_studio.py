# app/dt_studio.py
# Digital Twin Studio: Mass-Conserving Bead & Interactive 3D Controls (Drag & Drop CSV)
# ----------------------------------------------------------------------
# 의존성: pyvista, pyvistaqt, PySide6, pandas, numpy
# 실행 예:
#   & .\LLMvenv\Scripts\python.exe .\app\dt_studio.py --csv .\app\merged_data.csv
# ----------------------------------------------------------------------

from __future__ import annotations
import sys, os, math, argparse, traceback
from typing import Optional, Callable
import numpy as np
import pandas as pd



# ----- Qt / PyVista UI -----
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QEvent, QUrl, Signal
import pyvista as pv
try:
    from pyvistaqt import QtInteractor
except Exception as e:
    raise SystemExit(
        "pyvistaqt 가 필요합니다. `pip install pyvistaqt` 후 다시 실행하세요.\n"
        f"Import error: {e}"
    )

# ===========================
# 드래그&드롭 유틸
# ===========================
def _extract_csv_paths(mimedata: QtCore.QMimeData) -> list[str]:
    paths: list[str] = []
    if mimedata.hasUrls():
        for u in mimedata.urls():
            if isinstance(u, QUrl):
                p = u.toLocalFile()
                if p and p.lower().endswith(".csv"):
                    paths.append(p)
    return paths

class FileDropFilter(QtCore.QObject):
    """어떤 위젯에도 설치 가능한 파일 드롭 필터. CSV만 허용."""
    def __init__(self, on_drop: Callable[[list[str]], None], parent=None):
        super().__init__(parent)
        self.on_drop = on_drop

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        et = event.type()
        if et in (QEvent.DragEnter, QEvent.DragMove):
            paths = _extract_csv_paths(event.mimeData())
            if paths:
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Drop:
            paths = _extract_csv_paths(event.mimeData())
            if paths:
                event.acceptProposedAction()
                try:
                    self.on_drop(paths)
                except Exception:
                    # 조용히 로깅만
                    print("Drop handler error:\n", traceback.format_exc())
                return True
            return False
        return False

class DropLineEdit(QtWidgets.QLineEdit):
    """CSV만 드롭 허용하는 라인에디트."""
    fileDropped = Signal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if _extract_csv_paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        if _extract_csv_paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        paths = _extract_csv_paths(e.mimeData())
        if paths:
            self.setText(paths[0])
            self.fileDropped.emit(paths[0])
            e.acceptProposedAction()
        else:
            e.ignore()

# ===========================
# 계산 유틸
# ===========================
def _parse_time(series: pd.Series, time_format: Optional[str], assume_year: Optional[int]) -> pd.Series:
    """시간열을 pandas datetime Series로 변환."""
    if np.issubdtype(series.dtype, np.number):
        t = pd.to_datetime(series, unit="s", origin="unix", errors="coerce")
    else:
        s = series.astype(str)
        if assume_year is not None and (s.str.len() < 15).any():
            s = f"{assume_year}-" + s
        t = pd.to_datetime(s, format=time_format, errors="coerce")
    if t.isna().any():
        t = pd.to_datetime(np.arange(len(series)), unit="s", origin="unix")
    if not isinstance(t, pd.Series):
        t = pd.Series(t, index=series.index)
    return t

def _coerce_bool_col(col: pd.Series) -> pd.Series:
    """혼합형 컬럼을 True/False로 강제."""
    if col.dtype == bool:
        return col
    s = col.astype(str).str.strip().str.lower()
    true_like = {"true", "1", "t", "y", "yes"}
    false_like = {"false", "0", "f", "n", "no", ""}
    out = pd.Series(False, index=col.index)
    out[s.isin(true_like)] = True
    out[s.isin(false_like)] = False
    return out

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window=win, center=True, min_periods=1).median().to_numpy()

def _quantiles(arr, mask=None):
    arr = np.asarray(arr, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = [0, 25, 50, 75, 95, 99, 99.5, 100]
    vals = np.percentile(arr, qs)
    return {f"q{q:g}": float(v) for q, v in zip(qs, vals)}

def compute_mc_bead(
    df: pd.DataFrame,
    d_wire_mm: float,
    # 열 이름
    time_col: str = "time",
    r_rs_col: str = "R_RS",   # 로봇 속도
    r_ws_col: str = "R_WS",   # 와이어 속도
    width_col: str = "MPW",   # 용융풀 폭 -> Bead 폭으로 사용
    laser_col: str = "LASER_ON",
    # 시간 파싱
    time_format: Optional[str] = None,
    assume_year: Optional[int] = None,
    # 단위/스케일 보정
    rs_scale: float = 1.0,    # mm/min → mm/s 변환은 1/60
    ws_scale: float = 1.0,
    width_scale: float = 1.0, # μm → mm 변환은 1/1000
    # 이상치 캡핑
    cap_q: Optional[float] = 0.995,    # r_eq/height 분위수 캡핑 (None 비활성)
    max_radius_mm: Optional[float] = None,
    max_height_mm: Optional[float] = None,
    # 안전 분모
    min_rs_eps: float = 1e-3,
    # 적층 효율(0~1): 1보다 작으면 더 얇아짐
    efficiency: float = 1.0,
    # 단면 모델: "rect" | "semi-ellipse"
    xsection: str = "rect",
) -> pd.DataFrame:
    """질량 보존 기반 비드 파라미터 계산."""
    df = df.copy()

    # 1) Δt
    t = _parse_time(df[time_col], time_format, assume_year)
    dt_sec = t.diff().dt.total_seconds().fillna(0).clip(lower=0)
    df["delta_t"] = dt_sec

    # 2) 스케일 보정
    r_rs_raw = pd.to_numeric(df[r_rs_col], errors="coerce").fillna(0)
    r_ws_raw = pd.to_numeric(df[r_ws_col], errors="coerce").fillna(0)
    width_raw = pd.to_numeric(df.get(width_col, pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0)

    r_rs = (r_rs_raw * rs_scale).astype(float)
    r_ws = (r_ws_raw * ws_scale).astype(float)
    width = (width_raw * width_scale).astype(float)

    # 3) 길이 L
    L = (r_rs * dt_sec).clip(lower=0)
    df["seg_length"] = L

    # 4) 마스크
    laser_val = df.get(laser_col, pd.Series(True, index=df.index))
    laser_bool = _coerce_bool_col(laser_val)
    mask = (r_ws > 0) & (laser_bool)
    df["deposits"] = mask

    # 5) A_bead
    radius_wire = d_wire_mm / 2.0
    A_wire = math.pi * radius_wire * radius_wire
    r_rs_safe = np.maximum(r_rs, min_rs_eps)
    A_bead = efficiency * (A_wire * r_ws) / r_rs_safe
    A_bead = pd.Series(A_bead, index=df.index).fillna(0).where(mask, 0)
    df["A_bead"] = A_bead

    # 6) 폭/높이 (단면 모델)
    eps_w = 1e-6
    width = pd.Series(width, index=df.index).clip(lower=eps_w)
    if xsection == "rect":
        height = (A_bead / width).fillna(0).where(mask, 0)                         # H = A/W
    else:  # semi-ellipse (area = π a b / 2, a=width/2, height=2b)
        a = width / 2.0
        b = (2.0 * A_bead) / (math.pi * a.clip(lower=eps_w))
        height = (2.0 * b).fillna(0).where(mask, 0)

    r_eq = pd.Series(np.sqrt(A_bead / math.pi), index=df.index).where(mask, 0)

    # 7) 캡핑
    if cap_q is not None and 0 < cap_q < 1:
        try:
            rcap = np.nanpercentile(r_eq[mask], cap_q * 100)
            hcap = np.nanpercentile(height[mask], cap_q * 100)
            r_eq = r_eq.clip(upper=float(rcap))
            height = height.clip(upper=float(hcap))
        except Exception:
            pass
    if max_radius_mm is not None:
        r_eq = r_eq.clip(upper=float(max_radius_mm))
    if max_height_mm is not None:
        height = height.clip(upper=float(max_height_mm))

    # 결과 열
    df["Bead_Width"]  = width.where(mask, 0)
    df["Bead_Height"] = height.where(mask, 0)
    df["r_eq"]        = r_eq.where(mask, 0)

    # 진단
    print(">> Diagnostics (after scaling)")
    print("rows=", len(df), "mask_true=", int(mask.sum()))
    print("R_RS(mm/s):", _quantiles(r_rs, mask))
    print("R_WS(mm/s):", _quantiles(r_ws, mask))
    print("MPW(mm):   ", _quantiles(width, mask))
    print("A_bead(mm^2):", _quantiles(A_bead, mask))
    print("r_eq(mm):    ", _quantiles(r_eq, mask))
    print("H(mm):       ", _quantiles(height, mask))

    return df

# ===========================
# 렌더 유틸
# ===========================
def build_mesh(
    df: pd.DataFrame,
    mode: str = "tube",          # 'tube' | 'rect'
    color_by: str = "MPT",
    sample_step: int = 1,
    # tube
    tube_radius_mode: str = "area",  # 'area' | 'width' | 'min'
    tube_gain: float = 1.0,
    smooth_win: int = 1,
    tube_radius_cap: Optional[float] = None,
    # rect
    rect_frame: str = "world",   # 'world' | 'frenet'
    # playback
    progress_ratio: float = 1.0, # 0~1: 경로의 일부까지만 생성
):
    """
    df에서 3D Mesh(pyvista.PolyData)를 생성해 반환.
    성능을 위해 멀티블록 결합을 사용.
    """
    pts = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return None

    mpt = pd.to_numeric(df.get(color_by, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()
    r_eq = df["r_eq"].to_numpy()
    W = df["Bead_Width"].to_numpy()
    H = df["Bead_Height"].to_numpy()
    mask = df["deposits"].to_numpy(dtype=bool)

    step = max(sample_step, 1)
    end_idx = int((len(pts) - 1) * float(np.clip(progress_ratio, 0.0, 1.0)))
    end_idx = max(end_idx, 1)

    # 경로 스플라인(참조)
    spline = pv.Spline(pts[::step], n_points=max(200, len(pts)//step))
    spline["path_scalar"] = np.linspace(0, 1, spline.n_points)

    if mode == "tube":
        # 반지름 결정
        r_area = r_eq
        r_width = W * 0.5
        if tube_radius_mode == "area":
            r_tube = r_area
        elif tube_radius_mode == "width":
            r_tube = r_width
        else:
            r_tube = np.minimum(r_area, r_width)
        r_tube = _rolling_median(r_tube, smooth_win) * float(tube_gain)
        if tube_radius_cap is not None:
            r_tube = np.minimum(r_tube, float(tube_radius_cap))
        geoms = []
        for i in range(0, end_idx, step):
            if not mask[i]:
                continue
            p0, p1 = pts[i], pts[i+1]
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len <= 0:
                continue
            rad = float(max(0.0, r_tube[i]))
            if rad <= 0:
                continue
            cyl = pv.Cylinder(center=(p0 + p1)/2.0, direction=_unit(p1 - p0),
                              radius=rad, height=seg_len)
            val = float(mpt[i]) if not np.isnan(mpt[i]) else 0.0
            cyl["color_scalar"] = np.full(cyl.n_points, val)
            geoms.append(cyl)
        if not geoms:
            return spline
        mb = pv.MultiBlock(geoms)
        mesh = mb.combine()
        mesh["color_scalar"] = mesh["color_scalar"]  # ensure field exists
        return spline, mesh

    # RECT 모드
    def _frame_world(T: np.ndarray):
        bases = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0])]
        for base in bases:
            N = base - np.dot(base, T) * T
            n = np.linalg.norm(N)
            if n > 1e-9:
                N = N / n
                B = np.cross(T, N); B = B / (np.linalg.norm(B)+1e-12)
                return N, B
        return np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])

    def _frame_frenet(i: int, T: np.ndarray):
        if i > 0:
            T_prev = _unit(pts[i] - pts[i-1])
            N = T - T_prev
            n = np.linalg.norm(N)
            if n > 1e-9:
                N = N / n
                B = np.cross(T, N); B = B / (np.linalg.norm(B)+1e-12)
                return N, B
        return _frame_world(T)

    blocks = []
    for i in range(0, end_idx, step):
        if not mask[i]:
            continue
        p0, p1 = pts[i], pts[i+1]
        dir_vec = p1 - p0
        seg_len = float(np.linalg.norm(dir_vec))
        if seg_len <= 0:
            continue
        if W[i] <= 0 or H[i] <= 0:
            continue
        T = _unit(dir_vec)
        if rect_frame == "frenet":
            N, B = _frame_frenet(i, T)
        else:
            N, B = _frame_world(T)
        R = np.column_stack((N, B, T))   # local(x=폭,y=높이,z=길이) -> world
        M = np.eye(4); M[:3,:3] = R; M[:3,3] = (p0 + p1) * 0.5

        cube = pv.Cube(center=(0.0,0.0,0.0),
                       x_length=float(W[i]), y_length=float(H[i]), z_length=seg_len)
        cube.transform(M, inplace=True)
        val = float(mpt[i]) if not np.isnan(mpt[i]) else 0.0
        cube["color_scalar"] = np.full(cube.n_points, val)
        blocks.append(cube)

    if not blocks:
        return spline
    mb = pv.MultiBlock(blocks)
    union = mb.combine()
    union["color_scalar"] = union["color_scalar"]
    return spline, union

# ===========================
# 메인 UI
# ===========================
class DTStudio(QtWidgets.QMainWindow):
    def __init__(self, csv_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("DT Studio — Mass-Conserving Bead (Drag & Drop CSV)")
        self.resize(1400, 900)

        # 상태
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_mc: Optional[pd.DataFrame] = None
        self.last_mesh = None
        self.last_spline = None

        # 중앙 렌더러
        self.pv_widget = QtInteractor(self)
        self.setCentralWidget(self.pv_widget)
        self.plotter: pv.Plotter = self.pv_widget
        self.plotter.add_axes()
        self.plotter.set_background("white")

        # 드래그&드롭(윈도우 전체/렌더뷰/도크/라인에디트)
        self.setAcceptDrops(True)
        self.drop_filter = FileDropFilter(self.on_files_dropped, self)
        self.installEventFilter(self.drop_filter)
        self.pv_widget.setAcceptDrops(True)
        self.pv_widget.installEventFilter(self.drop_filter)

        # 컨트롤 패널
        self._build_controls()

                # =============================
        # 기본 rect 모드 세팅 (부팅 시 자동 적용)
        # =============================
        self.mode.setCurrentText("rect")
        self.auto_clim.setChecked(False)
        self.vmin.setValue(0.0)
        self.vmax.setValue(2000.0)
        self.max_h.setValue(3.0)          # 높이 상한 3 mm
        self.sample_step.setValue(2)
        self.color_col.setCurrentText("MPT")

        # CSV 자동 로드
        if csv_path and os.path.exists(csv_path):
            self.csv_edit.setText(csv_path)
            self.load_csv(csv_path)



    # ---------- UI 구성 ----------
    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls (드래그&드롭 지원)", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        w = QtWidgets.QWidget(); dock.setWidget(w)
        layout = QtWidgets.QVBoxLayout(w); layout.setContentsMargins(8,8,8,8)
        form = QtWidgets.QFormLayout(); layout.addLayout(form)

        # 파일 (드롭 가능한 LineEdit)
        self.csv_edit = DropLineEdit()
        self.csv_btn = QtWidgets.QPushButton("불러오기…")
        self.csv_edit.fileDropped.connect(self.load_csv)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.csv_edit); h.addWidget(self.csv_btn)
        layout.addLayout(h)
        self.csv_btn.clicked.connect(self._choose_csv)

        # 기본 파라미터
        self.wire_d = self._dblspin(1.2, 0.1, 5.0, 0.1)
        self.mode = self._combo(["tube","rect"], "tube")
        self.rect_frame = self._combo(["world","frenet"], "world")
        self.xsection = self._combo(["rect","semi-ellipse"], "rect")
        form.addRow("Wire D (mm)", self.wire_d)
        form.addRow("Render 모드", self.mode)
        form.addRow("Rect 정렬", self.rect_frame)
        form.addRow("단면 모델", self.xsection)

        # 스케일
        self.rs_scale = self._dblspin(1.0, 0.0001, 100, 0.0001); self.rs_scale.setDecimals(6)
        self.ws_scale = self._dblspin(1.0, 0.0001, 100, 0.0001); self.ws_scale.setDecimals(6)
        self.width_scale = self._dblspin(1.0, 1e-6, 1000, 1e-6); self.width_scale.setDecimals(6)
        form.addRow("R_RS 스케일", self.rs_scale)
        form.addRow("R_WS 스케일", self.ws_scale)
        form.addRow("MPW 스케일", self.width_scale)

        # 안전/효율/캡핑
        self.min_rs = self._dblspin(0.001, 0.0, 1.0, 0.0005); self.min_rs.setDecimals(6)
        self.eff = self._dblspin(1.0, 0.05, 1.0, 0.05)
        self.cap_q = self._dblspin(0.995, 0.5, 1.0, 0.001)
        self.max_r = self._dblspin(0.0, 0.0, 50.0, 0.1); self.max_r.setSpecialValueText("None")
        self.max_h = self._dblspin(0.0, 0.0, 50.0, 0.1); self.max_h.setSpecialValueText("None")
        form.addRow("min R_RS (mm/s)", self.min_rs)
        form.addRow("적층 효율 η", self.eff)
        form.addRow("캡 분위수 q", self.cap_q)
        form.addRow("반지름 상한(mm)", self.max_r)
        form.addRow("높이 상한(mm)", self.max_h)

        # 튜브 세부
        self.radius_mode = self._combo(["area","width","min"], "min")
        self.tube_gain = self._dblspin(0.7, 0.05, 3.0, 0.05)
        self.smooth_win = self._intspin(5, 1, 101, 2)  # 홀수 권장
        form.addRow("Tube 반지름", self.radius_mode)
        form.addRow("Tube 게인", self.tube_gain)
        form.addRow("반지름 스무딩 Win", self.smooth_win)

        # 컬러맵/스칼라
        self.color_col = self._combo([], None)  # CSV 로드 후 채움
        self.cmap = self._combo(["plasma","viridis","inferno","coolwarm","turbo"], "plasma")
        self.heat_gain = self._dblspin(1.0, 0.01, 100.0, 0.01)
        self.auto_clim = QtWidgets.QCheckBox("Auto clim"); self.auto_clim.setChecked(True)
        self.vmin = self._dblspin(0.0, -1e9, 1e9, 0.1)
        self.vmax = self._dblspin(1.0, -1e9, 1e9, 0.1)
        form.addRow("색상 스칼라", self.color_col)
        form.addRow("Colormap", self.cmap)
        form.addRow("Heat gain", self.heat_gain)
        form.addRow(self.auto_clim)
        form.addRow("vmin", self.vmin)
        form.addRow("vmax", self.vmax)

        # 필터/재생
        self.sample_step = self._intspin(1, 1, 50, 1)
        self.progress = QtWidgets.QSlider(Qt.Horizontal); self.progress.setRange(1,100); self.progress.setValue(100)
        form.addRow("샘플 스텝", self.sample_step)
        form.addRow("진행(%)", self.progress)

        # 버튼들
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_compute = QtWidgets.QPushButton("계산/렌더")
        self.btn_screenshot = QtWidgets.QPushButton("스크린샷")
        self.btn_export = QtWidgets.QPushButton("메시 내보내기(.vtp)")
        self.btn_reset_cam = QtWidgets.QPushButton("카메라 리셋")
        for b in (self.btn_compute, self.btn_screenshot, self.btn_export, self.btn_reset_cam):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
        # ------------------------------
        # 실제 적층 모드 버튼 추가
        # ------------------------------
        self.btn_realistic = QtWidgets.QPushButton("실제 적층 모드")
        layout.addWidget(self.btn_realistic)

        # 버튼 이벤트 연결
        self.btn_realistic.clicked.connect(self._set_realistic_mode)


        # 로그
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log, 1)

        # 드랍 필터 도크에도 설치(패널에 드롭 허용)
        w.setAcceptDrops(True); w.installEventFilter(self.drop_filter)
        dock.setAcceptDrops(True); dock.installEventFilter(self.drop_filter)

        # 시그널
        self.btn_compute.clicked.connect(self.recompute_and_render)
        self.btn_screenshot.clicked.connect(self.save_screenshot)
        self.btn_export.clicked.connect(self.export_mesh)
        self.btn_reset_cam.clicked.connect(self._reset_cam)


        #1 일부는 즉시 반영
        self.progress.valueChanged.connect(self.recompute_and_render)
        self.heat_gain.valueChanged.connect(self.recompute_and_render)
        self.auto_clim.stateChanged.connect(self.recompute_and_render)
        self.vmin.valueChanged.connect(self.recompute_and_render)
        self.vmax.valueChanged.connect(self.recompute_and_render)
        self.cmap.currentIndexChanged.connect(self.recompute_and_render)

        # # 일부는 실시간 반영
        # self.progress.valueChanged.connect(self._on_quick_update)
        # self.heat_gain.valueChanged.connect(self._on_quick_update)
        # self.auto_clim.stateChanged.connect(self._on_quick_update)
        # self.vmin.valueChanged.connect(self._on_quick_update)
        # self.vmax.valueChanged.connect(self._on_quick_update)
        # self.cmap.currentIndexChanged.connect(self._on_quick_update)

    def _set_realistic_mode(self):
        """실제 적층 모드: 실험 기반 기본 파라미터 자동 세팅"""
        try:
            # 실제 공정값에 맞춘 추천 세팅
            self.wire_d.setValue(0.7)         # 실제 와이어 직경 (mm)
            self.rs_scale.setValue(0.8)       # RS 스케일
            self.ws_scale.setValue(0.8)       # WS 스케일
            self.width_scale.setValue(0.75)   # MPW 스케일
            self.tube_gain.setValue(0.4)      # Tube gain
            self.rect_frame.setCurrentText("world")  # 좌표계 정렬
            self.heat_gain.setValue(1.0)      # 색상 대비는 1.0

            self.log.appendPlainText(">> ✅ 실제 적층 모드로 기본 파라미터 적용 완료")
            self.recompute_and_render()

        except Exception as e:
            self.log.appendPlainText(f"[Error] Realistic Mode 적용 실패: {e}")







    # ---- 위젯 팩토리 ---
    def _dblspin(self, val, lo, hi, step):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val); sb.setDecimals(6)
        sb.setMaximumWidth(160)
        return sb
    def _intspin(self, val, lo, hi, step):
        sb = QtWidgets.QSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val)
        sb.setMaximumWidth(160)
        return sb
    def _combo(self, items, current):
        cb = QtWidgets.QComboBox()
        cb.addItems(items)
        if current is not None and current in items:
            cb.setCurrentText(current)
        cb.setMaximumWidth(180)
        return cb

    # ---------- 드롭 콜백 ----------
    def on_files_dropped(self, paths: list[str]):
        # 여러개면 첫 번째 CSV 사용
        if not paths:
            return
        path = paths[0]
        self.csv_edit.setText(path)
        self.load_csv(path)

    # ---------- 동작 ----------
    def _choose_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "CSV 선택", "", "CSV (*.csv);;All Files (*)")
        if path:
            self.csv_edit.setText(path)
            self.load_csv(path)

    def log_msg(self, msg: str):
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def load_csv(self, path: Optional[str] = None):
        if not path:
            path = self.csv_edit.text().strip()
        if not path:
            return
        try:
            self.df_raw = pd.read_csv(path)
            self.log_msg(f"CSV 로드: {path}  (rows={len(self.df_raw)})")
            # 색상 컬럼 후보 채우기
            cols = list(self.df_raw.columns)
            self.color_col.clear()
            self.color_col.addItems(cols)
            # 기본 선택
            if "MPT" in cols:
                self.color_col.setCurrentText("MPT")
            elif "A_bead" in cols:
                self.color_col.setCurrentText("A_bead")
            # 좌표 존재 검사
            for c in ("X","Y","Z"):
                if c not in self.df_raw.columns:
                    raise ValueError(f"필수 좌표열 {c} 가 없습니다.")
            # 자동 계산/렌더
            self.recompute_and_render()
        except Exception:
            self.log_msg("로드 오류:\n" + traceback.format_exc())

    def recompute_and_render(self):
        if self.df_raw is None:
            return
        try:
            # 계산
            df = compute_mc_bead(
                self.df_raw,
                d_wire_mm=self.wire_d.value(),
                time_col="time",
                r_rs_col="R_RS",
                r_ws_col="R_WS",
                width_col="MPW",
                laser_col="LASER_ON",
                time_format="%m-%d %H:%M:%S.%f",
                assume_year=2025,
                rs_scale=self.rs_scale.value(),
                ws_scale=self.ws_scale.value(),
                width_scale=self.width_scale.value(),
                cap_q=float(self.cap_q.value()) if self.cap_q.value() < 0.9999 else 0.9999,
                max_radius_mm=self.max_r.value() if self.max_r.value() > 0 else None,
                max_height_mm=self.max_h.value() if self.max_h.value() > 0 else None,
                min_rs_eps=self.min_rs.value(),
                efficiency=self.eff.value(),
                xsection=self.xsection.currentText(),
            )
            self.df_mc = df
            self._render_full()
        except Exception:
            self.log_msg("계산/렌더 오류:\n" + traceback.format_exc())


    def _reset_cam(self):
        # 카메라 위치 초기화
        self.plotter.camera_position = None
        # 데이터 전체를 보기 좋은 시점으로 리셋
        self.plotter.reset_camera()


    def _render_full(self):
        if self.df_mc is None:
            return
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.set_background("white")

        # 메시 만들기
        out = build_mesh(
            self.df_mc,
            mode=self.mode.currentText(),
            color_by=self.color_col.currentText(),
            sample_step=self.sample_step.value(),
            tube_radius_mode=self.radius_mode.currentText(),
            tube_gain=self.tube_gain.value(),
            smooth_win=self.smooth_win.value(),
            tube_radius_cap=self.max_r.value() if self.max_r.value() > 0 else None,
            rect_frame=self.rect_frame.currentText(),
            progress_ratio=self.progress.value()/100.0,
        )

        # 결과가 (spline, mesh) 또는 spline 하나일 수 있음
        if isinstance(out, tuple):
            spline, mesh = out
            self.last_spline = spline; self.last_mesh = mesh
            self.plotter.add_mesh(spline, color="black", line_width=1, opacity=0.15)
            scalars = mesh["color_scalar"].copy()
            scalars *= self.heat_gain.value()  # heat gain
            # clim
            if self.auto_clim.isChecked():
                vmin, vmax = float(np.nanmin(scalars)), float(np.nanmax(scalars))
            else:
                vmin, vmax = self.vmin.value(), self.vmax.value()
            self.plotter.add_mesh(mesh, scalars=scalars, cmap=self.cmap.currentText(),
                                  smooth_shading=(self.mode.currentText()=="tube"),
                                  clim=(vmin, vmax))
        else:
            # spline만 있는 경우
            spline = out
            self.last_spline = spline; self.last_mesh = None
            self.plotter.add_mesh(spline, color="black", line_width=2)

        self.plotter.reset_camera()

    def _on_quick_update(self, *_):



     
    # Render 모드가 rect일 때 기본 세팅 적용
        if self.mode.currentText() == "rect":
            # Auto clim 해제
            self.auto_clim.setChecked(False)

            # vmin / vmax 수동 설정
            self.vmin.setValue(0.0)
            self.vmax.setValue(2000.0)

            # 높이 상한(mm) = 3
            idx = self.z_cap.findText("3")  # 콤보박스에서 "3" 찾기
            if idx != -1:
                self.z_cap.setCurrentIndex(idx)

            # 샘플 스텝 = 2
            self.sample_step.setValue(2)

            # 색상 스칼라 = MPT
            idx = self.scalar.findText("MPT")
            if idx != -1:
                self.scalar.setCurrentIndex(idx)

        # 원래 있던 갱신 로직 호출
        self.recompute_and_render()

        """색상/재생 슬라이더 등 빠른 재렌더."""
        if self.df_mc is None:
            return
        # 메시만 클리어 후 다시 올림
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.set_background("white")

        out = build_mesh(
            self.df_mc,
            mode=self.mode.currentText(),
            color_by=self.color_col.currentText(),
            sample_step=self.sample_step.value(),
            tube_radius_mode=self.radius_mode.currentText(),
            tube_gain=self.tube_gain.value(),
            smooth_win=self.smooth_win.value(),
            tube_radius_cap=self.max_r.value() if self.max_r.value() > 0 else None,
            rect_frame=self.rect_frame.currentText(),
            progress_ratio=self.progress.value()/100.0,
        )
        if isinstance(out, tuple):
            spline, mesh = out
            self.plotter.add_mesh(spline, color="black", line_width=1, opacity=0.15)
            scalars = mesh["color_scalar"].copy() * self.heat_gain.value()
            if self.auto_clim.isChecked():
                vmin, vmax = float(np.nanmin(scalars)), float(np.nanmax(scalars))
            else:
                vmin, vmax = self.vmin.value(), self.vmax.value()
            self.plotter.add_mesh(mesh, scalars=scalars, cmap=self.cmap.currentText(),
                                  smooth_shading=(self.mode.currentText()=="tube"),
                                  clim=(vmin, vmax))
        else:
            self.plotter.add_mesh(out, color="black", line_width=2)

        self.plotter.render()

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "스크린샷 저장", "dt_view.png", "PNG(*.png)")
        if not path:
            return
        try:
            self.plotter.screenshot(path)
            self.log_msg(f"스크린샷 저장: {path}")
        except Exception:
            self.log_msg("스크린샷 오류:\n" + traceback.format_exc())

    def export_mesh(self):
        if self.last_mesh is None:
            self.log_msg("내보낼 메시가 없습니다.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "메시 내보내기", "beads.vtp", "VTK PolyData (*.vtp)")
        if not path:
            return
        try:
            self.last_mesh.save(path)
            self.log_msg(f"메시 저장: {path}")
        except Exception:
            self.log_msg("메시 저장 오류:\n" + traceback.format_exc())

    # ---- 위젯 팩토리 끝 ----

# ===========================
# 엔트리
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="초기 로드할 CSV 경로(선택)")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = DTStudio(csv_path=args.csv)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
