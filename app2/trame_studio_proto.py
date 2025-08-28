# app2/trame_studio_proto.py
# Trame 3.x + trame-vtk 2.x (Vue3)
# 기능:
#  - 브라우저에서 CSV 경로 입력 → 로드
#  - 대용량 안전장치: 균등 다운샘플링, 튜브 on/off, 디시메이션
#  - PyVista Plotter(ren_win) ↔ VtkLocalView (Local Rendering)
#  - trame-vtk 정적 리소스: 모듈 enable + 경로 강제 매핑(fallback)
#  - 서버 blocking 실행 (포트 8090)

import os
import logging
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import pyvista as pv

from trame.app import get_server
from trame.widgets import html
from trame_vtk.widgets import vtk

# ─────────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trame_proto")

# ─────────────────────────────────────────────────────────────────────
# 서버 초기화 (Vue3)
# ─────────────────────────────────────────────────────────────────────
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# ─────────────────────────────────────────────────────────────────────
# trame-vtk 정적 리소스 등록
# ─────────────────────────────────────────────────────────────────────
def register_trame_vtk_static():
    # 'trame-vtk.js' 파일이 있는 폴더의 실제 경로를 직접 지정합니다.
    # 이 경로는 이전 단계에서 사용자가 직접 찾은 경로입니다.
    path_to_dist = r"C:\Users\KAMIC\Desktop\llm-pyqt-monitor\LLMvenv\Lib\site-packages\trame_vtk\modules\common\serve"
    
    if os.path.isdir(path_to_dist):
        server.enable_module({"serve": {"__trame_vtk": path_to_dist}})
        log.info(f"[trame-vtk] static resources manually served from: {path_to_dist}")
    else:
        log.error(f"[trame-vtk] The specified path does not exist: {path_to_dist}")

register_trame_vtk_static()

# ─────────────────────────────────────────────────────────────────────
# 상태값
# ─────────────────────────────────────────────────────────────────────
state.setdefault("status", "대기 중")
state.setdefault("csv_path", "")
state.setdefault("max_points", 20000)
state.setdefault("use_tube", True)
state.setdefault("tube_radius", 0.15)
state.setdefault("decimate_ratio", 0.6)
state.setdefault("loaded_points", 0)
state.setdefault("total_points", 0)

# ─────────────────────────────────────────────────────────────────────
# PyVista Plotter (Local render target)
# ─────────────────────────────────────────────────────────────────────
_pv_plotter = pv.Plotter(off_screen=True, border=False)
_pv_plotter.set_background("white")

# ─────────────────────────────────────────────────────────────────────
# CSV → PolyLine/Tube 생성
# ─────────────────────────────────────────────────────────────────────
def build_mesh_from_csv(
    csv_path: str,
    max_points: int = 20000,
    use_tube: bool = True,
    tube_radius: float = 0.15,
    decimate_ratio: float = 0.6,
) -> pv.PolyData:
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 경로가 존재하지 않습니다: {csv_path!r}")

    log.info("Loading CSV: %s", csv_path)

    try:
        df = pd.read_csv(csv_path, usecols=["X", "Y", "Z"])
    except Exception:
        df = pd.read_csv(csv_path)

    for c in ("X", "Y", "Z"):
        if c not in df.columns:
            raise ValueError("CSV에 X, Y, Z 컬럼이 필요합니다.")

    n_total = int(len(df))
    step = max(1, int(np.ceil(n_total / max(1, int(max_points)))))
    df_s = df.iloc[::step, :].copy()

    pts = (
        pd.DataFrame(
            {
                "X": pd.to_numeric(df_s["X"], errors="coerce").fillna(0.0),
                "Y": pd.to_numeric(df_s["Y"], errors="coerce").fillna(0.0),
                "Z": pd.to_numeric(df_s["Z"], errors="coerce").fillna(0.0),
            }
        )
        .to_numpy(dtype=float)
    )

    poly = pv.lines_from_points(pts, close=False)

    mesh = poly
    if use_tube:
        mesh = poly.tube(radius=float(tube_radius), n_sides=12, capping=False)
        if 0.0 < float(decimate_ratio) < 0.99:
            try:
                mesh = mesh.decimate(target_reduction=float(decimate_ratio))
            except Exception as de:
                log.warning("Decimate 실패(무시): %s", de)

    state.loaded_points = int(len(df_s))
    state.total_points = n_total

    log.info(
        "Mesh ready (display %d / total %d points)  tube=%s  radius=%.3f  decimate=%.2f",
        state.loaded_points,
        state.total_points,
        use_tube,
        float(tube_radius),
        float(decimate_ratio),
    )
    return mesh

# ─────────────────────────────────────────────────────────────────────
# Plotter 씬 갱신
# ─────────────────────────────────────────────────────────────────────
def update_scene_with_mesh(mesh: pv.PolyData):
    try:
        _pv_plotter.clear()
    except Exception:
        pass

    _pv_plotter.add_mesh(
        mesh,
        color="steelblue",
        lighting=False,
        smooth_shading=False,
        render_lines_as_tubes=False,
        opacity=1.0,
    )
    try:
        _pv_plotter.camera_position = "iso"
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────
# 컨트롤러
# ─────────────────────────────────────────────────────────────────────
@ctrl.add("load_csv_from_path")
def load_csv_from_path():
    try:
        path = state.csv_path.strip()
        if not path:
            state.status = "CSV 경로를 입력하세요."
            server.flush_state()
            return

        state.status = "CSV 로딩 중..."
        server.flush_state()

        mesh = build_mesh_from_csv(
            csv_path=path,
            max_points=int(state.max_points),
            use_tube=bool(state.use_tube),
            tube_radius=float(state.tube_radius),
            decimate_ratio=float(state.decimate_ratio),
        )

        update_scene_with_mesh(mesh)

        view = getattr(server.refs, "view", None)
        if view is not None:
            view.update()
            view.reset_camera()

        state.status = f"로드 완료: 표시 {state.loaded_points} / 원본 {state.total_points} 포인트"
        server.flush_state()
    except Exception as e:
        state.status = f"오류: {type(e).__name__} - {e}"
        log.error("CSV 처리 중 오류", exc_info=True)
        server.flush_state()

@ctrl.add("apply_render_options")
def apply_render_options():
    load_csv_from_path()

# ─────────────────────────────────────────────────────────────────────
# UI (Vue3)
# ─────────────────────────────────────────────────────────────────────
@server.controller.add("on_server_ready")
def ui_layout(**_):
    with server.ui.body:
        html.H2("Trame Studio Prototype — CSV → PolyLine/Tube → WebGL", style="margin: 8px 0;")

        with html.Div(style="margin: 6px 0;"):
            html.Span("상태: ")
            html.Strong(("{status}",))

        with html.Div(style="display:flex; gap:8px; align-items:center; margin: 6px 0;"):
            html.Label("CSV 경로:", style="min-width:80px;")
            html.Input(
                type="text",
                placeholder=r"C:\path\to\your.csv",
                style="flex:1; padding: 6px;",
                v_model=("csv_path", ""),
            )
            html.Button("로드", click=ctrl.load_csv_from_path, style="padding: 6px 12px;")

        with html.Fieldset(style="margin: 6px 0;"):
            html.Legend("렌더링 옵션")
            with html.Div(style="display:flex; flex-wrap:wrap; gap:14px; align-items:center;"):
                with html.Div(style="min-width: 240px;"):
                    html.Label("최대 포인트 (다운샘플링 상한)")
                    html.Input(type="number", min="1000", step="1000", v_model=("max_points", 20000), style="width:100%; padding:6px;")
                with html.Div(style="min-width: 160px;"):
                    html.Label("튜브 사용")
                    html.Input(type="checkbox", v_model=("use_tube", True), style="margin-left:8px;")
                with html.Div(style="min-width: 220px;"):
                    html.Label("튜브 반경")
                    html.Input(type="number", step="0.05", v_model=("tube_radius", 0.15), style="width:100%; padding:6px;")
                with html.Div(style="min-width: 220px;"):
                    html.Label("디시메이션 비율(0.0~0.99, 값이 클수록 많이 줄임)")
                    html.Input(type="number", min="0.0", max="0.99", step="0.05", v_model=("decimate_ratio", 0.6), style="width:100%; padding:6px;")
                html.Button("옵션 적용", click=ctrl.apply_render_options, style="padding: 6px 12px;")

        with html.Div(style="margin: 6px 0; color: #555;"):
            html.Span("표시 포인트: ")
            html.Strong(("{loaded_points}",))
            html.Span(" / 원본: ")
            html.Strong(("{total_points}",))

        # ✅ PyVista Plotter의 vtkRenderWindow 전달
        vtk.VtkLocalView(
            _pv_plotter.ren_win,
            ref="view",
            interactive_ratio=1,
            style="height: calc(100vh - 240px);"
        )

        with html.Div(style="margin-top: 8px; color:#777;"):
            html.Small("팁: 100만 행 이상이면 max_points, decimate_ratio, use_tube 조합으로 빠르게 미리보기 → 필요 시 범위/구간 리플로팅 권장")

# ─────────────────────────────────────────────────────────────────────
# 실행 (blocking) - 포트 8090
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[Trame] server created, Vue client_type =", server.client_type)
    print("[Trame] starting server (blocking)...")
    server.start(
        address="127.0.0.1",
        port=8099,
        open_browser=True,
        exec_mode="main",
    )