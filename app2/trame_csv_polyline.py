import pandas as pd
import pyvista as pv

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, vtk as vtk_widgets

# -------------------------------
# Trame 서버 초기화
# -------------------------------
server = get_server()
state, ctrl = server.state, server.controller

# 초기 빈 Mesh
current_mesh = pv.PolyData()
view = None


# -------------------------------
# CSV -> PolyLine 변환 함수
# -------------------------------
def load_csv(path):
    global current_mesh, view

    df = pd.read_csv(path)

    # X, Y, Z 좌표 있다고 가정
    if not all(c in df.columns for c in ["X", "Y", "Z"]):
        print("CSV must contain columns: X, Y, Z")
        return

    points = df[["X", "Y", "Z"]].values
    polyline = pv.lines_from_points(points)
    tube = polyline.tube(radius=0.1)

    current_mesh = tube

    if view:
        view.update_object(tube)   # 기존 뷰 업데이트


# -------------------------------
# UI 정의
# -------------------------------
def ui_layout():
    global view

    with SinglePageLayout(server) as layout:
        layout.title.set_text("CSV → PolyLine Viewer (Trame Prototype)")

        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-4"):
                vuetify.VFileInput(
                    label="Drag & Drop CSV here",
                    show_size=True,
                    dense=True,
                    outlined=True,
                    accept=".csv",
                    v_model=("csv_file", None),
                    change=ctrl.on_file_selected,
                )

                view = vtk_widgets.VtkLocalView(current_mesh, ref="view", interactive_ratio=1)

    return layout


# -------------------------------
# 이벤트 핸들러
# -------------------------------
@ctrl.add("on_file_selected")
def file_selected(event):
    file_path = state.csv_file
    if file_path:
        print(f"Loading CSV: {file_path}")
        load_csv(file_path)


# -------------------------------
# 실행
# -------------------------------
ui_layout()
server.start()
