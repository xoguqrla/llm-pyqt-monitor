# core/analysis_visualizer.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 등록용)
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class AnalysisVisualizer:
    """
    시각화/상관분석/3D 경로/파이비스타 시뮬레이션 준비 유틸.
    - 상관계수: 피어슨(명시), 숫자화/불리언 일관처리, LASER_ON True 필터 반영
    - 안정성 플롯: 누락 컬럼 방어
    - 3D 경로: 대용량 자동 샘플링(성능 보호)
    - PyVista: 시간 파싱 보강, spline 가이드 라인, 고성능 튜브 라인 렌더
    """

    VERSION = "v2"

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df
        self.cursor_connections = []

    # -------------------- 내부 유틸 --------------------

    def _clear_connections(self, fig):
        for cid in self.cursor_connections:
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self.cursor_connections.clear()

    def _coerce_bool(self, s: pd.Series) -> pd.Series:
        """불리언으로 일관 변환(True/False/NA)."""
        if s.dtype == bool:
            return s
        if s.dtype.kind in "iu":  # int/uint
            return s.astype("Int64").map(
                lambda x: True if x == 1 else (False if x == 0 else pd.NA)
            )
        return (
            s.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True, "1": True, "y": True, "yes": True, "t": True,
                    "false": False, "0": False, "n": False, "no": False, "f": False
                }
            )
        )

    def _to_numeric(self, cols):
        """선택 컬럼을 숫자형으로 강제 변환(errors='coerce')."""
        out = {}
        for c in cols:
            if c in self.df.columns:
                out[c] = pd.to_numeric(self.df[c], errors="coerce")
        return pd.DataFrame(out)

    # -------------------- 시각화: 안정성 --------------------

    def plot_stability(self, fig):
        """
        LP/WS/MPT 안정성 플롯.
        - 누락 컬럼 방어: 없으면 안내 텍스트 출력
        - 마우스 커서 가이드(디바운싱 포함)
        """
        self._clear_connections(fig)
        fig.clear()
        axes = fig.subplots(nrows=3, ncols=1)
        fig.suptitle('Process Stability Analysis', fontsize=16)

        # 1) LP
        ax = axes[0]
        plotted = False
        if 'S_LP' in self.df.columns:
            ax.plot(self.df.index, self.df['S_LP'], label='Set', linestyle='--')
            plotted = True
        if 'R_LP' in self.df.columns:
            ax.plot(self.df.index, self.df['R_LP'], label='Actual', alpha=0.8)
            plotted = True
        ax.set_title('Laser Power (LP)')
        ax.set_ylabel('Power (W)')
        if plotted:
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "LP columns missing", ha='center')
            ax.axis('off')

        # 2) WS
        ax = axes[1]
        plotted = False
        if 'S_WS' in self.df.columns:
            ax.plot(self.df.index, self.df['S_WS'], label='Set', linestyle='--')
            plotted = True
        if 'R_WS' in self.df.columns:
            ax.plot(self.df.index, self.df['R_WS'], label='Actual', alpha=0.8)
            plotted = True
        ax.set_title('Wire Feed Speed (WS)')
        ax.set_ylabel('Speed')
        if plotted:
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "WS columns missing", ha='center')
            ax.axis('off')

        # 3) MPT
        ax = axes[2]
        if 'MPT' in self.df.columns:
            ax.plot(self.df.index, self.df['MPT'], label='MPT', color='crimson')
            ax.set_title('Melt Pool Temperature (MPT)')
            ax.set_xlabel('Data Index (Time Flow)')
            ax.set_ylabel('Temperature')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "MPT column missing", ha='center')
            ax.axis('off')

        # 커서(디바운싱 포함)
        for ax in axes:
            cursor = Cursor(ax)
            cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
            self.cursor_connections.append(cid)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # -------------------- 시각화: 상관 대시보드 --------------------

    def plot_correlation_dashboard(self, fig):
        """
        피어슨 상관 대시보드(히트맵 + 상/하위 상관쌍 표).
        - LASER_ON True 필터(존재 시)
        - CONTACT 불리언을 0/1로 수치화
        - 상삼각만 사용해 탑-페어 선정
        """
        self._clear_connections(fig)
        fig.clear()

        df = self.df.copy()
        # LASER_ON 있으면 True만 사용
        if 'LASER_ON' in df.columns:
            mask = self._coerce_bool(df['LASER_ON'])
            df = df[mask == True].copy()

        corr_cols = ['R_LP', 'R_RS', 'R_WS', 'MPT', 'MPA', 'MPW', 'LOAD', 'CONTACT']
        present = [c for c in corr_cols if c in df.columns]

        # CONTACT가 있으면 0/1로 일관 수치화
        if 'CONTACT' in present:
            contact_cast = self._coerce_bool(df['CONTACT']).map({True: 1, False: 0})
            df = df.assign(CONTACT=contact_cast)

        num_df = self._to_numeric(present)
        if num_df.shape[1] < 2 or num_df.dropna(how='all').empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough data for correlation analysis.", ha='center')
            ax.axis('off')
            return fig

        # 피어슨 상관(명시)
        corr_matrix = num_df.corr(method='pearson', min_periods=2)

        # 상삼각(대각 제외)만 추출해 페어 나열
        cols = corr_matrix.columns
        tri_idx = np.triu_indices(len(cols), k=1)
        pairs_series = pd.Series(
            corr_matrix.values[tri_idx],
            index=pd.MultiIndex.from_arrays([cols[tri_idx[0]], cols[tri_idx[1]]])
        ).sort_values(ascending=False)

        positive_corr = pairs_series[pairs_series > 0.3].head(5)
        negative_corr = pairs_series[pairs_series < -0.3].sort_values(ascending=True).head(5)

        gs = fig.add_gridspec(1, 2, width_ratios=[6, 4], wspace=0.3)
        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1])

        fig.suptitle('Correlation Analysis Dashboard', fontsize=16, weight='bold')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=.5, ax=ax_left, annot_kws={"size": 9}
        )
        ax_left.set_title('Correlation Heatmap', fontsize=12)
        ax_left.tick_params(axis='x', rotation=45)
        ax_left.tick_params(axis='y', rotation=0)

        ax_right.axis('off')
        self._draw_correlation_table(ax_right, "Top 5 Positive Correlations", positive_corr)
        self._draw_correlation_table(ax_right, "Top 5 Negative Correlations", negative_corr, y_offset=0.45)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _draw_correlation_table(self, ax, title, data, y_offset=0.95):
        ax.text(0.5, y_offset, title, ha='center', va='bottom', fontsize=12, weight='bold')
        if isinstance(data, pd.Series) and not data.empty:
            cell_text = [[f'{str(a)} & {str(b)}', f'{val:.3f}'] for (a, b), val in data.items()]
            table = ax.table(
                cellText=cell_text, colLabels=['Variable Pair', 'Correlation'],
                cellLoc='center', loc='center', bbox=[0, y_offset - 0.45, 1, 0.4]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
        else:
            ax.text(0.5, y_offset - 0.2, "No significant correlation.", ha='center', va='center', fontsize=10)

    # -------------------- 시각화: 3D 경로(정적) --------------------

    def plot_3d_path(self, fig):
        """
        3D 산점도 경로(MPT 컬러). 매우 큰 데이터는 자동 샘플링으로 부담 완화.
        """
        self._clear_connections(fig)
        fig.clear()

        df_plot = self.df
        n = len(df_plot)
        # 대용량 자동 샘플링(예: 50k 초과 시 20k로 다운샘플)
        if n > 50000:
            df_plot = df_plot.sample(n=20000, random_state=42).sort_index()

        ax = fig.add_subplot(111, projection='3d')
        if not set(['X', 'Y', 'Z', 'MPT']).issubset(df_plot.columns):
            ax.text(0.5, 0.5, "X/Y/Z/MPT columns required.", ha='center')
            ax.axis('off')
            fig.tight_layout()
            return fig

        sc = ax.scatter(
            df_plot['X'], df_plot['Y'], df_plot['Z'],
            c=df_plot['MPT'], cmap='plasma', s=5
        )
        ax.set_title('3D Process Path with MPT', fontsize=14)
        ax.set_xlabel('X-Coordinate')
        ax.set_ylabel('Y-Coordinate')
        ax.set_zlabel('Z-Coordinate')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label('MPT')
        fig.tight_layout()
        return fig

    # -------------------- 파이비스타: 시뮬 준비 --------------------

    def prepare_pyvista_simulation(self, plotter):
        """
        고성능 시뮬레이션 준비:
        - 하나의 폴리라인 + GPU 튜브 렌더(라인 폭)로 누적 표현(알파 업데이트)
        - 헤드(sphere) 1개만 이동
        - 반환: dict(sim_df, points, path_poly, path_actor, base_rgba, rgba_current, head_actor)
        """
        plotter.clear()
        sim_df = self.df.copy()

        # 시간 파싱
        def _parse_time(s):
            for fmt in [
                '%m_%d_%H_%M_%S_%f',
                '%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S.%f'
            ]:
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            return pd.to_datetime(s, errors='coerce')

        if 'time' not in sim_df.columns or not {'X', 'Y', 'Z'}.issubset(sim_df.columns):
            return None

        sim_df['time'] = sim_df['time'].apply(_parse_time)
        sim_df.dropna(subset=['time'], inplace=True)
        if len(sim_df) < 2:
            return None

        sim_df['time_delta_ms'] = sim_df['time'].diff().dt.total_seconds().fillna(0) * 1000
        points = sim_df[['X', 'Y', 'Z']].to_numpy()

        # 전체 경로 폴리라인 (포인트 수 = 경로 길이)
        path_poly = pv.lines_from_points(points, close=False)

        # 컬러 소스(기본: MPT) → RGBA 만들기
        values = sim_df['MPT'].to_numpy() if 'MPT' in sim_df.columns else np.zeros(len(sim_df))
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        norm = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba = (cm.get_cmap('coolwarm')(norm) * 255).astype(np.uint8)  # (N,4)

        # LASER_ON이 있으면 꺼진 구간은 회색/반투명
        if 'LASER_ON' in sim_df.columns:
            onmask = self._coerce_bool(sim_df['LASER_ON']) == True
            off_rgba = np.array([160, 160, 160, 110], dtype=np.uint8)
            rgba[~onmask.to_numpy()] = off_rgba

        base_rgba = rgba.copy()      # 최종 목표 색(누적되며 보일 색)
        rgba[:, 3] = 0               # 시작 시 전부 안 보이게
        path_poly['RGBA'] = rgba     # 현재 프레임 RGBA

        # 경로 actor (GPU 라인 튜브 렌더)
        path_actor = plotter.add_mesh(
            path_poly,
            scalars='RGBA',
            rgba=True,
            render_lines_as_tubes=True,  # 지오메트리 증식 없이 두꺼운 라인
            line_width=6
        )

        # 가이드(선택) — 옅은 그레이
        try:
            guide = pv.Spline(points, len(points))
            plotter.add_mesh(guide, color='lightgray', line_width=1, opacity=0.25)
        except Exception:
            pass

        # 헤드(로봇 팁)
        head_actor = plotter.add_mesh(
            pv.Sphere(radius=0.8, center=points[0]), color='red', ambient=0.3
        )

        try:
            plotter.camera_position = 'iso'
        except Exception:
            pass
        plotter.add_axes()
        plotter.set_background('white')

        return {
            "sim_df": sim_df,
            "points": points,
            "path_poly": path_poly,
            "path_actor": path_actor,
            "base_rgba": base_rgba,       # 고정 팔레트(보여줄 색)
            "rgba_current": rgba.copy(),  # 매 프레임 갱신(알파만 올림)
            "head_actor": head_actor,
        }

    # -------------------- 파이비스타: 런타임 리컬러 --------------------

    def recolor_simulation(self, handles, color_by: str = 'MPT'):
        """시뮬레이션 색상 변수를 런타임에 변경(MPT/MPA/LOAD 등)."""
        sim_df = handles["sim_df"]
        if color_by not in sim_df.columns:
            return False

        vals = pd.to_numeric(sim_df[color_by], errors='coerce').fillna(0).to_numpy()
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        norm = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba = (cm.get_cmap('coolwarm')(norm) * 255).astype(np.uint8)

        if 'LASER_ON' in sim_df.columns:
            onmask = self._coerce_bool(sim_df['LASER_ON']) == True
            off_rgba = np.array([160, 160, 160, 110], dtype=np.uint8)
            rgba[~onmask.to_numpy()] = off_rgba

        # 누적 상태는 유지하고, 아직 안 보인 구간은 alpha=0 유지
        cur = handles["rgba_current"]
        seen = cur[:, 3] > 0
        rgba[~seen, 3] = 0

        handles["base_rgba"] = rgba.copy()
        handles["rgba_current"] = rgba.copy()
        handles["path_poly"]["RGBA"] = rgba
        return True

    # -------------------- 파이비스타 + Matplotlib: 통합 대시보드 --------------------

    def prepare_integrated_dashboard(self, plotter, fig):
        """
        통합 대시보드용 초기 세팅.
        - 좌: PyVista (prepare_pyvista_simulation 재사용)
        - 우: Matplotlib (시간축 라인플롯 + 수직 커서 vlines)
        return: sim_handles(dict) or None
        """
        sim_handles = self.prepare_pyvista_simulation(plotter)
        if sim_handles is None:
            return None

        # 시간 정렬 + 인덱스 리셋
        sim_df = sim_handles["sim_df"].copy()
        sim_df.sort_values("time", inplace=True)
        sim_df.reset_index(drop=True, inplace=True)

        # 2D Figure 구성
        fig.clear()
        cols_candidates = {
            "MPT": ("MPT", "Temperature"),
            "R_LP": ("R_LP", "Laser Power"),
            "R_WS": ("R_WS", "Wire Feed Speed"),
            # 필요시: "LOAD": ("LOAD", "Load") 등 추가
        }
        present = [k for k in cols_candidates if k in sim_df.columns]
        if not present:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No numeric columns for 2D plots.", ha="center")
            ax.axis("off")
            # 그래도 3D는 사용 가능하게 sim_handles 반환
            sim_handles["sim_df"] = sim_df
            sim_handles["points"] = sim_df[['X', 'Y', 'Z']].to_numpy()
            sim_handles["mpl_fig"] = fig
            sim_handles["vlines"] = {}
            sim_handles["time_vec"] = sim_df["time"].to_numpy()
            return sim_handles

        n = min(3, len(present))
        axes = fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        vlines = {}
        t = sim_df["time"]  # pandas datetime Series

        for ax, key in zip(axes, present[:n]):
            label, ylabel = cols_candidates[key]
            y = pd.to_numeric(sim_df[key], errors="coerce")
            ax.plot(t, y, label=label)  # x축: time
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # 수직 커서(처음엔 시작 시각)
            v = ax.axvline(x=t.iloc[0], color="red", alpha=0.6)
            vlines[key] = v

            # 날짜 포맷터/로케이터
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        axes[-1].set_xlabel("Time")
        fig.autofmt_xdate()
        fig.tight_layout()

        # 핸들 업데이트 (업데이트 루프에서 사용)
        sim_handles["vlines"] = vlines
        sim_handles["mpl_fig"] = fig
        sim_handles["time_vec"] = t.to_numpy()  # numpy datetime64 배열

        # (중요) 시뮬레이션 쪽도 정렬된 sim_df로 교체
        sim_handles["sim_df"] = sim_df
        sim_handles["points"] = sim_df[['X', 'Y', 'Z']].to_numpy()

        return sim_handles


class Cursor:
    """
    마우스 좌표 가이드(수평/수직 라인 + 텍스트).
    - 디바운싱으로 과도한 redraw 방지
    """
    def __init__(self, ax, min_interval_sec: float = 1/60):
        self.ax = ax
        self.lx = ax.axhline(color='gray', linewidth=0.5, linestyle='--')
        self.ly = ax.axvline(color='gray', linewidth=0.5, linestyle='--')
        self.txt = ax.text(
            0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', pad=2)
        )
        self.lx.set_visible(False)
        self.ly.set_visible(False)
        self.txt.set_visible(False)
        self._last_draw = 0.0
        self._min_interval = float(min_interval_sec)

    def mouse_move(self, event):
        # 디바운싱
        now = time.perf_counter()
        if now - self._last_draw < self._min_interval:
            return

        is_visible = self.lx.get_visible()
        if not event.inaxes or event.inaxes != self.ax:
            if is_visible:
                self.lx.set_visible(False)
                self.ly.set_visible(False)
                self.txt.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            return

        if not is_visible:
            self.lx.set_visible(True)
            self.ly.set_visible(True)
            self.txt.set_visible(True)

        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            self.lx.set_ydata(y)
            self.ly.set_xdata(x)
            self.txt.set_text(f'x={x:.1f}, y={y:.2f}')
            self.ax.figure.canvas.draw_idle()
            self._last_draw = now
