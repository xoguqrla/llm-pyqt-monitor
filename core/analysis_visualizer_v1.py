# # core/analysis_visualizer.py

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import pyvista as pv

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False


# class AnalysisVisualizer:
#     def __init__(self, df: pd.DataFrame):
#         if not isinstance(df, pd.DataFrame):
#             raise ValueError("Input must be a pandas DataFrame.")
#         self.df = df
#         self.cursor_connections = []

#     def _clear_connections(self, fig):
#         for cid in self.cursor_connections:
#             try:
#                 fig.canvas.mpl_disconnect(cid)
#             except Exception:
#                 pass
#         self.cursor_connections.clear()

#     def plot_stability(self, fig):
#         self._clear_connections(fig)
#         fig.clear()
#         axes = fig.subplots(nrows=3, ncols=1)
#         fig.suptitle('Process Stability Analysis', fontsize=16)
#         axes[0].plot(self.df.index, self.df['S_LP'], label='Set', linestyle='--')
#         axes[0].plot(self.df.index, self.df['R_LP'], label='Actual', alpha=0.8)
#         axes[0].set_title('Laser Power (LP)'); axes[0].set_ylabel('Power (W)')
#         axes[0].legend(); axes[0].grid(True)
#         axes[1].plot(self.df.index, self.df['S_WS'], label='Set', linestyle='--')
#         axes[1].plot(self.df.index, self.df['R_WS'], label='Actual', alpha=0.8)
#         axes[1].set_title('Wire Feed Speed (WS)'); axes[1].set_ylabel('Speed')
#         axes[1].legend(); axes[1].grid(True)
#         axes[2].plot(self.df.index, self.df['MPT'], label='MPT', color='crimson')
#         axes[2].set_title('Melt Pool Temperature (MPT)'); axes[2].set_xlabel('Data Index (Time Flow)')
#         axes[2].set_ylabel('Temperature'); axes[2].legend(); axes[2].grid(True)
#         for ax in axes:
#             cursor = Cursor(ax)
#             cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
#             self.cursor_connections.append(cid)
#         fig.tight_layout(rect=[0, 0, 1, 0.96])
#         return fig

#     def plot_correlation_dashboard(self, fig):
#         self._clear_connections(fig)
#         fig.clear()
#         df_on = self.df[self.df['LASER_ON'] == True].copy()
#         corr_cols = ['R_LP', 'R_RS', 'R_WS', 'MPT', 'MPA', 'MPW', 'LOAD', 'CONTACT']
#         valid_cols = [col for col in corr_cols if col in df_on.columns]
#         if not valid_cols or df_on.empty or len(df_on) < 2:
#             ax = fig.add_subplot(111)
#             ax.text(0.5, 0.5, "Not enough data for correlation analysis.", ha='center')
#             ax.axis('off')
#             return fig
            
#         corr_matrix = df_on[valid_cols].corr()
#         pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
#         pairs = pairs[pairs.index.get_level_values(0) != pairs.index.get_level_values(1)]
#         positive_corr = pairs[pairs > 0.3].head(5)
#         negative_corr = pairs[pairs < -0.3].sort_values(ascending=True).head(5)
#         gs = fig.add_gridspec(1, 2, width_ratios=[6, 4], wspace=0.3)
#         ax_left = fig.add_subplot(gs[0])
#         ax_right = fig.add_subplot(gs[1])
#         fig.suptitle('Correlation Analysis Dashboard', fontsize=16, weight='bold')
#         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#         sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
#                     linewidths=.5, ax=ax_left, annot_kws={"size": 9})
#         ax_left.set_title('Correlation Heatmap', fontsize=12)
#         ax_left.tick_params(axis='x', rotation=45); ax_left.tick_params(axis='y', rotation=0)
#         ax_right.axis('off')
#         self._draw_correlation_table(ax_right, "Top 5 Positive Correlations", positive_corr)
#         self._draw_correlation_table(ax_right, "Top 5 Negative Correlations", negative_corr, y_offset=0.45)
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         return fig

# # core/analysis_visualizer.py -> AnalysisVisualizer 클래스 내부

#     def _draw_correlation_table(self, ax, title, data, y_offset=0.95):
#         ax.text(0.5, y_offset, title, ha='center', va='bottom', fontsize=12, weight='bold')
#         if not data.empty:
#             # <<<< 핵심 수정: 변수 이름을 str()로 감싸서 강제로 문자열로 변환 >>>>
#             cell_text = [[f'{str(var1)} & {str(var2)}', f'{val:.3f}'] for (var1, var2), val in data.items()]
            
#             table = ax.table(cellText=cell_text, colLabels=['Variable Pair', 'Correlation'],
#                              cellLoc='center', loc='center', bbox=[0, y_offset - 0.45, 1, 0.4])
#             table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
#         else:
#             ax.text(0.5, y_offset - 0.2, "No significant correlation.", ha='center', va='center', fontsize=10)

#     def plot_3d_path(self, fig):
#         self._clear_connections(fig)
#         fig.clear()
#         ax = fig.add_subplot(111, projection='3d')
#         sc = ax.scatter(self.df['X'], self.df['Y'], self.df['Z'], c=self.df['MPT'], cmap='plasma', s=5)
#         ax.set_title('3D Process Path with MPT', fontsize=14)
#         ax.set_xlabel('X-Coordinate'); ax.set_ylabel('Y-Coordinate'); ax.set_zlabel('Z-Coordinate')
#         cbar = fig.colorbar(sc, ax=ax, shrink=0.7); cbar.set_label('MPT')
#         fig.tight_layout()
#         return fig
        
#     def prepare_pyvista_simulation(self, plotter):
#         """PyVista 기반 시뮬레이션을 위한 데이터와 Actor를 준비합니다."""
#         plotter.clear()

#         sim_df = self.df.copy()
#         sim_df['time_str'] = sim_df['time']
#         sim_df['time'] = pd.to_datetime(sim_df['time'], format='%m_%d_%H_%M_%S_%f', errors='coerce')
#         sim_df.dropna(subset=['time'], inplace=True)
#         sim_df['time_delta_ms'] = sim_df['time'].diff().dt.total_seconds().fillna(0) * 1000

#         if len(sim_df) < 2:
#             return None, {}

#         points = sim_df[['X', 'Y', 'Z']].values

#         # 전체 경로 가이드라인
#         guide_line = pv.PolyData(points)
#         plotter.add_mesh(guide_line, color='gray', style='wireframe', opacity=0.3, line_width=2)
        
#         # 공정 헤드
#         head_actor = plotter.add_mesh(pv.Sphere(radius=0.8, center=points[0]), color='red', ambient=0.3)
        
#         # MPT 값에 따른 컬러맵 설정
#         mpt_values = sim_df['MPT'].values
#         min_mpt, max_mpt = np.nanmin(mpt_values), np.nanmax(mpt_values)
#         if min_mpt == max_mpt:
#             min_mpt -= 1; max_mpt +=1

#         # 모든 비드 조각을 미리 생성하여 숨김
#         bead_actors = []
#         for i in range(len(points) - 1):
#             p1, p2 = points[i], points[i+1]
#             segment = pv.Line(p1, p2)
#             tube = segment.tube(radius=0.3)
#             if sim_df['LASER_ON'].iloc[i]:
#                 scalar_val = mpt_values[i]
#                 scalars_array = np.full(tube.n_points, scalar_val)
#                 actor = plotter.add_mesh(
#                     tube,
#                     scalars=scalars_array,
#                     cmap='coolwarm',
#                     clim=[min_mpt, max_mpt],
#                     ambient=0.3,
#                     smooth_shading=True
#                 )
#             else:
#                 actor = plotter.add_mesh(
#                     tube,
#                     color='gray',
#                     ambient=0.1,
#                     opacity=0.5,
#                     smooth_shading=True
#                 )
#             # 경로 축적: 처음부터 모두 보이게 (이력 남김)
#             actor.SetVisibility(True)
#             bead_actors.append(actor)
        
#         plotter.camera_position = 'iso'
#         plotter.enable_zoom_scaling()
#         plotter.add_axes()
#         plotter.add_scalar_bar(title='MPT (Temperature)', vertical=True)
        
#         animation_handles = {
#             "sim_df": sim_df,
#             "points": points,
#             "head_actor": head_actor,
#             "bead_actors": bead_actors,
#         }
        
#         return animation_handles

# class Cursor:
#     def __init__(self, ax):
#         self.ax = ax
#         self.lx = ax.axhline(color='gray', linewidth=0.5, linestyle='--')
#         self.ly = ax.axvline(color='gray', linewidth=0.5, linestyle='--')
#         self.txt = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
#                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', pad=2))
#         self.lx.set_visible(False); self.ly.set_visible(False); self.txt.set_visible(False)

#     def mouse_move(self, event):
#         is_visible = self.lx.get_visible()
#         if not event.inaxes or event.inaxes != self.ax:
#             if is_visible:
#                 self.lx.set_visible(False); self.ly.set_visible(False); self.txt.set_visible(False)
#                 self.ax.figure.canvas.draw_idle()
#             return
#         if not is_visible:
#             self.lx.set_visible(True); self.ly.set_visible(True); self.txt.set_visible(True)
#         if event.xdata is not None and event.ydata is not None:
#             x, y = event.xdata, event.ydata
#             self.lx.set_ydata(y); self.ly.set_xdata(x)
#             self.txt.set_text(f'x={x:.1f}, y={y:.2f}')
#             self.ax.figure.canvas.draw_idle()
            


# core/analysis_visualizer.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class AnalysisVisualizer:
    """
    시각화/상관분석/3D 경로/파이비스타 시뮬레이션 준비 유틸.
    - 상관계수: 피어슨(명시), 숫자화/불리언 일관처리, LASER_ON True 필터 반영
    - 안정성 플롯: 누락 컬럼 방어
    - 3D 경로: 대용량 자동 샘플링(성능 보호)
    - PyVista: 일관된 반환(dict), 시간 파싱 보강, spline 가이드 라인
    """

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
            return s.astype("Int64").map(lambda x: True if x == 1 else (False if x == 0 else pd.NA))
        return s.astype(str).str.strip().str.lower().map({
            "true": True, "1": True, "y": True, "yes": True, "t": True,
            "false": False, "0": False, "n": False, "no": False, "f": False
        })

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

        # 피어슨 상관 명시 및 최소 표본수 보장
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
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=.5, ax=ax_left, annot_kws={"size": 9})
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
            table = ax.table(cellText=cell_text, colLabels=['Variable Pair', 'Correlation'],
                             cellLoc='center', loc='center', bbox=[0, y_offset - 0.45, 1, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
        else:
            ax.text(0.5, y_offset - 0.2, "No significant correlation.", ha='center', va='center', fontsize=10)

    # -------------------- 시각화: 3D 경로 --------------------

    def plot_3d_path(self, fig):
        """
        3D 산점도 경로(MPT 컬러). 매우 큰 데이터는 자동 샘플링으로 부담 완화.
        """
        self._clear_connections(fig)
        fig.clear()

        df_plot = self.df
        n = len(df_plot)
        # 선택 개선: 대용량 자동 샘플링(예: 50k 초과 시 20k로 다운샘플)
        if n > 50000:
            df_plot = df_plot.sample(n=20000, random_state=42).sort_index()

        ax = fig.add_subplot(111, projection='3d')
        if not set(['X', 'Y', 'Z', 'MPT']).issubset(df_plot.columns):
            ax.text(0.5, 0.5, "X/Y/Z/MPT columns required.", ha='center')
            ax.axis('off')
            fig.tight_layout()
            return fig

        sc = ax.scatter(df_plot['X'], df_plot['Y'], df_plot['Z'],
                        c=df_plot['MPT'], cmap='plasma', s=5)
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
        PyVista 기반 시뮬레이션 준비.
        - 시간 파싱을 다중 포맷으로 시도
        - spline 가이드 라인
        - 일관된 반환(dict)
        """
        plotter.clear()

        sim_df = self.df.copy()

        # 다양한 시간 포맷 지원
        def _parse_time(s):
            for fmt in ['%m_%d_%H_%M_%S_%f', '%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            return pd.to_datetime(s, errors='coerce')

        if 'time' not in sim_df.columns:
            return {"sim_df": sim_df, "points": np.empty((0, 3))}

        sim_df['time'] = sim_df['time'].apply(_parse_time)
        sim_df.dropna(subset=['time'], inplace=True)

        if len(sim_df) < 2 or not set(['X', 'Y', 'Z']).issubset(sim_df.columns):
            return {"sim_df": sim_df, "points": np.empty((0, 3))}

        # 시간 간격(ms)
        sim_df['time_delta_ms'] = sim_df['time'].diff().dt.total_seconds().fillna(0) * 1000

        points = sim_df[['X', 'Y', 'Z']].to_numpy()

        # 가이드 라인: Spline
        try:
            guide = pv.Spline(points, len(points))
            plotter.add_mesh(guide, color='gray', line_width=2, opacity=0.3)
        except Exception:
            # Spline 실패 시, 라인 분절로 대체
            for i in range(len(points) - 1):
                plotter.add_mesh(pv.Line(points[i], points[i + 1]), color='gray', opacity=0.3)

        # 공정 헤드
        head_actor = plotter.add_mesh(pv.Sphere(radius=0.8, center=points[0]), color='red', ambient=0.3)

        # 컬러 맵핑 위한 MPT 범위
        if 'MPT' in sim_df.columns:
            mpt_values = sim_df['MPT'].to_numpy()
        else:
            mpt_values = np.zeros(len(sim_df))
        vmin, vmax = np.nanmin(mpt_values), np.nanmax(mpt_values)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        # LASER_ON 여부
        if 'LASER_ON' in sim_df.columns:
            laser_on = self._coerce_bool(sim_df['LASER_ON'])
        else:
            laser_on = pd.Series([True] * len(sim_df), index=sim_df.index)

        # 비드 segment 생성
        bead_actors = []
        for i in range(len(points) - 1):
            seg = pv.Line(points[i], points[i + 1]).tube(radius=0.3)
            if laser_on.iloc[i] is True:
                scalars = np.full(seg.n_points, mpt_values[i])
                actor = plotter.add_mesh(
                    seg,
                    scalars=scalars,
                    cmap='coolwarm',
                    clim=[vmin, vmax],
                    ambient=0.3,
                    smooth_shading=True
                )
            else:
                actor = plotter.add_mesh(
                    seg,
                    color='gray',
                    ambient=0.1,
                    opacity=0.5,
                    smooth_shading=True
                )
            actor.SetVisibility(True)
            bead_actors.append(actor)

        # 보기/바
        try:
            plotter.camera_position = 'iso'
        except Exception:
            pass
        try:
            plotter.enable_zoom_scaling()
        except Exception:
            pass
        plotter.add_axes()
        plotter.add_scalar_bar(title='MPT (Temperature)', vertical=True)

        return {
            "sim_df": sim_df,
            "points": points,
            "head_actor": head_actor,
            "bead_actors": bead_actors,
        }


class Cursor:
    """
    마우스 좌표 가이드(수평/수직 라인 + 텍스트).
    - 디바운싱으로 과도한 redraw 방지
    """
    def __init__(self, ax, min_interval_sec: float = 1/60):
        self.ax = ax
        self.lx = ax.axhline(color='gray', linewidth=0.5, linestyle='--')
        self.ly = ax.axvline(color='gray', linewidth=0.5, linestyle='--')
        self.txt = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', pad=2))
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
