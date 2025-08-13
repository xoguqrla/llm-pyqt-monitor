# core/analysis_visualizer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyvista as pv

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class AnalysisVisualizer:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df
        self.cursor_connections = []

    def _clear_connections(self, fig):
        for cid in self.cursor_connections:
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self.cursor_connections.clear()

    def plot_stability(self, fig):
        # ... (This function is correct, no changes needed)
        self._clear_connections(fig)
        fig.clear()
        axes = fig.subplots(nrows=3, ncols=1)
        fig.suptitle('Process Stability Analysis', fontsize=16)
        axes[0].plot(self.df.index, self.df['S_LP'], label='Set', linestyle='--')
        axes[0].plot(self.df.index, self.df['R_LP'], label='Actual', alpha=0.8)
        axes[0].set_title('Laser Power (LP)'); axes[0].set_ylabel('Power (W)')
        axes[0].legend(); axes[0].grid(True)
        axes[1].plot(self.df.index, self.df['S_WS'], label='Set', linestyle='--')
        axes[1].plot(self.df.index, self.df['R_WS'], label='Actual', alpha=0.8)
        axes[1].set_title('Wire Feed Speed (WS)'); axes[1].set_ylabel('Speed')
        axes[1].legend(); axes[1].grid(True)
        axes[2].plot(self.df.index, self.df['MPT'], label='MPT', color='crimson')
        axes[2].set_title('Melt Pool Temperature (MPT)'); axes[2].set_xlabel('Data Index (Time Flow)')
        axes[2].set_ylabel('Temperature'); axes[2].legend(); axes[2].grid(True)
        for ax in axes:
            cursor = Cursor(ax)
            cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
            self.cursor_connections.append(cid)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_correlation_dashboard(self, fig):
        # ... (This function is correct, no changes needed)
        self._clear_connections(fig)
        fig.clear()
        df_on = self.df[self.df['LASER_ON'] == True].copy()
        corr_cols = ['R_LP', 'R_RS', 'R_WS', 'MPT', 'MPA', 'MPW', 'LOAD', 'CONTACT']
        valid_cols = [col for col in corr_cols if col in df_on.columns]
        corr_matrix = df_on[valid_cols].corr()
        pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
        pairs = pairs[pairs.index.get_level_values(0) != pairs.index.get_level_values(1)]
        positive_corr = pairs[pairs > 0.3].head(5)
        negative_corr = pairs[pairs < -0.3].sort_values(ascending=True).head(5)
        gs = fig.add_gridspec(1, 2, width_ratios=[6, 4], wspace=0.3)
        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1])
        fig.suptitle('Correlation Analysis Dashboard', fontsize=16, weight='bold')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=.5, ax=ax_left, annot_kws={"size": 9})
        ax_left.set_title('Correlation Heatmap', fontsize=12)
        ax_left.tick_params(axis='x', rotation=45); ax_left.tick_params(axis='y', rotation=0)
        ax_right.axis('off')
        self._draw_correlation_table(ax_right, "Top 5 Positive Correlations", positive_corr)
        self._draw_correlation_table(ax_right, "Top 5 Negative Correlations", negative_corr, y_offset=0.45)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _draw_correlation_table(self, ax, title, data, y_offset=0.95):
        # ... (This function is correct, no changes needed)
        ax.text(0.5, y_offset, title, ha='center', va='bottom', fontsize=12, weight='bold')
        if not data.empty:
            cell_text = [[f'{var1} & {var2}', f'{val:.3f}'] for (var1, var2), val in data.items()]
            table = ax.table(cellText=cell_text, colLabels=['Variable Pair', 'Correlation'],
                             cellLoc='center', loc='center', bbox=[0, y_offset - 0.45, 1, 0.4])
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
        else:
            ax.text(0.5, y_offset - 0.2, "No significant correlation.", ha='center', va='center', fontsize=10)

    def plot_3d_path(self, fig):
        # ... (This function is correct, no changes needed)
        self._clear_connections(fig)
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.df['X'], self.df['Y'], self.df['Z'], c=self.df['MPT'], cmap='plasma', s=5)
        ax.set_title('3D Process Path with MPT', fontsize=14)
        ax.set_xlabel('X-Coordinate'); ax.set_ylabel('Y-Coordinate'); ax.set_zlabel('Z-Coordinate')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7); cbar.set_label('MPT')
        fig.tight_layout()
        return fig
        
    def prepare_pyvista_simulation(self, plotter):
        """PyVista-based simulation preparation."""
        plotter.clear()

        sim_df = self.df[self.df['LASER_ON'] == True].copy()
        sim_df['time_str'] = sim_df['time']
        
        # ## CORE FIX ##
        # Re-add the format string to tell pandas exactly how to read the time.
        # This is faster and removes the warning.
        sim_df['time'] = pd.to_datetime(sim_df['time'], format='%m_%d_%H_%M_%S_%f', errors='coerce')
        
        sim_df.dropna(subset=['time'], inplace=True)
        sim_df['time_delta_ms'] = sim_df['time'].diff().dt.total_seconds().fillna(0) * 1000

        if sim_df.empty or len(sim_df) < 2:
            return None, None, None, None

        points = sim_df[['X', 'Y', 'Z']].values

        guide_line = pv.PolyData(points)
        plotter.add_mesh(guide_line, color='gray', style='wireframe', opacity=0.3, line_width=2)
        
        head_actor = plotter.add_mesh(pv.Sphere(radius=0.8, center=points[0]), color='#ff4757', ambient=0.3)
        
        initial_line = pv.PolyData(points[:2])
        bead_actor = plotter.add_mesh(initial_line.tube(radius=0.3), color='#4a69bd', ambient=0.3, smooth_shading=True)
        
        plotter.camera_position = 'iso'
        plotter.enable_zoom_scaling()
        plotter.add_axes()
        
        return sim_df, points, head_actor, bead_actor

class Cursor:
    # ... (This class is correct, no changes needed)
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='gray', linewidth=0.5, linestyle='--')
        self.ly = ax.axvline(color='gray', linewidth=0.5, linestyle='--')
        self.txt = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', pad=2))
        self.lx.set_visible(False); self.ly.set_visible(False); self.txt.set_visible(False)

    def mouse_move(self, event):
        is_visible = self.lx.get_visible()
        if not event.inaxes or event.inaxes != self.ax:
            if is_visible:
                self.lx.set_visible(False); self.ly.set_visible(False); self.txt.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            return
        if not is_visible:
            self.lx.set_visible(True); self.ly.set_visible(True); self.txt.set_visible(True)
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            self.lx.set_ydata(y); self.ly.set_xdata(x)
            self.txt.set_text(f'x={x:.1f}, y={y:.2f}')
            self.ax.figure.canvas.draw_idle()