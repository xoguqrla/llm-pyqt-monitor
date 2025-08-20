# app/mpt_3d_sim_mcbead.py
import argparse, math
import numpy as np
import pandas as pd
import pyvista as pv


def _parse_time(series: pd.Series, time_format: str | None, assume_year: int | None):
    """Parse time column to pandas datetime, return a Series."""
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
    """Coerce a column with mixed types into boolean True/False."""
    if col.dtype == bool:
        return col
    s = col.astype(str).str.strip().str.lower()
    true_like = {"true", "1", "t", "y", "yes"}
    false_like = {"false", "0", "f", "n", "no", ""}
    out = pd.Series(False, index=col.index)
    out[s.isin(true_like)] = True
    out[s.isin(false_like)] = False
    return out


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
    time_col: str = "time",
    r_rs_col: str = "R_RS",
    r_ws_col: str = "R_WS",
    width_col: str = "MPW",
    laser_col: str = "LASER_ON",
    time_format: str | None = None,
    assume_year: int | None = None,
    # 단위/스케일 보정
    rs_scale: float = 1.0,    # R_RS에 곱할 스케일 (예: mm/min → mm/s 이면 1/60)
    ws_scale: float = 1.0,    # R_WS에 곱할 스케일
    width_scale: float = 1.0, # MPW 단위 변환 (예: μm → mm 이면 1/1000)
    # 이상치 캡핑
    cap_q: float = 0.995,     # r_eq, height 분위수 캡핑 (0<cap_q<1, None이면 비활성)
    max_radius_mm: float | None = None,  # r_eq 절대 상한
    max_height_mm: float | None = None,  # Bead_Height 절대 상한
    min_rs_eps: float = 1e-3, # 분모 안전 하한(mm/s)
    efficiency: float = 1.0,  # 적층 효율(0~1). 1보다 작으면 더 얇아짐.
) -> pd.DataFrame:
    """Compute mass-conserving bead attributes for each segment with scaling/capping."""
    df = df.copy()

    # 1) time → Δt
    t = _parse_time(df[time_col], time_format, assume_year)
    dt_sec = t.diff().dt.total_seconds().fillna(0).clip(lower=0)
    df["delta_t"] = dt_sec

    # 2) 속도/폭 스케일 보정
    r_rs_raw = pd.to_numeric(df[r_rs_col], errors="coerce").fillna(0)
    r_ws_raw = pd.to_numeric(df[r_ws_col], errors="coerce").fillna(0)
    width_raw = pd.to_numeric(df.get(width_col, pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0)

    r_rs = (r_rs_raw * rs_scale).astype(float)
    r_ws = (r_ws_raw * ws_scale).astype(float)
    width = (width_raw * width_scale).astype(float)

    # 3) 세그먼트 길이 L = R_RS * Δt
    L = (r_rs * dt_sec).clip(lower=0)
    df["seg_length"] = L

    # 4) 적층 마스크(느슨하게): 와이어가 공급되고 레이저가 ON이면 적층으로 간주
    laser_val = df.get(laser_col, pd.Series(True, index=df.index))
    laser_bool = _coerce_bool_col(laser_val)
    mask = (r_ws > 0) & (laser_bool)
    df["deposits"] = mask

    # 5) A_bead by mass conservation (효율, 분모 안전장치)
    #    A_bead = η * [π * (D_wire/2)^2 * R_WS] / max(R_RS, min_rs_eps)
    radius_wire = d_wire_mm / 2.0
    A_wire = math.pi * radius_wire * radius_wire  # mm^2
    r_rs_safe = np.maximum(r_rs, min_rs_eps)
    A_bead = efficiency * (A_wire * r_ws) / r_rs_safe
    A_bead = pd.Series(A_bead, index=df.index).fillna(0).where(mask, 0)
    df["A_bead"] = A_bead

    # 6) width → height
    eps_w = 1e-6
    width = pd.Series(width, index=df.index).clip(lower=eps_w)
    height = (A_bead / width).fillna(0).where(mask, 0)

    # 7) r_eq (equivalent radius)
    r_eq = pd.Series(np.sqrt(A_bead / math.pi), index=df.index).where(mask, 0)

    # 8) 분위수/절대 상한 캡핑
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

    df["Bead_Width"]  = width.where(mask, 0)
    df["Bead_Height"] = height.where(mask, 0)
    df["r_eq"]        = r_eq.where(mask, 0)

    # 9) 진단용 요약
    print(">> Diagnostics (after scaling, before render)")
    print("rows=", len(df),
          "mask_true=", int(mask.sum()),
          "ws>0=", int((r_ws > 0).sum()),
          "laser_true=", int(laser_bool.sum()))
    print("R_RS(mm/s):", _quantiles(r_rs, mask))
    print("R_WS(mm/s):", _quantiles(r_ws, mask))
    print("MPW(mm):   ", _quantiles(width, mask))
    print("A_bead(mm^2):", _quantiles(A_bead, mask))
    print("r_eq(mm):    ", _quantiles(r_eq, mask))
    print("H(mm):       ", _quantiles(height, mask))

    return df


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window=win, center=True, min_periods=1).median().to_numpy()


def render_mc_bead(
    df: pd.DataFrame,
    mode: str = "tube",   # 'tube' or 'rect'
    color_by: str = "MPT",
    sample_step: int = 1,
    # tube 전용
    tube_radius_mode: str = "area",  # 'area' | 'width' | 'min'
    tube_gain: float = 1.0,
    smooth_win: int = 1,
    max_radius_mm: float | None = None,  # 최종 반지름 상한(튜브에 재적용)
):
    """Render mass-conserving beads along the toolpath using PyVista."""
    pts = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    mpt = pd.to_numeric(df.get(color_by, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy()
    r_eq = df["r_eq"].to_numpy()
    W = df["Bead_Width"].to_numpy()
    H = df["Bead_Height"].to_numpy()
    mask = df["deposits"].to_numpy(dtype=bool)

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.set_background("white")

    # base path(경로 스플라인)
    step = max(sample_step, 1)
    npts = max(200, len(pts) // step) if len(pts) > 1 else 2
    spline = pv.Spline(pts[::step], n_points=npts)
    plotter.add_mesh(spline, color="black", line_width=1, opacity=0.15)

    if mode == "tube":
        # --- 튜브 반지름 결정 ---
        r_area = r_eq
        r_width = W * 0.5
        if tube_radius_mode == "area":
            r_tube = r_area
        elif tube_radius_mode == "width":
            r_tube = r_width
        elif tube_radius_mode == "min":
            r_tube = np.minimum(r_area, r_width)
        else:
            raise ValueError("tube_radius_mode must be 'area', 'width', or 'min'")

        # 스무딩 & 게인 & 상한
        r_tube = _rolling_median(r_tube, smooth_win)
        r_tube = np.maximum(0.0, r_tube * float(tube_gain))
        if max_radius_mm is not None:
            r_tube = np.minimum(r_tube, float(max_radius_mm))

        geoms = []
        for i in range(0, len(pts) - 1, step):
            if not mask[i]:
                continue
            p0, p1 = pts[i], pts[i + 1]
            dir_vec = p1 - p0
            seg_len = np.linalg.norm(dir_vec)
            if seg_len <= 0 or r_tube[i] <= 0:
                continue
            center = (p0 + p1) * 0.5
            cyl = pv.Cylinder(center=center, direction=_unit(dir_vec),
                              radius=float(r_tube[i]), height=float(seg_len))
            val = float(mpt[i]) if not np.isnan(mpt[i]) else 0.0
            cyl["color_scalar"] = np.full(cyl.n_points, val)
            geoms.append(cyl)

        if geoms:
            mb = pv.MultiBlock(geoms)
            mesh = mb.combine()
            plotter.add_mesh(mesh, scalars="color_scalar", cmap="plasma", smooth_shading=True)

    elif mode == "rect":
        blocks = []
        for i in range(0, len(pts) - 1, step):
            if not mask[i]:
                continue
            p0, p1 = pts[i], pts[i + 1]
            dir_vec = p1 - p0
            seg_len = np.linalg.norm(dir_vec)
            if seg_len <= 0 or W[i] <= 0 or H[i] <= 0:
                continue

            cube = pv.Cube(center=(0.0, 0.0, 0.0),
                           x_length=float(W[i]), y_length=float(H[i]), z_length=float(seg_len))
            z_axis = np.array([0.0, 0.0, 1.0])
            v = _unit(dir_vec)
            axis = np.cross(z_axis, v); axis_norm = np.linalg.norm(axis)
            c = float(np.dot(z_axis, v)); angle = 0.0
            if axis_norm > 1e-12:
                c = max(-1.0, min(1.0, c))
                angle = math.degrees(math.acos(c))
            if angle != 0.0 and axis_norm > 1e-12:
                cube = cube.rotate_vector(axis=axis/axis_norm, angle=angle, point=(0.0, 0.0, 0.0), inplace=False)
            cube = cube.translate((p0 + p1) * 0.5, inplace=False)

            val = float(mpt[i]) if not np.isnan(mpt[i]) else 0.0
            cube["color_scalar"] = np.full(cube.n_points, val)
            blocks.append(cube)

        if blocks:
            mb = pv.MultiBlock(blocks)
            union = mb.combine()
            plotter.add_mesh(union, scalars="color_scalar", cmap="plasma", smooth_shading=True)

    else:
        raise ValueError("mode must be 'tube' or 'rect'")

    plotter.camera.zoom(1.4)
    plotter.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--wire-d", type=float, required=True, help="Wire diameter in mm (e.g., 1.2)")
    ap.add_argument("--mode", choices=["tube", "rect"], default="tube")
    ap.add_argument("--time-format", type=str, default=None)
    ap.add_argument("--assume-year", type=int, default=None)
    ap.add_argument("--sample-step", type=int, default=1)
    # 스케일/캡핑/효율
    ap.add_argument("--rs-scale", type=float, default=1.0, help="Scale for R_RS (e.g., 1/60 if mm/min → mm/s)")
    ap.add_argument("--ws-scale", type=float, default=1.0, help="Scale for R_WS (e.g., 1/60 if mm/min → mm/s)")
    ap.add_argument("--width-scale", type=float, default=1.0, help="Scale for MPW (e.g., 1/1000 if μm → mm)")
    ap.add_argument("--cap-q", type=float, default=0.995, help="Quantile cap for r_eq & height (0<q<1, None to disable)")
    ap.add_argument("--max-radius-mm", type=float, default=None)
    ap.add_argument("--max-height-mm", type=float, default=None)
    ap.add_argument("--min-rs-eps", type=float, default=1e-3)
    ap.add_argument("--efficiency", type=float, default=1.0)
    # 튜브 옵션
    ap.add_argument("--tube-radius-mode", choices=["area", "width", "min"], default="area")
    ap.add_argument("--tube-gain", type=float, default=1.0)
    ap.add_argument("--smooth-win", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = compute_mc_bead(
        df,
        d_wire_mm=args.wire_d,
        time_col="time",
        r_rs_col="R_RS",
        r_ws_col="R_WS",
        width_col="MPW",
        laser_col="LASER_ON",
        time_format=args.time_format,
        assume_year=args.assume_year,
        rs_scale=args.rs_scale,
        ws_scale=args.ws_scale,
        width_scale=args.width_scale,
        cap_q=args.cap_q,
        max_radius_mm=args.max_radius_mm,
        max_height_mm=args.max_height_mm,
        min_rs_eps=args.min_rs_eps,
        efficiency=args.efficiency,
    )

    print("Mean A_bead (mm^2):", df["A_bead"].replace(0, np.nan).mean())
    print("Mean width/height (mm):",
          df["Bead_Width"].replace(0, np.nan).mean(),
          df["Bead_Height"].replace(0, np.nan).mean())

    render_mc_bead(
        df, mode=args.mode, color_by="MPT", sample_step=args.sample_step,
        tube_radius_mode=args.tube_radius_mode, tube_gain=args.tube_gain,
        smooth_win=args.smooth_win, max_radius_mm=args.max_radius_mm
    )


if __name__ == "__main__":
    main()
