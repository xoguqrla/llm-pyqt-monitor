import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def find_col(df, keys):
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None


def load_and_prepare(csv, time=None, x=None, y=None, z=None, mpt=None,
                     time_format=None, assume_year=None,
                     order="time", layer_by="auto", z_bin=None):
    df = pd.read_csv(csv)

    time_col = time or find_col(df, ['time', 'timestamp', 'ts', 'date'])
    x_col    = x    or find_col(df, ['x'])
    y_col    = y    or find_col(df, ['y'])
    z_col    = z    or find_col(df, ['z'])
    mpt_col  = mpt  or find_col(df, ['mpt'])
    missing = [n for n,c in [('time',time_col),('X',x_col),('Y',y_col),('Z',z_col),('MPT',mpt_col)] if c is None]
    if missing:
        raise SystemExit(f"필수 컬럼 미발견: {missing}\n열 목록: {list(df.columns)}")

    # ---- time parse (format + assume_year 지원) ----
    try:
        s = df[time_col].astype(str)
        if time_format and assume_year and "%Y" not in time_format:
            s = s.map(lambda v: f"{assume_year}-{v}")
            time_fmt = "%Y-" + time_format
        else:
            time_fmt = time_format
        t = pd.to_datetime(s, format=time_fmt, errors="coerce") if time_fmt else pd.to_datetime(s, errors="coerce")
        df["_t"] = t if t.notna().sum() > 0.2*len(s) else df[time_col]
    except Exception:
        df["_t"] = df[time_col]

    # ---- numeric & clean ----
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=[x_col,y_col,z_col,mpt_col])
    for c in [x_col,y_col,z_col,mpt_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[x_col,y_col,z_col,mpt_col]).copy()

    # ---- ordering ----
    order = (order or "time").lower()
    if order == "z":
        df = df.sort_values([z_col]).reset_index(drop=True)
    elif order in ("ztime","z-time","zt"):
        df = df.sort_values([z_col, "_t"]).reset_index(drop=True)
    else:  # "time"
        df = df.sort_values(["_t"]).reset_index(drop=True)

    # ---- layer grouping ----
    # auto: BEAD_NUMBER / LAYER / PASS 같은 컬럼 있으면 그것으로, 없으면 z-bin 으로
    layer_key = None
    cand = find_col(df, ['bead', 'layer', 'pass'])
    if layer_by == "auto":
        if cand is not None:
            layer_key = cand
        else:
            # bin by z
            if z_bin is None:
                # 자동 bin 폭: Z range를 50등분 기준
                zmin, zmax = float(df[z_col].min()), float(df[z_col].max())
                z_bin = max(1e-9, (zmax - zmin)/50.0)
            df["_layer_bin"] = (df[z_col] / z_bin).round().astype(int)
            layer_key = "_layer_bin"
    elif layer_by == "zbin":
        if z_bin is None:
            zmin, zmax = float(df[z_col].min()), float(df[z_col].max())
            z_bin = max(1e-9, (zmax - zmin)/50.0)
        df["_layer_bin"] = (df[z_col] / z_bin).round().astype(int)
        layer_key = "_layer_bin"
    else:
        # 특정 컬럼명 지정됨
        if layer_by not in df.columns:
            raise SystemExit(f"layer_by='{layer_by}' 컬럼을 찾을 수 없습니다.")
        layer_key = layer_by

    return df, time_col, x_col, y_col, z_col, mpt_col, layer_key


def pad_limits(a, b, p=0.05):
    span = (b-a) if (b-a)!=0 else 1.0
    return a - span*p, b + span*p


def build_animation(csv, time=None, x=None, y=None, z=None, mpt=None,
                    time_format=None, assume_year=None,
                    order="ztime", layer_by="auto", z_bin=None,
                    mode="layered", out="mpt_3d_sim.mp4", fps=15, interval=50,
                    max_frames=600, trail=400, fig_size=6.0,
                    size_min=20.0, size_max=200.0):
    """
    mode:
      - 'line'     : 시간순 단일 라인(지그재그 위험)
      - 'scatter'  : 점만(형상 확인용)
      - 'layered'  : 레이어별로 끊어서 라인(권장; 역 원뿔에 적합)
    order:
      - 'time' | 'z' | 'ztime'(권장)
    """

    df, time_col, x_col, y_col, z_col, mpt_col, layer_key = load_and_prepare(
        csv, time, x, y, z, mpt, time_format, assume_year, order, layer_by, z_bin
    )

    # 다운샘플: 프레임 수 제한
    N = len(df)
    idx_all = np.linspace(0, N-1, max_frames).astype(int) if N>max_frames else np.arange(N)
    df = df.iloc[idx_all].reset_index(drop=True)

    X = df[x_col].to_numpy()
    Y = df[y_col].to_numpy()
    Z = df[z_col].to_numpy()
    MPT = df[mpt_col].to_numpy()
    T = df["_t"].astype(str).to_numpy()

    mmin, mmax = float(MPT.min()), float(MPT.max())
    den = (mmax - mmin) if (mmax - mmin) != 0 else 1.0
    sizes = size_min + (size_max - size_min) * (MPT - mmin) / den

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*pad_limits(float(X.min()), float(X.max())))
    ax.set_ylim(*pad_limits(float(Y.min()), float(Y.max())))
    ax.set_zlim(*pad_limits(float(Z.min()), float(Z.max())))
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_zlabel(z_col)
    title = ax.set_title(f"MPT 3D Simulation ({mode}, {order})")

    # 준비물
    line, = ax.plot([], [], [], lw=1)
    sc = ax.scatter([], [], [], s=[])
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    trail = min(trail, len(df))

    # 레이어 인덱스 미리 계산
    layer_vals = df[layer_key].to_numpy() if layer_key else None

    def init():
        line.set_data([], []); line.set_3d_properties([])
        sc._offsets3d = ([], [], [])
        txt.set_text("")
        return line, sc, txt, title

    def update(i):
        # 현재 프레임까지의 부분 데이터
        xs, ys, zs = X[:i+1], Y[:i+1], Z[:i+1]

        if mode == "scatter":
            # 산점도만 (형상 위주)
            sc._offsets3d = (xs, ys, zs)
            sc.set_sizes(sizes[:i+1])
            line.set_data([], []); line.set_3d_properties([])

        elif mode == "layered" and layer_vals is not None:
            # 레이어별로 선을 끊어서 그림(지그재그 방지)
            ax.collections.clear()  # 지난 프레임 산점/라인 지우기(속도 관리)
            ax.lines = []           # 지난 라인 지우기

            # 현재까지 포함된 고유 레이어들
            layers = np.unique(layer_vals[:i+1])
            for lv in layers:
                mask = (layer_vals[:i+1] == lv)
                ax.plot(xs[mask], ys[mask], zs[mask], lw=1)  # 동일 레이어 내에서만 연결

            # 현재 지점 강조
            sc = ax.scatter([X[i]], [Y[i]], [Z[i]], s=[sizes[i]])
        else:
            # 단일 라인(지그재그 가능)
            s_idx = max(0, i - trail)
            line.set_data(xs[s_idx:], ys[s_idx:])
            line.set_3d_properties(zs[s_idx:])
            sc._offsets3d = ([X[i]], [Y[i]], [Z[i]])
            sc.set_sizes([sizes[i]])

        title.set_text(f"MPT 3D Simulation ({mode}, {order}) | {i+1}/{len(df)}")
        txt.set_text(f"time: {T[i]} | MPT: {MPT[i]:.4f}")
        return line, sc, txt, title

    anim = FuncAnimation(fig, update, init_func=init, frames=len(df), interval=50, blit=False)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=-1)
        anim.save(out, writer=writer)
        print(f"[ok] saved: {out}")
    except Exception:
        alt = out.rsplit('.',1)[0] + ".gif"
        anim.save(alt, writer=PillowWriter(fps=max(1, fps//2)))
        print(f"[info] ffmpeg 없음 → GIF로 저장: {alt}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time")
    ap.add_argument("--x")
    ap.add_argument("--y")
    ap.add_argument("--z")
    ap.add_argument("--mpt")
    ap.add_argument("--time-format", default="%m-%d %H:%M:%S.%f",
                    help="예: '%m-%d %H:%M:%S.%f' (월-일 시:분:초.밀리초)")
    ap.add_argument("--assume-year", type=int, default=2025, help="연도 없는 타임스탬프일 때 붙일 연도")
    ap.add_argument("--order", default="ztime", choices=["time","z","ztime"],
                    help="정렬 기준: time | z | ztime(Z→time)")
    ap.add_argument("--layer-by", default="auto",
                    help="'auto'(기본) | 'zbin' | 특정 컬럼명(예: BEAD_NUMBER)")
    ap.add_argument("--z-bin", type=float, default=None, help="layer-by=zbin 일 때 Z bin 폭")
    ap.add_argument("--mode", default="layered", choices=["layered","scatter","line"],
                    help="표현 방식: layered(권장) | scatter | line")
    ap.add_argument("--out", default="mpt_3d_sim.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--interval", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--trail", type=int, default=400)
    ap.add_argument("--fig", type=float, default=6.0)
    ap.add_argument("--size-min", type=float, default=20.0)
    ap.add_argument("--size-max", type=float, default=200.0)
    args = ap.parse_args()

    build_animation(
        csv=args.csv, time=args.time, x=args.x, y=args.y, z=args.z, mpt=args.mpt,
        time_format=args.time_format, assume_year=args.assume_year,
        order=args.order, layer_by=args.layer_by, z_bin=args.z_bin,
        mode=args.mode, out=args.out, fps=args.fps, interval=args.interval,
        max_frames=args.max_frames, trail=args.trail, fig_size=args.fig,
        size_min=args.size_min, size_max=args.size_max
    )
