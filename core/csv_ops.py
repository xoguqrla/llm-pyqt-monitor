# core/csv_ops.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd


# =========================
# Helper / Utility
# =========================

class CustomJSONEncoder(json.JSONEncoder):
    """NumPy 타입을 Python 기본 타입으로 변환하는 커스텀 JSON 인코더"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)


def standardize_columns(cols):
    """컬럼명을 표준 형식(소문자, snake_case)으로 변환"""
    return [re.sub(r'[^0-9a-zA-Z_]+', '_', str(c).strip().lower()).strip('_') for c in cols]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """날짜/시간 관련 컬럼을 datetime 객체로 자동 변환"""
    candidates = [c for c in df.columns
                  if any(k in c.lower() for k in ('time', 'date', 'datetime', 'timestamp', 'ts'))]
    for c in candidates:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        parsed = None
        # mm_dd_HH_MM_SS_ms 형식 먼저 시도
        try:
            cand = pd.to_datetime(s, format='%m_%d_%H_%M_%S_%f', errors='coerce')
            if cand.notna().mean() >= 0.8:
                parsed = cand
        except (TypeError, ValueError):
            pass

        if parsed is None and pd.api.types.is_object_dtype(s):
            try:
                cand = pd.to_datetime(s, format='mixed', errors='coerce', utc=False)
                if cand.notna().mean() >= 0.8:
                    parsed = cand
            except (TypeError, ValueError):
                pass

        if parsed is not None:
            df[c] = parsed
    return df


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """존재하지 않는 컬럼은 NaN 시리즈 반환, 숫자형으로 변환"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _mad_to_robust_std(s: pd.Series) -> float:
    """Median Absolute Deviation을 이용한 강건 표준편차 추정 (Gaussian 등가치)"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float('nan')
    med = s.median()
    mad = (s - med).abs().median()
    return float(1.4826 * mad)


def _linear_piecewise_ratio_to_score(r: float,
                                     anchors=(0.0, 0.10, 0.25, 0.50),
                                     scores=(100.0, 80.0, 50.0, 0.0)) -> float:
    """
    비율(오차 또는 비율 스케일)을 점수로 변환하는 간단한 구간 선형 보간.
    anchors 는 오름차순, scores 는 anchors에 대응.
    """
    r = max(0.0, float(r))
    # 오른쪽 무한대는 마지막 점수로 클립
    if r >= anchors[-1]:
        return float(scores[-1])
    # 왼쪽은 첫 점수
    if r <= anchors[0]:
        return float(scores[0])
    # 구간 탐색
    for i in range(1, len(anchors)):
        if anchors[i-1] <= r <= anchors[i]:
            x0, x1 = anchors[i-1], anchors[i]
            y0, y1 = scores[i-1], scores[i]
            # 선형 보간
            t = (r - x0) / (x1 - x0) if x1 != x0 else 0.0
            return float(y0 + t * (y1 - y0))
    return float(scores[-1])


# =========================
# Similarity / Stability
# =========================

def calculate_similarity_score(current_metrics: dict, golden_metrics: dict) -> dict:
    """
    두 공정의 주요 파라미터/결과물을 비교하여 유사도 점수를 계산.
    (모든 값이 존재할 필요는 없음. 빠진 값은 0점 처리.)
    """
    scores = {}
    # 가중치는 예시이며, 현장 피드백에 따라 조정 권장
    weights = {
        "utilization_ratio": 0.15,
        "energy_input_total": 0.15,
        "cum_volume_est": 0.20,
        "wire_feed_efficiency": 0.10,
        "mpt_mean": 0.20,
        "mpt_std": 0.20,
    }

    for key, weight in weights.items():
        cur = current_metrics.get(key)
        ref = golden_metrics.get(key)
        if cur is None or ref is None or ref == 0:
            score = 0.0
        else:
            # 오차율 기반 점수 (1 - |오차율|) * 100, 최소 0
            err_ratio = abs(cur - ref) / abs(ref)
            score = max(0.0, 1.0 - err_ratio) * 100.0
        scores[f"{key}_similarity"] = round(score, 1)

    # 가중 평균
    final = sum(scores[f"{k}_similarity"] * w for k, w in weights.items())
    scores["final_similarity_score"] = round(final, 1)
    return scores


def calculate_stability_scores(stats: dict,
                               golden_stats: Optional[dict] = None,
                               mode: str = "auto") -> dict:
    """
    안정성 점수 계산.
    - golden_stats 가 있으면 상대평가(비율 기반) 우선
    - golden_stats 가 없으면 완화된 절대평가 밴드 사용
    stats 예:
    {
        "mpt": { "mean":..., "std_dev":..., "median":..., "robust_std":... },
        "load": { "std_dev":... }
    }
    """
    # 기본 초기화
    temp_stability_score = 0.0
    temp_suitability_score = 0.0
    load_stability_score = 0.0

    # 상대평가(골든) 경로
    if golden_stats:
        gm = golden_stats.get("mpt", {}) if isinstance(golden_stats.get("mpt"), dict) else {}
        gl = golden_stats.get("load", {}) if isinstance(golden_stats.get("load"), dict) else {}

        # 현재
        mpt_med = stats.get("mpt", {}).get("median")
        mpt_rstd = stats.get("mpt", {}).get("robust_std")
        mpt_mean = stats.get("mpt", {}).get("mean")
        mpt_std = stats.get("mpt", {}).get("std_dev")
        load_std = stats.get("load", {}).get("std_dev")

        # 골든
        gm_med = gm.get("median")
        gm_rstd = gm.get("robust_std", gm.get("std_dev"))  # 골든에 robust_std 없으면 std_dev 사용
        gm_mean = gm.get("mean")
        gm_std = gm.get("std_dev")
        gl_std = gl.get("std_dev")

        # 1) MPT 안정성(변동성) : robust_std 비율(또는 std_dev 비율) → 작을수록 좋음
        if mpt_rstd is None or np.isnan(mpt_rstd):
            # robust 없으면 std 사용
            cur_var = mpt_std
            ref_var = gm_std
        else:
            cur_var = mpt_rstd
            ref_var = gm_rstd
        if cur_var is not None and ref_var and ref_var > 0:
            var_ratio = cur_var / ref_var
            # var_ratio=1 → 100점, 1.5 → ~80, 2.5 → ~50, 5.0 → 0 (예시)
            temp_stability_score = _linear_piecewise_ratio_to_score(var_ratio,
                                                                    anchors=(0.0, 1.0, 1.5, 2.5, 5.0),
                                                                    scores=(100, 100, 80, 50, 0))
        else:
            temp_stability_score = 0.0

        # 2) MPT 적정성(레시피 목표 근접성) : median(또는 mean) 오차율
        if mpt_med is None or np.isnan(mpt_med):
            ref_loc = gm_mean  # 골든 median이 없으면 mean 사용
            cur_loc = mpt_mean
        else:
            ref_loc = gm_med if gm_med is not None else gm_mean
            cur_loc = mpt_med
        if ref_loc not in (None, 0) and cur_loc is not None:
            mean_err_ratio = abs(cur_loc - ref_loc) / abs(ref_loc)
            # 0% → 100, 10% → 80, 25% → 50, 50% → 0
            temp_suitability_score = _linear_piecewise_ratio_to_score(mean_err_ratio,
                                                                      anchors=(0.0, 0.10, 0.25, 0.50),
                                                                      scores=(100, 80, 50, 0))
        else:
            temp_suitability_score = 0.0

        # 3) LOAD 안정성: std_dev 비율
        if load_std is not None and gl_std and gl_std > 0:
            load_ratio = load_std / gl_std
            load_stability_score = _linear_piecewise_ratio_to_score(load_ratio,
                                                                    anchors=(0.0, 1.0, 1.5, 2.5, 5.0),
                                                                    scores=(100, 100, 80, 50, 0))
        else:
            load_stability_score = 0.0

    else:
        # 절대평가(완화밴드)
        # 현재
        mpt_mean = stats.get("mpt", {}).get("mean")
        mpt_std = stats.get("mpt", {}).get("std_dev")
        mpt_med = stats.get("mpt", {}).get("median")
        mpt_rstd = stats.get("mpt", {}).get("robust_std")
        load_std = stats.get("load", {}).get("std_dev")

        # 변동성: robust_std(없으면 std) 기준으로 완화 밴드
        var = mpt_rstd if (mpt_rstd is not None and not np.isnan(mpt_rstd)) else mpt_std
        if var is not None and not np.isnan(var):
            # var 5 → 100, 20 → 80, 60 → 50, 150 → 0 (예시)
            temp_stability_score = _linear_piecewise_ratio_to_score(var,
                                                                    anchors=(0.0, 5.0, 20.0, 60.0, 150.0),
                                                                    scores=(100, 100, 80, 50, 0))
        else:
            temp_stability_score = 0.0

        # 적정성: median(없으면 mean) 목표 1350 대비 상대오차
        loc = mpt_med if (mpt_med is not None and not np.isnan(mpt_med)) else mpt_mean
        if loc not in (None, 0):
            target = 1350.0
            mean_err_ratio = abs(loc - target) / target
            temp_suitability_score = _linear_piecewise_ratio_to_score(mean_err_ratio,
                                                                      anchors=(0.0, 0.10, 0.25, 0.50),
                                                                      scores=(100, 80, 50, 0))
        else:
            temp_suitability_score = 0.0

        # LOAD 변동성: 완화 밴드
        if load_std is not None and not np.isnan(load_std):
            # 0.02 → 100, 0.08 → 80, 0.2 → 50, 0.5 → 0 (예시)
            load_stability_score = _linear_piecewise_ratio_to_score(load_std,
                                                                    anchors=(0.0, 0.02, 0.08, 0.20, 0.50),
                                                                    scores=(100, 100, 80, 50, 0))
        else:
            load_stability_score = 0.0

    final_score = 0.5 * temp_stability_score + 0.3 * temp_suitability_score + 0.2 * load_stability_score
    return {
        "temperature_stability": round(temp_stability_score, 1),
        "temperature_suitability": round(temp_suitability_score, 1),
        "load_stability": round(load_stability_score, 1),
        "final_score": round(final_score, 1)
    }


# =========================
# Metrics / Profiling
# =========================

def extend_process_metrics(df: pd.DataFrame, meta: dict) -> dict:
    """
    LW-DED 공정 데이터에서 시간, 에너지, 적층 부피, 안정성 지표까지 확장 생성.
    또한 '유효 가공 구간(valid)' 마스크를 정의하여 가공 통계를 분리/보강.
    """
    has_time = "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"])
    if not has_time:
        # time이 없으면 최소 메타만 유지
        meta["process_metrics"] = meta.get("process_metrics", {})
        return meta

    df["dt"] = df["time"].diff().dt.total_seconds().fillna(0)
    total_time = (df["time"].max() - df["time"].min()).total_seconds() if len(df) else 0.0
    total_time = float(total_time) if total_time and total_time > 0 else 0.0

    # 레이저 ON 마스크
    if "laser_on" in df.columns:
        mask_on = df["laser_on"].astype(str).str.lower().isin(["1", "true", "t"])
    else:
        mask_on = pd.Series(False, index=df.index)

    # 유효 가공 구간(valid) 마스크(예시 임계)
    r_lp = _safe_series(df, "r_lp")
    r_ws = _safe_series(df, "r_ws")
    contact = _safe_series(df, "contact")
    valid = mask_on.copy()
    # 실출력, 와이어, 접촉 임계 - 장비/소재별 조정 필요
    valid &= r_lp.fillna(0) > 0
    valid &= r_ws.fillna(0) > 0.05
    # contact 컬럼이 없을 수 있으므로 NaN은 False로 간주되지 않도록 fillna(0)
    valid &= contact.fillna(0) > 0.2

    # 시간 가중 합산
    laser_on_time = float(df.loc[mask_on, "dt"].sum()) if total_time else 0.0
    laser_off_time = float(total_time - laser_on_time) if total_time else None
    utilization = (laser_on_time / total_time) if total_time else None

    valid_time = float(df.loc[valid, "dt"].sum()) if total_time else 0.0
    on_ratio = (laser_on_time / total_time) if total_time else None
    valid_ratio = (valid_time / total_time) if total_time else None

    # 에너지/적층량
    energy_input = None
    if "r_lp" in df.columns:
        lp = pd.to_numeric(df["r_lp"], errors="coerce").fillna(0)
        energy_input = float((lp * df["dt"]).sum()) if len(lp) else None

    cum_volume = aw_mean = aw_peak = volume_rate_mean = volume_rate_std = None
    if {"mpa", "mpw"}.issubset(df.columns):
        A = pd.to_numeric(df["mpa"], errors="coerce").fillna(0)
        W = pd.to_numeric(df["mpw"], errors="coerce").fillna(0)
        AW = A * W
        k = 1e-3  # 장비/단위 보정 상수(예시)
        dV = k * AW * df["dt"]
        if not dV.empty:
            cum_volume = float(dV.cumsum().iloc[-1])
            aw_mean = float(AW.mean())
            aw_peak = float(AW.max())
            volume_rate = dV / df["dt"].replace(0, np.nan)
            volume_rate_mean = float(volume_rate.mean())
            volume_rate_std = float(volume_rate.std())

    # 와이어 효율
    wire_feed_idle_time = wire_eff = None
    if "r_ws" in df.columns:
        ws = pd.to_numeric(df["r_ws"], errors="coerce").fillna(0)
        wire_feed_idle_time = float(df.loc[mask_on & (ws <= 0.05), "dt"].sum())
        wire_eff = ((laser_on_time - wire_feed_idle_time) / laser_on_time) if (laser_on_time and laser_on_time > 0) else None

    # 레이저/스캔 변동성
    laser_power_std = float(pd.to_numeric(df.get("r_lp"), errors="coerce").std()) if "r_lp" in df else None
    scan_speed_std = float(pd.to_numeric(df.get("r_rs"), errors="coerce").std()) if "r_rs" in df else None

    # MPT 통계(전체/ON/VALID)
    def _stats_for(s: pd.Series) -> Dict[str, Any]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"max": None, "mean": None, "std": None, "median": None, "robust_std": None}
        return {
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "robust_std": _mad_to_robust_std(s)
        }

    mpt_stats_all = _stats_for(_safe_series(df, "mpt"))
    mpt_stats_on = _stats_for(_safe_series(df, "mpt")[mask_on])
    mpt_stats_valid = _stats_for(_safe_series(df, "mpt")[valid])

    load_stats_all = _stats_for(_safe_series(df, "load"))
    load_stats_on = _stats_for(_safe_series(df, "load")[mask_on])
    load_stats_valid = _stats_for(_safe_series(df, "load")[valid])

    # 비드/레이어 간단 요약
    bead_count = None
    if "bead_number" in df.columns:
        try:
            bead_count = int(pd.to_numeric(df["bead_number"], errors="coerce").nunique())
        except Exception:
            bead_count = None

    meta["process_metrics"] = {
        "total_time_sec": total_time,
        "laser_on_time_sec": laser_on_time,
        "laser_off_time_sec": laser_off_time,
        "utilization_ratio": utilization,
        "avg_sampling_rate_hz": (len(df) / total_time) if total_time else None,
        "energy_input_total": energy_input,
        "cum_volume_est": cum_volume,
        "aw_mean": aw_mean,
        "aw_peak": aw_peak,
        "volume_rate_mean": volume_rate_mean,
        "volume_rate_std": volume_rate_std,
        "wire_feed_idle_time_sec": wire_feed_idle_time,
        "wire_feed_efficiency": wire_eff,
        "laser_power_std": laser_power_std,
        "scan_speed_std": scan_speed_std,
        # 전체/ON/VALID 구분 통계 (필요 시 LLM/RAG가 참고 가능)
        "mpt_all": mpt_stats_all,
        "mpt_on": mpt_stats_on,
        "mpt_valid": mpt_stats_valid,
        "load_all": load_stats_all,
        "load_on": load_stats_on,
        "load_valid": load_stats_valid,
        # 비율
        "on_ratio": on_ratio,
        "valid_ratio": valid_ratio,
        # 구조 요약
        "bead_count": bead_count,
    }

    # 호환성 필드(기존 키 유지: downstream에서 직접 참조하는 경우 대비)
    meta["process_metrics"].update({
        "mpt_max": mpt_stats_all["max"],
        "mpt_mean": mpt_stats_all["mean"],
        "mpt_std": mpt_stats_all["std"],
    })

    return meta


def _profile_columns(df: pd.DataFrame) -> dict:
    """
    RAG 호환 컬럼 프로파일: dtype / 결측 / 간단 통계(min/max/mean)
    """
    cols = {}
    for c in df.columns:
        s = df[c]
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        dtype = str(s.dtype)
        info = {"dtype": dtype, "non_null": non_null, "nulls": nulls}
        if pd.api.types.is_numeric_dtype(s):
            try:
                info["stats"] = {
                    "min": float(np.nanmin(s)),
                    "max": float(np.nanmax(s)),
                    "mean": float(np.nanmean(s))
                }
            except Exception:
                pass
        cols[c] = info
    return cols


# =========================
# Main Entry
# =========================

def load_and_meta(csv_path: Path,
                  meta_dir: Path,
                  golden_standard_path: Optional[Path] = None
                  ) -> Tuple[pd.DataFrame, Dict[str, Any], Path]:
    """
    CSV를 로드하고, 유효 구간 기반 강건 통계를 포함한 메타데이터를 생성.
    골든 스탠더드가 제공되면 유사도/상대 안정성 점수도 계산하여 포함.

    Returns:
        df, meta, meta_path
    """
    # --- CSV 로드 ---
    last_err = None
    for enc in ("utf-8", "cp949", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    # ---- 'file' 계열 컬럼 강제 제거 (대/소문자, 공백 포함) ----
    drop_cols = [c for c in df.columns if str(c).strip().lower() == "file"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 컬럼 표준화
    df.columns = standardize_columns(df.columns)

    # 표준화 후에도 'file' 컬럼 제거 (이중 안전장치)
    drop_cols = [c for c in df.columns if str(c).strip().lower() == "file"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 날짜 파싱
    df = parse_dates(df)

    # 공정 ID와 날짜
    process_datetime = "N/A"
    process_id = f"LWDED_{csv_path.stem}"
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        dt_obj = df['time'].iloc[0]
        if pd.notna(dt_obj):
            process_datetime = dt_obj.isoformat()
            try:
                process_id = f"LWDED_{dt_obj.strftime('%y%m%d_%H%M%S')}"
            except Exception:
                # strftime 실패 시 스템 사용
                process_id = f"LWDED_{csv_path.stem}"

    # 메타데이터 기본 구조 (RAG 기대 키 포함)
    meta: Dict[str, Any] = {
        "process_id": process_id,
        "process_datetime": process_datetime,
        "file_name": csv_path.name,
        "file_path": str(csv_path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "file": str(csv_path.name),
        "columns": _profile_columns(df),
    }

    # 공정 메트릭 확장 (유효 구간 포함)
    meta = extend_process_metrics(df, meta)

    # --- 안정성 점수 계산을 위한 통계 수집(VALID 기준 우선, 없으면 ON, 없으면 전체) ---
    pm = meta.get("process_metrics", {})
    # mpt
    mpt_valid = pm.get("mpt_valid", {}) or {}
    mpt_on = pm.get("mpt_on", {}) or {}
    mpt_all = pm.get("mpt_all", {}) or {}
    # load
    load_valid = pm.get("load_valid", {}) or {}
    load_on = pm.get("load_on", {}) or {}
    load_all = pm.get("load_all", {}) or {}

    def _pick_priority(*vals):
        """VALID -> ON -> ALL 순으로 첫 유효값 선택, 없으면 None"""
        for v in vals:
            if isinstance(v, (int, float)) and not np.isnan(v):
                return float(v)
        return None

    stats_current = {
        "mpt": {
            "mean": _pick_priority(mpt_valid.get("mean"), mpt_on.get("mean"), mpt_all.get("mean")),
            "std_dev": _pick_priority(mpt_valid.get("std"), mpt_on.get("std"), mpt_all.get("std")),
            "median": _pick_priority(mpt_valid.get("median"), mpt_on.get("median"), mpt_all.get("median")),
            "robust_std": _pick_priority(mpt_valid.get("robust_std"), mpt_on.get("robust_std"), mpt_all.get("robust_std")),
        },
        "load": {
            "std_dev": _pick_priority(load_valid.get("std"), load_on.get("std"), load_all.get("std")),
        }
    }

    # --- 골든 스탠더드가 있으면 로드 ---
    golden_meta: Optional[Dict[str, Any]] = None
    if golden_standard_path and Path(golden_standard_path).exists():
        try:
            with open(golden_standard_path, "r", encoding="utf-8") as f:
                golden_meta = json.load(f)
        except Exception as e:
            print(f"[Golden Standard 로드 오류] {e}")

    golden_stats = None
    if golden_meta and isinstance(golden_meta, dict):
        gm = golden_meta.get("process_metrics", {})
        # 골든도 VALID -> ON -> ALL 우선
        gm_mpt_valid = gm.get("mpt_valid", {}) or {}
        gm_mpt_on = gm.get("mpt_on", {}) or {}
        gm_mpt_all = gm.get("mpt_all", {}) or {}

        gm_load_valid = gm.get("load_valid", {}) or {}
        gm_load_on = gm.get("load_on", {}) or {}
        gm_load_all = gm.get("load_all", {}) or {}

        golden_stats = {
            "mpt": {
                "mean": _pick_priority(gm_mpt_valid.get("mean"), gm_mpt_on.get("mean"), gm_mpt_all.get("mean")),
                "std_dev": _pick_priority(gm_mpt_valid.get("std"), gm_mpt_on.get("std"), gm_mpt_all.get("std")),
                "median": _pick_priority(gm_mpt_valid.get("median"), gm_mpt_on.get("median"), gm_mpt_all.get("median")),
                "robust_std": _pick_priority(gm_mpt_valid.get("robust_std"), gm_mpt_on.get("robust_std"), gm_mpt_all.get("robust_std")),
            },
            "load": {
                "std_dev": _pick_priority(gm_load_valid.get("std"), gm_load_on.get("std"), gm_load_all.get("std")),
            }
        }

        # 유사도 점수 (구조/결과물)
        try:
            meta["process_similarity_score"] = calculate_similarity_score(
                current_metrics={
                    "utilization_ratio": pm.get("utilization_ratio"),
                    "energy_input_total": pm.get("energy_input_total"),
                    "cum_volume_est": pm.get("cum_volume_est"),
                    "wire_feed_efficiency": pm.get("wire_feed_efficiency"),
                    "mpt_mean": stats_current["mpt"]["mean"],
                    "mpt_std": stats_current["mpt"]["std_dev"],
                },
                golden_metrics={
                    "utilization_ratio": gm.get("utilization_ratio"),
                    "energy_input_total": gm.get("energy_input_total"),
                    "cum_volume_est": gm.get("cum_volume_est"),
                    "wire_feed_efficiency": gm.get("wire_feed_efficiency"),
                    "mpt_mean": _pick_priority(gm_mpt_valid.get("mean"), gm_mpt_on.get("mean"), gm_mpt_all.get("mean")),
                    "mpt_std": _pick_priority(gm_mpt_valid.get("std"), gm_mpt_on.get("std"), gm_mpt_all.get("std")),
                }
            )
        except Exception as e:
            print(f"[유사도 점수 계산 오류] {e}")

    # --- 안정성 점수 계산 ---
    try:
        meta["process_stability_score"] = calculate_stability_scores(stats_current, golden_stats=golden_stats)
    except Exception as e:
        # 실패 시 최소 정보 제공
        print(f"[안정성 점수 계산 오류] {e}")
        meta["process_stability_score"] = {
            "temperature_stability": None,
            "temperature_suitability": None,
            "load_stability": None,
            "final_score": None
        }

    # --- JSON 저장 ---
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{process_id}.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[메타 저장 오류] {e}")

    return df, meta, meta_path
