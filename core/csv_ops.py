# core/csv_ops.py
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

def standardize_columns(cols):
    return [re.sub(r'[^0-9a-zA-Z_]+', '_', str(c).strip().lower()).strip('_') for c in cols]

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [c for c in df.columns
                  if any(k in c.lower() for k in ('time','date','datetime','timestamp','ts'))]
    for c in candidates:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        parsed = None
        # numeric epoch s/ms
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            ss = s.dropna()
            if not ss.empty:
                try:
                    mn, mx = float(np.nanmin(ss)), float(np.nanmax(ss))
                    if 1e9 <= mn <= 1e12 and 1e9 <= mx <= 1e12:
                        cand = pd.to_datetime(s, unit='s', errors='coerce', utc=False)
                    elif 1e12 <= mn <= 1e15 and 1e12 <= mx <= 1e15:
                        cand = pd.to_datetime(s, unit='ms', errors='coerce', utc=False)
                    else:
                        cand = None
                    if cand is not None and cand.notna().mean() >= 0.8:
                        parsed = cand
                except Exception:
                    parsed = None
        # string mixed
        if parsed is None and pd.api.types.is_object_dtype(s):
            try:
                cand = pd.to_datetime(s, format='mixed', errors='coerce', utc=False)
            except TypeError:
                cand = pd.to_datetime(s, errors='coerce', utc=False)
            if cand.notna().mean() >= 0.8:
                parsed = cand
        if parsed is not None:
            df[c] = parsed
    return df

def load_and_meta(csv_path: Path, meta_dir: Path) -> tuple[pd.DataFrame, dict, Path]:
    # encoding fallback
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

    df.columns = standardize_columns(df.columns)
    df = parse_dates(df)

    meta = {"file": csv_path.name, "rows": int(len(df)), "cols": int(df.shape[1]), "columns": {}}
    for col in df.columns:
        s = df[col]
        col_meta = {
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "sample": s.dropna().astype(str).head(3).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s) and s.dropna().size:
            col_meta["stats"] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}
        meta["columns"][col] = col_meta

    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / f"{csv_path.stem}.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, meta, out