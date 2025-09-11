# compare_corrs.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# === 1) 설정 ===
CSV_PATH = "001_1_data.csv"
# AnalysisVisualizer에서 쓰던 후보 컬럼
CANDIDATE_COLS = ['R_LP', 'R_RS', 'R_WS', 'MPT', 'MPA', 'MPW', 'LOAD', 'CONTACT']

# === 2) 헬퍼: bool/숫자 변환 ===
def coerce_bool(s):
    if s.dtype == bool:
        return s
    if s.dtype.kind in "iu":  # int/uint
        return s.astype(float).astype('Int64').map(lambda x: True if x==1 else (False if x==0 else np.nan))
    # 문자열일 수 있음
    return s.astype(str).str.strip().str.lower().map({
        'true': True, '1': True, 'y': True, 'yes': True, 't': True,
        'false': False, '0': False, 'n': False, 'no': False, 'f': False
    })

def to_numeric(df, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce')
    return pd.DataFrame(out)

# === 3) 데이터 로드 & 전처리 ===
df = pd.read_csv(CSV_PATH)

# LASER_ON 필터: 있으면 True만, 없으면 전체 사용
if 'LASER_ON' in df.columns:
    mask_bool = coerce_bool(df['LASER_ON'])
    df = df[mask_bool == True].copy()

# 분석에 사용할 실제 컬럼 선정(존재 + 수치화 가능)
present_cols = [c for c in CANDIDATE_COLS if c in df.columns]
num_df = to_numeric(df, present_cols)

# 모든 상관은 "쌍별로 유효한 행" 기준으로 계산될 수 있도록, 각 쌍에서 NaN 드랍
if len(num_df.columns) < 2 or num_df.dropna(how='all').empty:
    raise SystemExit("유효한 수치형 컬럼이 2개 미만이거나 데이터가 비었습니다.")

# === 4) pandas 피어슨 상관행렬 ===
corr_pd = num_df.corr(method='pearson', min_periods=2)

# === 5) pearsonr 쌍별 계산 ===
# pandas와 동일하게, 각 (x,y)쌍마다 공통 유효행만 골라 계산한다
pairs = []
for i, c1 in enumerate(num_df.columns):
    for c2 in num_df.columns[i:]:
        # 공통 유효행
        m = num_df[c1].notna() & num_df[c2].notna()
        n = int(m.sum())
        if n >= 2:
            r, p = pearsonr(num_df.loc[m, c1], num_df.loc[m, c2])
            pairs.append((c1, c2, r, p, n))
        else:
            pairs.append((c1, c2, np.nan, np.nan, n))

# === 6) 비교 보고 ===
print("=== pandas .corr() (pearson) ===")
print(corr_pd.round(6))
print()

# pearsonr 결과를 행렬 형태로 재구성
cols = num_df.columns.tolist()
pearsonr_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
n_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
for c1, c2, r, p, n in pairs:
    pearsonr_mat.loc[c1, c2] = r
    pearsonr_mat.loc[c2, c1] = r
    n_mat.loc[c1, c2] = n
    n_mat.loc[c2, c1] = n

print("=== scipy.stats.pearsonr (쌍별 유효행 기준) ===")
print(pearsonr_mat.round(6))
print()

# 차이 행렬
diff = (corr_pd - pearsonr_mat).abs()
print("=== |pandas - pearsonr| 차이 (절대값) ===")
print(diff.round(12))
print()

# 임계치 이상만 강조
TOL = 1e-10
bad = (diff > TOL)
if bad.any().any():
    print(f"[경고] 임계치 {TOL} 초과 차이 존재 → 아래 쌍 확인")
    where_bad = np.where(bad.values)
    for i, j in zip(*where_bad):
        c1, c2 = corr_pd.index[i], corr_pd.columns[j]
        print(f"  - ({c1}, {c2}): pandas={corr_pd.loc[c1,c2]:.12f}, pearsonr={pearsonr_mat.loc[c1,c2]:.12f}, "
              f"diff={diff.loc[c1,c2]:.12e}, n={int(n_mat.loc[c1,c2])}")
else:
    print("[OK] 두 방식이 동일합니다(수치 오차 범위 내).")

# 보조 정보: 각 컬럼별 NaN 개수와 유효 개수
print("\n=== NaN/유효 개수 ===")
nan_counts = num_df.isna().sum()
valid_counts = num_df.notna().sum()
print(pd.DataFrame({"nan": nan_counts, "valid": valid_counts}))
