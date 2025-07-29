# core/plotting.py
from __future__ import annotations
import pandas as pd
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

def df_to_table(tbl: QTableWidget, df: pd.DataFrame):
    tbl.clear()
    tbl.setRowCount(df.shape[0])
    tbl.setColumnCount(df.shape[1])
    tbl.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            tbl.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

def plot_df_line(ax, canvas, df: pd.DataFrame):
    ax.clear()
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) >= 1:
        y = df[num_cols[0]]; x = df.index
        ax.plot(x, y, marker='o')
        ax.set_xlabel("Index"); ax.set_ylabel(num_cols[0]); ax.set_title("Line Chart")
    else:
        ax.text(0.5, 0.5, "No numeric columns", ha='center')
    canvas.draw()

def build_report_text(df: pd.DataFrame, sql: str|None):
    md = "# 자동 분석 보고서\n\n"
    if sql:
        md += "## 사용 SQL\n```sql\n" + sql + "\n```\n"
    md += f"## 결과 요약\n- 행 수: {len(df)}\n\n"
    try:
        md += df.head(20).to_markdown(index=False)
    except Exception:
        md += df.head(20).to_string(index=False)
    return md
