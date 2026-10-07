"""Đọc bản Excel của bộ dữ liệu."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_EXCEL = GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.xlsx"

sach = pd.ExcelFile(FILE_EXCEL)
print("Các sheet trong file:", sach.sheet_names)

dfx = pd.read_excel(FILE_EXCEL, sheet_name="Sheet1")
print("Kích thước:", dfx.shape)
print("Kiểu của diem_tb_bai_tap:", dfx["diem_tb_bai_tap"].dtype)
print("Năm ô đầu:", dfx["diem_tb_bai_tap"].head(5).tolist())
