"""Đọc thử bộ dữ liệu của sách và in kích thước bảng."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_DU_LIEU = GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv"

df = pd.read_csv(FILE_DU_LIEU, encoding="utf-8")

print("Số dòng, số cột:", df.shape)
print("Năm cột đầu    :", list(df.columns)[:5])
print("Tên cột mục tiêu:", df.columns[-2], "và", df.columns[-1])
