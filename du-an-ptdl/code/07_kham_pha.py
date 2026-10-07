"""Khám phá cấu trúc một bảng lạ: xem nhanh, lọc theo điều kiện, khảo sát biến phân loại."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

df.info()                                  # gộp shape, dtypes và số ô không thiếu

# Lọc theo hai điều kiện, rồi so tỉ lệ không đạt của nhóm với toàn bộ
mat_na = (df["so_bai_tap_nop"] <= 3) & (df["ty_le_hoan_thanh_video"] < 0.3)
hoc_it = df.loc[mat_na, ["ma_sv", "so_bai_tap_nop", "ty_le_hoan_thanh_video", "ket_qua"]]
print("\nSố dòng thỏa điều kiện:", len(hoc_it))
print(hoc_it.head(3))
print("Tỉ lệ không đạt trong nhóm:",
      round((hoc_it["ket_qua"] == "Không đạt").mean() * 100, 1), "%")
print("Tỉ lệ không đạt toàn bộ  :",
      round((df["ket_qua"] == "Không đạt").mean() * 100, 1), "%")
ba_tinh = df[df["tinh_thanh"].isin(["Hà Nội", "Hải Phòng", "Quảng Ninh"])]
print("Ba tỉnh phía bắc         :", len(ba_tinh), "dòng, điểm cuối kỳ trung bình",
      round(ba_tinh["diem_cuoi_ky"].mean(), 3))

# Khảo sát biến phân loại: số mức thật so với số mức khai trong bảng ý nghĩa cột
print()
for cot in ["hinh_thuc_hoc", "gioi_tinh", "nganh"]:
    print(f"{cot}: {df[cot].nunique()} mức")
print()
print(df["gioi_tinh"].value_counts(dropna=False))
print()
print(df["ket_qua"].value_counts(normalize=True).round(4))
