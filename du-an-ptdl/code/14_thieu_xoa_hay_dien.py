"""Đo cái giá của việc xóa dòng, xóa cột, và đo mức méo khi điền theo tỉ lệ thiếu."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

print("Bảng gốc                     :", df.shape)
print("dropna()                     :", df.dropna().shape,
      "mất", len(df) - len(df.dropna()), "dòng =",
      round((1 - len(df.dropna()) / len(df)) * 100, 1), "%")
print("dropna(subset=diem_giua_ky)  :", df.dropna(subset=["diem_giua_ky"]).shape)
print("dropna(how='all')            :", df.dropna(how="all").shape)
print("dropna(thresh=18)            :", df.dropna(thresh=18).shape)
cot_thieu = df.columns[df.isna().any()]
print("drop(columns=4 cột thiếu)    :",
      df.drop(columns=cot_thieu).shape)
so_o_thieu = int(df.isna().sum().sum())
print("Tổng số ô thiếu              :", so_o_thieu, "/", df.size,
      "=", round(so_o_thieu / df.size * 100, 2), "%")

# Thiếu bao nhiêu thì điền còn đáng tin: khoét thêm ô thiếu rồi điền trung vị và đo lại
# 2.861 dòng đủ
day = df.dropna(subset=["diem_giua_ky"]).reset_index(drop=True)
goc_corr = day["diem_giua_ky"].corr(day["diem_cuoi_ky"])
goc_std = day["diem_giua_ky"].std()
rng = np.random.default_rng(42)   # khóa hạt ngẫu nhiên để tái lập
print(f"\nTrên {len(day)} dòng đủ: tương quan {goc_corr:.4f}, độ lệch chuẩn {goc_std:.4f}")
print(f"{'tỉ lệ thiếu':>12} {'tương quan':>11} {'giảm':>7} {'độ lệch chuẩn':>15} {'giảm':>7}")
for ty_le in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    cot = day["diem_giua_ky"].copy()
    chon = rng.choice(len(cot), size=int(len(cot) * ty_le),
                      replace=False)
    cot.iloc[chon] = np.nan
    # khoét ô thiếu rồi điền trung vị
    dien = cot.fillna(cot.median())
    c = dien.corr(day["diem_cuoi_ky"])
    print(f"{ty_le * 100:11.0f}% {c:11.4f} "
          f"{(c / goc_corr - 1) * 100:6.1f}% "
          f"{dien.std():15.4f} "
          f"{(dien.std() / goc_std - 1) * 100:6.1f}%")
