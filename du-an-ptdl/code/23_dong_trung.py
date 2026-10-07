"""Đếm dòng trùng theo hai nghĩa, xem các dòng xung đột, và ghi báo cáo dòng trùng khóa."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
RA = GOC_DU_AN / "output"
RA.mkdir(exist_ok=True)
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

print("Trùng khít mọi cột, keep='first':", int(df.duplicated().sum()))
print("Trùng khít mọi cột, keep='last' :", int(df.duplicated(keep="last").sum()))
print("Trùng khít mọi cột, keep=False  :", int(df.duplicated(keep=False).sum()))
print("Trùng ma_sv,        keep='first':", int(df.duplicated(subset=["ma_sv"]).sum()))
print("Trùng ma_sv,        keep=False  :", int(df.duplicated(subset=["ma_sv"], keep=False).sum()))

d = df.drop_duplicates().reset_index(drop=True)
print("\nSau drop_duplicates():", d.shape, "| còn trùng ma_sv:",
      int(d.duplicated(subset=["ma_sv"]).sum()))

# Các dòng còn trùng mã: xem chúng khác nhau ở cột nào
xung_dot = d[d.duplicated(subset=["ma_sv"], keep=False)].sort_values("ma_sv")
print("Số dòng xung đột:", len(xung_dot), "| số mã:", xung_dot["ma_sv"].nunique())
cot_khac = {}
for ma, nhom in xung_dot.groupby("ma_sv"):          # cột nào gây xung đột
    for cot in d.columns:
        if nhom[cot].astype("str").nunique() > 1:
            cot_khac[cot] = cot_khac.get(cot, 0) + 1
print("Cột gây xung đột:", cot_khac)
print(xung_dot.loc[:, ["ma_sv", "so_lan_dien_dan", "diem_cuoi_ky"]].head(4).to_string(index=False))

xung_dot.to_csv(RA / "ch03-trung-ma-sv.csv", index=False, encoding="utf-8")
print("\nĐã ghi", len(xung_dot), "dòng vào output/ch03-trung-ma-sv.csv")
