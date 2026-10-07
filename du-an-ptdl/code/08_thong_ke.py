"""Thống kê mô tả, thống kê theo nhóm, bẫy độ lệch chuẩn và tương quan với biến mục tiêu."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 20)       # thiếu dòng này, bảng in ra bị thay bằng "..."
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "so_lan_dien_dan", "nop_tre_tb_gio",
          "diem_giua_ky", "diem_cuoi_ky"]

print(df[COT_SO].describe().round(2).T)
print()
print(df[["nganh", "gioi_tinh", "ket_qua"]].describe())

# Thống kê theo nhóm: cách gọi gọn, và cách tổng hợp có đặt tên
print()
print(df.groupby("hinh_thuc_hoc")["diem_cuoi_ky"].agg(
    ["count", "mean", "median", "std"]).round(3))
print()
print(df.groupby("thiet_bi_chinh").agg(
    so_sv=("ma_sv", "count"),
    diem_tb=("diem_cuoi_ky", "mean"),
    do_lech=("diem_cuoi_ky", "std"),
    bai_tap_tb=("so_bai_tap_nop", "mean"),
).round(3))
print()
print(df.groupby("vua_lam_vua_hoc")["diem_cuoi_ky"].agg(["count", "mean", "std"]).round(3))

# Bẫy im lặng: hai cách gọi hàm độ lệch chuẩn trên cùng sáu dòng đầu
nho = df.loc[:5, ["hinh_thuc_hoc", "diem_cuoi_ky"]]
print()
print(nho.groupby("hinh_thuc_hoc")["diem_cuoi_ky"].agg(np.std).round(4))   # KHÔNG dùng
print()
print(nho.groupby("hinh_thuc_hoc")["diem_cuoi_ky"].agg("std").round(4))    # cách đúng

# Tương quan với biến mục tiêu, và tác động của 14 giá trị canh lên hệ số
print()
print(df[COT_SO].corr(numeric_only=True)["diem_cuoi_ky"]
      .sort_values(ascending=False).round(3))
cot, muc_tieu = "tong_thoi_luong_phut", "diem_cuoi_ky"
bo_canh = df[df[cot] != 99999]
print("\nPearson trên cả bảng    :", round(df[cot].corr(df[muc_tieu]), 3))
print("Spearman trên cả bảng   :", round(df[cot].corr(df[muc_tieu], method="spearman"), 3))
print("Pearson sau khi bỏ 99999:", round(bo_canh[cot].corr(bo_canh[muc_tieu]), 3))
print("Số dòng đã bỏ           :", len(df) - len(bo_canh))
