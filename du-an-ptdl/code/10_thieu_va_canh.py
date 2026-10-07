"""Đếm và định vị giá trị thiếu, rồi tìm giá trị canh bằng miền giá trị hợp lý."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

bao_cao = pd.DataFrame({
    "so_o_thieu": df.isna().sum(),
    "ty_le_phan_tram": (df.isna().mean() * 100).round(2),
})
print(bao_cao[bao_cao["so_o_thieu"] > 0].sort_values("so_o_thieu", ascending=False))
print("\nSố dòng có ít nhất một ô thiếu:", df.isna().any(axis=1).sum())
print("Số dòng thiếu từ hai ô trở lên :", (df.isna().sum(axis=1) >= 2).sum())
print("Tổng số ô thiếu                :", df.isna().sum().sum())

# Việc thiếu có mang thông tin không: so biến mục tiêu giữa nhóm thiếu và nhóm không thiếu
print("\nĐiểm cuối kỳ của nhóm thiếu và nhóm không thiếu điểm giữa kỳ:")
print(df.groupby(df["diem_giua_ky"].isna())["diem_cuoi_ky"].agg(["count", "mean"]).round(3))

# Giá trị canh: đếm số dòng nằm ngoài miền hợp lý đã viết trước khi xem dữ liệu
print("\ntong_thoi_luong_phut bằng 99999:", (df["tong_thoi_luong_phut"] == 99999).sum())
print("so_lan_dang_nhap âm            :", (df["so_lan_dang_nhap"] < 0).sum())
print("nop_tre_tb_gio trên 100 giờ    :", (df["nop_tre_tb_gio"] > 100).sum())
print("Mười giá trị nộp muộn lớn nhất :",
      sorted(df["nop_tre_tb_gio"].nlargest(10).tolist()))
