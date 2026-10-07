"""Phân biệt ngoại lai do lỗi ghi nhận với ngoại lai có thật: bước nhảy và giả thuyết sai đơn vị."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT = "nop_tre_tb_gio"
iqr = df[COT].quantile(0.75) - df[COT].quantile(0.25)
lon = sorted(df[COT].nlargest(10).tolist())
print("Khoảng tứ phân vị:", iqr, "giờ")
print("Mười giá trị lớn nhất:", lon)

# Bước nhảy: khoảng cách giữa hai giá trị liền nhau, đo bằng khoảng tứ phân vị
print("\nKhoảng cách giữa hai giá trị liền nhau:")
for truoc, sau in zip(lon, lon[1:]):               # khoảng cách hai giá trị liền nhau
    rong = (sau - truoc) / iqr
    print(f"  {truoc:6.1f} -> {sau:6.1f}  rộng {sau - truoc:6.1f} giờ = {rong:5.2f} IQR"
          + ("  <-- bước nhảy" if rong >= 0.9 else ""))

# Giả thuyết: tám ô này được ghi bằng phút thay vì bằng giờ
nghi = sorted(df.loc[df[COT] > 100, COT].tolist())
binh_thuong = df.loc[df[COT] <= 100, COT]
print("\nGiả thuyết sai đơn vị, chia cho 60:")
print("  nguyên bản:", nghi)
print("  chia 60   :", [round(v / 60, 2) for v in nghi])
print("  miền của phần còn lại: từ", binh_thuong.min(), "tới", binh_thuong.max(), "giờ")
print("  tám giá trị sau khi chia 60 đều nằm trong miền đó:",
      all(binh_thuong.min() <= v / 60 <= binh_thuong.max() for v in nghi))

# Miền hợp lý theo nghiệp vụ, viết trước khi xem dữ liệu
MIEN_HOP_LY = {"so_lan_dang_nhap": (0, 500), "tong_thoi_luong_phut": (0, 20000),
               "ty_le_hoan_thanh_video": (0.0, 1.0), "nop_tre_tb_gio": (0.0, 2000.0)}
print("\nSố dòng nằm ngoài miền hợp lý:")
for cot, (thap, cao) in MIEN_HOP_LY.items():
    print(f"  {cot:24s} [{thap}, {cao}]  {((df[cot] < thap) | (df[cot] > cao)).sum():3d} dòng")
