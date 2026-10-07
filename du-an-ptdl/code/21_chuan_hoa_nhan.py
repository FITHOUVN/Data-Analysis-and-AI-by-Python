"""Chuẩn hóa nhãn phân loại: bỏ khoảng trắng, gộp biến thể bằng bảng tra, và bẫy capitalize."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

# Danh sách nhãn mong đợi, viết TRƯỚC khi xem dữ liệu, theo bảng ý nghĩa cột
NHAN_MONG_DOI = {"gioi_tinh": {"Nam", "Nữ", "Khác"},
                 "vua_lam_vua_hoc": {"Có", "Không"},
                 "nganh": {"Công nghệ thông tin", "Kinh tế", "Quản trị kinh doanh",
                           "Kế toán", "Luật", "Ngôn ngữ Anh", "Du lịch"}}
BANG_TRA = {"nam": "Nam", "nữ": "Nữ", "khác": "Khác", "CO": "Có", "KHONG": "Không"}

d = df.copy()
for cot, mong_doi in NHAN_MONG_DOI.items():
    d[cot] = d[cot].str.strip().replace(BANG_TRA)
    thua = set(d[cot].dropna().unique()) - mong_doi
    print(f"{cot:18s} {df[cot].nunique():3d} mức -> {d[cot].nunique():3d} mức"
          f" | nhãn lạ còn lại: {thua if thua else 'không có'}")

print("\nSố dòng đã sửa nhãn theo từng cột:")
for cot in NHAN_MONG_DOI:
    print(f"  {cot:18s} {int((df[cot] != d[cot]).sum()):4d} dòng")
print("\ngioi_tinh sau khi chuẩn hóa:", d["gioi_tinh"].value_counts().to_dict())
print("vua_lam_vua_hoc sau khi chuẩn hóa:", d["vua_lam_vua_hoc"].value_counts().to_dict())

# Bẫy: capitalize gộp đúng gioi_tinh nhưng viết lại cả cột tinh_thanh
cap = df["tinh_thanh"].str.strip().str.capitalize()
print("Số mức tinh_thanh:", df["tinh_thanh"].nunique(), "-> sau capitalize:", cap.nunique())
co_so_do = df["tinh_thanh"].notna()
print("Số ô bị đổi chữ   :",
      int((df.loc[co_so_do, "tinh_thanh"].str.strip() != cap[co_so_do]).sum()),
      "trên", int(co_so_do.sum()), "ô có nhãn")
print("Ba nhãn đầu       :", sorted(df["tinh_thanh"].dropna().unique())[:3])
print("Sau capitalize    :", sorted(cap.dropna().unique())[:3])
print("nganh sau capitalize:", sorted(df["nganh"].str.strip().str.capitalize().unique())[-2:])
