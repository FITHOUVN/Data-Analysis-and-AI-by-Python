"""Đọc file CSV lần đầu, rồi bốn tham số của read_csv chữa bốn triệu chứng đã gặp."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_DU_LIEU = GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv"
TAM = GOC_DU_AN / "data" / "processed"
TAM.mkdir(exist_ok=True)

# Miền mong đợi, viết ra TRƯỚC khi xem dữ liệu: chín cột này phải là cột số
COT_SO_MONG_DOI = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
                   "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
                   "nop_tre_tb_gio", "diem_giua_ky", "diem_cuoi_ky"]

thu = pd.read_csv(FILE_DU_LIEU, encoding="utf-8", nrows=5)        # bước 2
print(thu[["ma_sv", "hinh_thuc_hoc", "diem_cuoi_ky", "ket_qua"]])

df = pd.read_csv(FILE_DU_LIEU, encoding="utf-8")
print("\nSố dòng, số cột:", df.shape)                              # bước 3
print("\nCột lẽ ra là số mà không phải số:")                       # bước 4
kieu = df[COT_SO_MONG_DOI].dtypes
print(kieu[kieu == "str"])
print("\nSố ô thiếu theo cột:")
print(df.isna().sum()[df.isna().sum() > 0])                        # bước 5

# usecols + nrows: chỉ đọc phần đang cần
mau = pd.read_csv(FILE_DU_LIEU, encoding="utf-8", nrows=5,
                  usecols=["diem_cuoi_ky", "ma_sv", "hinh_thuc_hoc", "diem_giua_ky"])
print("\nusecols cho các cột:", list(mau.columns), "| kích thước", mau.shape)

f_ma = TAM / "ma-lop.csv"                    # mã lớp có số 0 ở đầu
f_ma.write_text("ma_lop,si_so\n007,45\n012,38\n", encoding="utf-8")
print("Không khai dtype:", pd.read_csv(f_ma, encoding="utf-8")["ma_lop"].tolist())
print("Khai dtype str  :",
      pd.read_csv(f_ma, encoding="utf-8", dtype={"ma_lop": "str"})["ma_lop"].tolist())

f_sep = TAM / "sep.csv"                      # file xuất theo quy ước vùng miền Việt Nam
f_sep.write_text("ma_sv;diem_giua_ky;diem_cuoi_ky\nSV001;6,5;7,0\nSV002;4,0;5,5\n",
                 encoding="utf-8")
sai = pd.read_csv(f_sep, encoding="utf-8")
print("Đọc mặc định       :", sai.shape, list(sai.columns))
dung = pd.read_csv(f_sep, encoding="utf-8", sep=";", decimal=",")
print("Khai sep và decimal:", dung.shape, dung["diem_cuoi_ky"].dtype)
