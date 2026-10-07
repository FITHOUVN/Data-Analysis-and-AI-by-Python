"""Khoanh ngoại lai bằng quy tắc 1,5 IQR, rồi kiểm nhãn, kiểu dữ liệu và dòng trùng."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "so_lan_dien_dan", "nop_tre_tb_gio",
          "diem_giua_ky", "diem_cuoi_ky"]
COT_PHAN_LOAI = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                 "hinh_thuc_hoc", "thiet_bi_chinh", "ket_qua"]

print("Ngưỡng 1,5 IQR và số dòng vượt ngưỡng:")
for cot in COT_SO:
    q1, q3 = df[cot].quantile(0.25), df[cot].quantile(0.75)
    iqr = q3 - q1
    duoi, tren = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    vuot = ((df[cot] < duoi) | (df[cot] > tren)).sum()
    print(f"{cot:24s} Q1={q1:8.2f} Q3={q3:8.2f} IQR={iqr:8.2f} "
          f"[{duoi:9.2f}, {tren:9.2f}] vượt {vuot:4d}")

print("\nSố mức thô so với số mức sau khi chuẩn hóa thử:")
for cot in COT_PHAN_LOAI:
    tho = df[cot].nunique()
    chuan = df[cot].astype(str).str.strip().str.lower().nunique()
    print(f"{cot:18s} thô {tho:3d} mức | sau chuẩn hóa {chuan:3d} mức",
          "  <-- lệch" if tho != chuan else "")

sai = pd.to_numeric(df["diem_tb_bai_tap"], errors="coerce")
dung = pd.to_numeric(df["diem_tb_bai_tap"].str.replace(",", ".", regex=False),
                     errors="coerce")
print("\nSố ô dùng dấu phẩy thập phân:", df["diem_tb_bai_tap"].str.contains(",").sum())
print("Đổi kiểu thẳng  : thiếu", sai.isna().sum(), "ô, trung bình", round(sai.mean(), 3))
print("Thay dấu rồi đổi: thiếu", dung.isna().sum(), "ô, trung bình", round(dung.mean(), 3))

print("\nDòng trùng khít mọi cột  :", df.duplicated().sum())
print("Dòng trùng ma_sv         :", df.duplicated(subset=["ma_sv"]).sum())
print("Số mã sinh viên khác nhau:", df["ma_sv"].nunique(), "trên", len(df), "dòng")
