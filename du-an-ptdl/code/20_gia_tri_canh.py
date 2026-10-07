"""Đổi giá trị canh thành giá trị thiếu rồi điền, và đo tác động lên thống kê và tương quan."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

MUC_TIEU = "diem_cuoi_ky"
# Mỗi dòng: cột, điều kiện nhận ra giá trị canh, mô tả để in báo cáo
GIA_TRI_CANH = [("tong_thoi_luong_phut", df["tong_thoi_luong_phut"] == 99999, "bằng 99999"),
                ("so_lan_dang_nhap", df["so_lan_dang_nhap"] < 0, "âm")]

d = df.copy()
for cot, dieu_kien, mo_ta in GIA_TRI_CANH:
    goc = d[cot]
    d[cot] = goc.mask(dieu_kien)                       # giá trị canh thành giá trị thiếu
    trung_vi = d[cot].median()
    d[cot] = d[cot].fillna(trung_vi)
    print(f"{cot} ({mo_ta}, {int(dieu_kien.sum())} dòng)")
    print(f"  giữ nguyên   : trung bình {goc.mean():9.2f}  độ lệch {goc.std():9.2f}"
          f"  tương quan {goc.corr(df[MUC_TIEU]):+.4f}")
    print(f"  thành thiếu  : còn {int(goc.mask(dieu_kien).notna().sum())} ô có số đo,"
          f" trung vị {trung_vi}")
    print(f"  sau khi điền : trung bình {d[cot].mean():9.2f}  độ lệch {d[cot].std():9.2f}"
          f"  tương quan {d[cot].corr(df[MUC_TIEU]):+.4f}")
    print(f"  kiểu dữ liệu : {goc.dtype} -> {d[cot].dtype}")

d["so_lan_dang_nhap"] = d["so_lan_dang_nhap"].astype("int64")   # phục hồi kiểu số nguyên
print("\nSau khi phục hồi kiểu:", d["so_lan_dang_nhap"].dtype,
      "| nhỏ nhất:", d["so_lan_dang_nhap"].min())
