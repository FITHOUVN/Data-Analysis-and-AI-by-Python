"""Bốn hệ quả của việc một DataFrame là tập các Series dùng chung một chỉ mục bất biến."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_DU_LIEU = GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv"
df = pd.read_csv(FILE_DU_LIEU, encoding="utf-8")

# Hệ quả 1: một tên cột cho Series, một danh sách tên cột cho DataFrame
print("1 |", type(df["ma_sv"]).__name__, "có .str:", df["ma_sv"].str[:2].head(2).tolist())
print("1 |", type(df[["ma_sv"]]).__name__, "có .str:", hasattr(df[["ma_sv"]], "str"))

# Hệ quả 2: mỗi cột một kiểu, nên sum() nối chuỗi còn mean() dừng
nho = df.loc[:3, ["hinh_thuc_hoc", "so_bai_tap_nop", "diem_cuoi_ky"]]
print("2 | sum()              :", nho.sum().to_dict())
print("2 | sum(numeric_only=1):", nho.sum(numeric_only=True).to_dict())
try:
    nho.mean()
except TypeError as loi:
    print("2 | mean()             :", loi)

# Hệ quả 3: lọc xong, chỉ mục vẫn giữ nhãn dòng của bảng gốc
yeu = df[df["diem_cuoi_ky"] < 4.0]
print("3 |", len(yeu), "dòng, năm nhãn đầu:", yeu.index[:5].tolist())
print("3 | sau reset_index   :", yeu.reset_index(drop=True).index[:5].tolist())

# Hệ quả 4: chỉ mục bất biến, sửa nhãn phải tạo chỉ mục mới
try:
    df.columns[0] = "ma_sinh_vien"
except TypeError as loi:
    print("4 | gán vào Index     :", loi)
