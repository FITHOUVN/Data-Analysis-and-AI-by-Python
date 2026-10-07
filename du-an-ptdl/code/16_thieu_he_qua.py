"""Đo hệ quả của bốn cách xử lý giá trị thiếu lên phân bố và lên quan hệ với biến mục tiêu."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

goc, y = df["diem_giua_ky"], df["diem_cuoi_ky"]
trung_vi, trung_binh = goc.median(), goc.mean()

cach = {"xóa dòng thiếu": (goc.dropna(), y[goc.notna()]),
        "điền trung vị": (goc.fillna(trung_vi), y),
        "điền trung bình": (goc.fillna(trung_binh), y),
        "điền 0": (goc.fillna(0.0), y)}
dong = [{"cach": ten, "n": len(x), "trung_binh": round(x.mean(), 3),
         "do_lech": round(x.std(), 4), "Q1": round(x.quantile(0.25), 3),
         "Q3": round(x.quantile(0.75), 3),
         "so_o_bang_trung_vi": int((x == trung_vi).sum()),
         "tuong_quan": round(x.corr(yy), 4)} for ten, (x, yy) in cach.items()]
print(pd.DataFrame(dong).to_string(index=False))

# Điền bằng mức hay gặp nhất xóa mất nhóm thiếu; điền "Không rõ" giữ nhóm đó lại
hay_gap = df["tinh_thanh"].mode()[0]
print("\nMức hay gặp nhất của tinh_thanh:", hay_gap,
      "| số dòng trước khi điền:", int((df["tinh_thanh"] == hay_gap).sum()))
print("Nếu điền bằng mức hay gặp nhất :", int((df["tinh_thanh"].fillna(hay_gap) == hay_gap).sum()),
      "dòng, nhóm thiếu không còn truy lại được")

# Trung vị tính trên toàn bộ dữ liệu khác trung vị tính trên tập huấn luyện
tr, te = train_test_split(df, test_size=0.2, random_state=42)
print("\nTrung vị tong_thoi_luong_phut toàn bộ    :", df["tong_thoi_luong_phut"].median())
print("Trung vị tong_thoi_luong_phut tập huấn luyện:", tr["tong_thoi_luong_phut"].median())
