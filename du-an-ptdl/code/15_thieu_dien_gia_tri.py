"""Điền giá trị thiếu cho biến số và biến phân loại; bẫy inplace; lưu lại giá trị đã điền."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT_SO_DIEN = ["diem_giua_ky", "ty_le_hoan_thanh_video"]

trung_vi = df[COT_SO_DIEN].median()
print("Trung vị dùng để điền:", trung_vi.round(3).to_dict())

d = df.fillna(trung_vi.to_dict())                       # mỗi cột một giá trị riêng
for cot in ["tinh_thanh", "thiet_bi_chinh"]:
    d[cot] = d[cot].fillna("Không rõ")                  # giữ nhóm thiếu thành một mức riêng
print("Số ô thiếu còn lại   :", int(d.isna().sum().sum()))
print("Số mức tinh_thanh    :", df["tinh_thanh"].nunique(), "->", d["tinh_thanh"].nunique())
print("Số dòng mức Không rõ :", int((d["tinh_thanh"] == "Không rõ").sum()))

# Bẫy: gọi inplace trên MỘT CỘT là gọi trên kết quả trung gian, dữ liệu không đổi
thu = df.copy()
thu["diem_giua_ky"].fillna(6.0, inplace=True)           # KHÔNG có tác dụng
print("Sau fillna(inplace=True) trên một cột, số ô thiếu:",
      int(thu["diem_giua_ky"].isna().sum()))
thu["diem_giua_ky"] = thu["diem_giua_ky"].fillna(6.0)   # cách sách dùng: gán lại
print("Sau phép gán lại, số ô thiếu                     :",
      int(thu["diem_giua_ky"].isna().sum()))

# Lưu lại giá trị đã điền để dùng cho dữ liệu mới
bo_dien = SimpleImputer(strategy="median").fit(df[COT_SO_DIEN])
da_hoc = {c: float(v) for c, v in zip(COT_SO_DIEN, bo_dien.statistics_)}
print("Giá trị bộ điền đã học:", da_hoc)
moi = pd.DataFrame({"diem_giua_ky": [np.nan], "ty_le_hoan_thanh_video": [0.5]})
print("Điền cho một dòng mới :", bo_dien.transform(moi).round(3).tolist())
