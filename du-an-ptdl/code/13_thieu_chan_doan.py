"""Chẩn đoán từng cột có ô thiếu: thiếu bao nhiêu, thiếu ở đâu, thiếu có hệ thống không."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

MUC_TIEU = "diem_cuoi_ky"
do_lech = df[MUC_TIEU].std()

dong = []
for cot in df.columns[df.isna().any()]:
    thieu = df[cot].isna()
    tb_thieu = df.loc[thieu, MUC_TIEU].mean()
    tb_du = df.loc[~thieu, MUC_TIEU].mean()
    dong.append({"cot": cot, "so_o": int(thieu.sum()),
                 "ty_le_%": round(thieu.mean() * 100, 2),
                 "diem_nhom_thieu": round(tb_thieu, 3),
                 "diem_nhom_du": round(tb_du, 3),
                 "chenh_tren_do_lech": round((tb_thieu - tb_du) / do_lech, 3)})
print(pd.DataFrame(dong).sort_values("so_o", ascending=False).to_string(index=False))

print("\nĐộ lệch chuẩn của", MUC_TIEU, ":", round(do_lech, 3))
print("Số dòng có ít nhất một ô thiếu:", df.isna().any(axis=1).sum())
print("Tổng số ô thiếu / tổng số ô   :", df.isna().sum().sum(), "/", df.size)

# Nhóm thiếu tỉnh thành có kết quả học tập khác hẳn: căn cứ cho quyết định ở mục 3.1.4
thieu_tinh = df["tinh_thanh"].isna()
print("\nTỉ lệ không đạt của nhóm thiếu tinh_thanh:",
      round((df.loc[thieu_tinh, "ket_qua"] == "Không đạt").mean() * 100, 1), "%")
print("Tỉ lệ không đạt của toàn bộ bảng        :",
      round((df["ket_qua"] == "Không đạt").mean() * 100, 1), "%")
