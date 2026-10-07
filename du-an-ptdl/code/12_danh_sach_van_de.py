"""Quét bộ dữ liệu thô theo năm bước và ghi báo cáo cho chương sau đối chiếu."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
RA = GOC_DU_AN / "data" / "processed"
RA.mkdir(exist_ok=True)
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "so_lan_dien_dan", "nop_tre_tb_gio",
          "diem_giua_ky", "diem_cuoi_ky"]
COT_PHAN_LOAI = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                 "hinh_thuc_hoc", "thiet_bi_chinh", "ket_qua"]
ghi = []                                  # mỗi phát hiện một dòng, để ghi ra file ở cuối

print("--- 1. Giá trị thiếu ---")
for cot, so in df.isna().sum().items():
    if so:
        print(f"{cot:24s} {so:4d} ô  {so / len(df) * 100:5.2f}%")
        ghi.append(("thiếu", cot, so))

print("\n--- 2. Giá trị canh ---")
for cot, dieu_kien, mo_ta in [("tong_thoi_luong_phut", df["tong_thoi_luong_phut"] == 99999,
                               "bằng 99999"),
                              ("so_lan_dang_nhap", df["so_lan_dang_nhap"] < 0, "âm")]:
    print(f"{cot:24s} {mo_ta:12s} {dieu_kien.sum():3d} dòng")
    ghi.append(("canh", cot, int(dieu_kien.sum())))

print("\n--- 3. Ngoại lai theo quy tắc 1,5 IQR ---")
for cot in COT_SO:
    q1, q3 = df[cot].quantile(0.25), df[cot].quantile(0.75)
    iqr = q3 - q1
    vuot = ((df[cot] < q1 - 1.5 * iqr) | (df[cot] > q3 + 1.5 * iqr)).sum()
    if vuot:
        print(f"{cot:24s} vượt ngưỡng {vuot:4d} dòng")
        ghi.append(("ngoại lai", cot, int(vuot)))

print("\n--- 4. Nhãn và kiểu dữ liệu ---")
for cot in COT_PHAN_LOAI:
    if df[cot].nunique() != df[cot].astype(str).str.strip().str.lower().nunique():
        print(f"{cot:24s} nhãn không nhất quán")
        ghi.append(("nhãn", cot, df[cot].nunique()))
for cot in df.dtypes[df.dtypes == "str"].index:
    if df[cot].astype(str).str.contains(",").any():
        print(f"{cot:24s} cột số bị đọc thành chuỗi")
        ghi.append(("kiểu", cot, int(df[cot].astype(str).str.contains(",").sum())))

print("\n--- 5. Dòng trùng ---")
print("trùng hoàn toàn:", df.duplicated().sum(), "| trùng ma_sv:",
      df.duplicated(subset=["ma_sv"]).sum())
ghi.append(("trùng", "mọi cột", int(df.duplicated().sum())))
ghi.append(("trùng", "ma_sv", int(df.duplicated(subset=["ma_sv"]).sum())))

bao_cao = pd.DataFrame(ghi, columns=["loai", "cot", "so_dong_hoac_o"])
bao_cao.to_csv(RA / "ch02-bao-cao-kham-pha.csv", index=False, encoding="utf-8")
print("\nĐã ghi", len(bao_cao), "dòng vào data/processed/ch02-bao-cao-kham-pha.csv")
