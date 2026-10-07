"""Làm sạch toàn bộ bộ dữ liệu theo bảy bước và ghi ra data/processed/du-lieu-sach.csv."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
RA = GOC_DU_AN / "data" / "processed"
RA.mkdir(exist_ok=True)

BANG_TRA = {"nam": "Nam", "nữ": "Nữ", "khác": "Khác", "CO": "Có", "KHONG": "Không"}
COT_PHAN_LOAI = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc", "hinh_thuc_hoc",
                 "tinh_thanh", "thiet_bi_chinh", "ket_qua"]
COT_DIEN_TRUNG_VI = ["so_lan_dang_nhap", "tong_thoi_luong_phut",
                     "ty_le_hoan_thanh_video", "diem_giua_ky"]
COT_NGUYEN = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "so_bai_tap_nop", "so_lan_dien_dan"]
KIEU_MONG_DOI = {"so_lan_dang_nhap": "int64", "tong_thoi_luong_phut": "int64",
                 "so_bai_tap_nop": "int64", "so_lan_dien_dan": "int64",
                 "ty_le_hoan_thanh_video": "float64", "diem_tb_bai_tap": "float64",
                 "nop_tre_tb_gio": "float64", "diem_giua_ky": "float64",
                 "diem_cuoi_ky": "float64", "ngay_dang_ky": "datetime64[us]"}


def doc_ngay(cot: pd.Series) -> pd.Series:
    """Đọc một cột ngày có nhiều định dạng, mỗi lượt một định dạng tường minh."""
    ket = pd.Series(pd.NaT, index=cot.index, dtype="datetime64[us]")
    for dinh_dang in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        chua = ket.isna()
        ket[chua] = pd.to_datetime(cot[chua], format=dinh_dang, errors="coerce")
    return ket


def lam_sach(df: pd.DataFrame) -> pd.DataFrame:
    """Bảy bước làm sạch, đúng thứ tự chạy ở mục 3.4.3 của sách."""
    d = df.drop_duplicates().reset_index(drop=True)                      # 1
    assert d.duplicated().sum() == 0
    d["diem_tb_bai_tap"] = pd.to_numeric(                                # 2
        d["diem_tb_bai_tap"].str.replace(",", ".", regex=False), errors="coerce")
    d["ngay_dang_ky"] = doc_ngay(d["ngay_dang_ky"])
    assert d["ngay_dang_ky"].dt.month.nunique() == 2
    for cot in COT_PHAN_LOAI:                                            # 3
        d[cot] = d[cot].str.strip().replace(BANG_TRA)
    d["tong_thoi_luong_phut"] = d["tong_thoi_luong_phut"].mask(          # 4
        d["tong_thoi_luong_phut"] == 99999)
    d["so_lan_dang_nhap"] = d["so_lan_dang_nhap"].mask(d["so_lan_dang_nhap"] < 0)
    d = d.fillna(d[COT_DIEN_TRUNG_VI].median().to_dict())                # 5
    for cot in ["tinh_thanh", "thiet_bi_chinh"]:
        d[cot] = d[cot].fillna("Không rõ")
    assert d.isna().sum().sum() == 0
    for cot in COT_NGUYEN:                                               # 6
        d[cot] = d[cot].round().astype("int64")
    return d


def kiem(d: pd.DataFrame) -> None:
    """Bốn phép kiểm nghiệm thu và năm dòng báo cáo số liệu, chạy trước khi ghi file."""
    print("Kích thước          :", d.shape)
    print("Số ô thiếu          :", int(d.isna().sum().sum()))
    print("Dòng trùng khít     :", int(d.duplicated().sum()))
    lech = {c: str(d[c].dtype) for c, k in KIEU_MONG_DOI.items() if str(d[c].dtype) != k}
    print("Cột lệch kiểu       :", lech if lech else "không có")
    print("Số mức từng cột phân loại:", {c: int(d[c].nunique()) for c in COT_PHAN_LOAI})
    print("ket_qua             :", d["ket_qua"].value_counts().to_dict())
    print("diem_cuoi_ky        : trung bình", round(d["diem_cuoi_ky"].mean(), 3),
          "độ lệch chuẩn", round(d["diem_cuoi_ky"].std(), 3))
    print("Tương quan tong_thoi_luong_phut với diem_cuoi_ky:",
          round(d["tong_thoi_luong_phut"].corr(d["diem_cuoi_ky"]), 3))
    print("Còn trùng ma_sv     :", int(d.duplicated(subset=["ma_sv"]).sum()),
          "dòng, giữ lại có ghi chú")


def main() -> None:
    df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")
    print("Bản thô:", df.shape)
    d = lam_sach(df)
    kiem(d)
    d.to_csv(RA / "du-lieu-sach.csv", index=False, encoding="utf-8")
    print("\nĐã ghi", len(d), "dòng vào data/processed/du-lieu-sach.csv")


if __name__ == "__main__":
    main()
