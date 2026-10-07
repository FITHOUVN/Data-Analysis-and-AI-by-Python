"""Mã hóa biến thứ bậc và biến danh nghĩa (mục 4.2.1 và 4.2.2)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

MUC_TIEU = "diem_cuoi_ky"
# Thứ tự các mức phải khai bằng tay: không hàm nào suy ra được từ dữ liệu
THU_TU_NHOM_TUOI = ["18-22", "23-30", "31-40", "Trên 40"]
THU_TU_MUC_VIDEO = ["Thấp", "Trung bình", "Cao"]
COT_DANH_NGHIA = ["nganh", "gioi_tinh", "vua_lam_vua_hoc", "hinh_thuc_hoc",
                  "thiet_bi_chinh"]


def ma_hoa_thu_bac(d: pd.DataFrame) -> None:
    """Mã hóa cột thứ bậc nhom_tuoi với thứ tự khai tường minh."""
    bo = OrdinalEncoder(categories=[THU_TU_NHOM_TUOI]).fit(d[["nhom_tuoi"]])
    ma = bo.transform(d[["nhom_tuoi"]]).ravel()
    print("Thứ tự đã khai :", list(bo.categories_[0]))
    print("Thứ tự tự sắp  :", list(OrdinalEncoder().fit(d[["nhom_tuoi"]]).categories_[0]))
    print(pd.DataFrame({"nhom_tuoi": d["nhom_tuoi"], "ma": ma})
          .drop_duplicates().sort_values("ma").to_string(index=False))


def khi_thu_tu_sai(d: pd.DataFrame) -> None:
    """Một cột thứ bậc mà thứ tự bảng chữ cái khác hẳn thứ tự đúng."""
    muc = pd.cut(d["ty_le_hoan_thanh_video"], bins=[-0.001, 0.4, 0.7, 1.0],
                 labels=THU_TU_MUC_VIDEO).astype("str").to_frame("muc_video")
    dung = OrdinalEncoder(categories=[THU_TU_MUC_VIDEO]).fit_transform(muc).ravel()
    tu_sap_bo = OrdinalEncoder().fit(muc)
    tu_sap = tu_sap_bo.transform(muc).ravel()
    print("Số dòng mỗi mức:", muc["muc_video"].value_counts().to_dict())
    print("Thứ tự tự sắp:", list(tu_sap_bo.categories_[0]))
    print("Tương quan mã với diem_cuoi_ky: đúng thứ tự %+.4f | bảng chữ cái %+.4f"
          % (pd.Series(dung).corr(d[MUC_TIEU]), pd.Series(tu_sap).corr(d[MUC_TIEU])))


def ma_hoa_mot_nong(d: pd.DataFrame) -> None:
    """Mã hóa một nóng năm cột danh nghĩa, giữ tên cột bằng set_output."""
    bo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    bo.set_output(transform="pandas")              # giữ tên cột ở kết quả
    X = bo.fit_transform(d[COT_DANH_NGHIA])
    print("Năm cột danh nghĩa thành", X.shape[1], "cột nhị phân, kiểu",
          X.dtypes.unique().tolist())
    print(X.loc[:2, [c for c in X.columns if c.startswith("gioi_tinh")]].to_string())

    bo_bo_mot = OneHotEncoder(sparse_output=False, drop="first")
    bo_bo_mot.set_output(transform="pandas")
    X2 = bo_bo_mot.fit_transform(d[COT_DANH_NGHIA])
    print("Với drop='first':", X2.shape[1], "cột; mức bị bỏ của từng cột:",
          [c[0] for c in bo_bo_mot.categories_])
    dummies = pd.get_dummies(d[COT_DANH_NGHIA])
    print("pd.get_dummies cho", dummies.shape[1], "cột, kiểu",
          dummies.dtypes.unique().tolist())


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("-- Biến thứ bậc: nhom_tuoi --")
    ma_hoa_thu_bac(d)
    print("\n-- Khi thứ tự bảng chữ cái khác thứ tự đúng --")
    khi_thu_tu_sai(d)
    print("\n-- Biến danh nghĩa: mã hóa một nóng --")
    ma_hoa_mot_nong(d)


if __name__ == "__main__":
    main()
