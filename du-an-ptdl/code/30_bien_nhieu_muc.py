"""Biến phân loại nhiều mức và mức xuất hiện rất ít (mục 4.2.3)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

MUC_TIEU = "diem_cuoi_ky"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
# Sáu cột phân loại được chọn làm đặc trưng. Danh sách này hẹp hơn COT_PHAN_LOAI tám cột
# của Chương 3, nên nó mang một tên khác để khỏi nhập nhằng.
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def so_muc(d: pd.DataFrame) -> None:
    """Số mức, mức hiếm nhất và tỉ lệ của mức đó cho từng cột phân loại."""
    dong = []
    for c in DAC_TRUNG_PL + ["tinh_thanh"]:
        dem = d[c].value_counts()
        dong.append({"cot": c, "so_muc": len(dem), "muc_hiem_nhat": dem.index[-1],
                     "so_dong": int(dem.iloc[-1]),
                     "ty_le_%": round(dem.iloc[-1] / len(d) * 100, 2)})
    print(pd.DataFrame(dong).to_string(index=False))


def gia_cua_mot_cot_nhieu_muc(d: pd.DataFrame) -> None:
    """Thêm tinh_thanh vào đặc trưng được gì và mất gì, đo bằng R2."""
    for ten, cat in (("6 cột phân loại", DAC_TRUNG_PL),
                     ("7 cột, thêm tinh_thanh", DAC_TRUNG_PL + ["tinh_thanh"])):
        X = d[COT_SO + cat]
        X_tr, X_te, y_tr, y_te = train_test_split(X, d[MUC_TIEU], test_size=0.2,
                                                  random_state=42)
        tien = ColumnTransformer([("so", StandardScaler(), COT_SO),
                                  ("pl", OneHotEncoder(handle_unknown="ignore"), cat)])
        mo = Pipeline([("tien", tien), ("hoi_quy", LinearRegression())]).fit(X_tr, y_tr)
        print("%-24s %3d đặc trưng | R2 = %.4f"
              % (ten, mo.named_steps["tien"].transform(X_tr).shape[1],
                 r2_score(y_te, mo.predict(X_te))))


def gop_muc_hiem(d: pd.DataFrame) -> None:
    """Tham số min_frequency gộp các mức dưới ngưỡng thành một mức chung."""
    for nguong in (0, 100, 150):
        bo = OneHotEncoder(sparse_output=False,
                           min_frequency=nguong if nguong else None).fit(d[["tinh_thanh"]])
        print("min_frequency = %-4s -> %2d cột; mức bị gộp: %s"
              % (nguong or "None", len(bo.get_feature_names_out()),
                 list(bo.infrequent_categories_[0]) if nguong else "không có"))


def muc_hiem_sau_khi_chia(d: pd.DataFrame) -> None:
    """Mức 'Khác' của gioi_tinh còn bao nhiêu dòng trong mỗi tập."""
    tr, te = train_test_split(d, test_size=0.2, random_state=42)
    for ten, phan in (("toàn bộ", d), ("tập huấn luyện", tr), ("tập kiểm tra", te)):
        print("%-15s %s" % (ten, {k: int(v) for k, v in phan["gioi_tinh"].value_counts().items()}))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("-- Số mức và mức hiếm nhất của từng cột phân loại --")
    so_muc(d)
    print("\n-- Cái giá của một cột 16 mức --")
    gia_cua_mot_cot_nhieu_muc(d)
    print("\n-- Gộp mức hiếm bằng min_frequency --")
    gop_muc_hiem(d)
    print("\n-- Mức 'Khác' của gioi_tinh sau khi chia tập --")
    muc_hiem_sau_khi_chia(d)


if __name__ == "__main__":
    main()
