"""Hai đặc trưng không được đưa vào mô hình và cách nhận ra (mục 4.3.4)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

MUC_TIEU = "diem_cuoi_ky"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def ma_sinh_vien_lam_dac_trung(d: pd.DataFrame) -> None:
    """Thêm ma_sv vào đặc trưng: mô hình khớp hoàn hảo tập huấn luyện mà không học thêm gì."""
    X, y = d[COT_SO + ["ma_sv"]], d[MUC_TIEU]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    la = set(X_te["ma_sv"]) - set(X_tr["ma_sv"])
    print("Số mã sinh viên khác nhau:", int(d["ma_sv"].nunique()), "trên", len(d), "dòng")
    print("Mã của tập kiểm tra chưa từng thấy trong tập huấn luyện:",
          len(la), "trên", int(X_te["ma_sv"].nunique()))
    for ten, tien in (
            ("có thêm ma_sv", ColumnTransformer(
                [("so", "passthrough", COT_SO),
                 ("ma", OneHotEncoder(handle_unknown="ignore"), ["ma_sv"])])),
            ("không có ma_sv", ColumnTransformer([("so", "passthrough", COT_SO)]))):
        mo = Pipeline([("tien", tien), ("hoi_quy", LinearRegression())]).fit(X_tr, y_tr)
        print("%-15s %4d đặc trưng | R2 huấn luyện %.4f | R2 kiểm tra %.4f"
              % (ten, mo.named_steps["tien"].transform(X_tr).shape[1],
                 r2_score(y_tr, mo.predict(X_tr)), r2_score(y_te, mo.predict(X_te))))


def cot_suy_ra_tu_muc_tieu(d: pd.DataFrame) -> None:
    """Dùng diem_cuoi_ky để dự đoán ket_qua cho kết quả đẹp giả."""
    print("Quy tắc ẩn trong dữ liệu, kiểm bằng bảng chéo:")
    print(pd.crosstab(d[MUC_TIEU] >= 4.0, d["ket_qua"]).to_string())
    y_nhan = (d["ket_qua"] == "Không đạt").astype("int64")
    for ten, cot_so in (("có diem_cuoi_ky", COT_SO + [MUC_TIEU]), ("không có", COT_SO)):
        X = d[cot_so + DAC_TRUNG_PL]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_nhan, test_size=0.2,
                                                  random_state=42, stratify=y_nhan)
        tien = ColumnTransformer([("so", StandardScaler(), cot_so),
                                  ("pl", OneHotEncoder(handle_unknown="ignore"), DAC_TRUNG_PL)])
        for nhan, mo_hinh in (("hồi quy logistic", LogisticRegression(max_iter=1000)),
                              ("cây sâu 3 tầng", DecisionTreeClassifier(max_depth=3,
                                                                        random_state=42))):
            mo = Pipeline([("tien", tien), ("phan_loai", mo_hinh)]).fit(X_tr, y_tr)
            du_doan = mo.predict(X_te)
            print("%-16s %-18s accuracy %.4f | ma trận nhầm lẫn %s"
                  % (ten, nhan, accuracy_score(y_te, du_doan),
                     confusion_matrix(y_te, du_doan).tolist()))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("-- Mã sinh viên làm đặc trưng --")
    ma_sinh_vien_lam_dac_trung(d)
    print("\n-- Cột suy ra trực tiếp từ biến mục tiêu --")
    cot_suy_ra_tu_muc_tieu(d)


if __name__ == "__main__":
    main()
