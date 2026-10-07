"""Ba giới hạn của hồi quy tuyến tính: phi tuyến, ngoại lai, dự đoán ngoài miền (mục 5.2.4)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def ba_con_so(goc, moi, y_te) -> str:
    """Ba con số bắt buộc của mọi so sánh 'có đổi hay không'."""
    doi = int((np.round(goc, 4) != np.round(moi, 4)).sum())
    return ("đổi %d/%d dự đoán | lệch lớn nhất %.4f | lệch R2 %+.4f"
            % (doi, len(goc), float(np.abs(goc - moi).max()),
               r2_score(y_te, moi) - r2_score(y_te, goc)))


def them_binh_phuong(X: pd.DataFrame) -> pd.DataFrame:
    """Thêm tám cột bình phương của tám cột số, giữ nguyên các cột còn lại."""
    ra = X.copy()
    for c in COT_SO:
        ra[c + "_bp"] = X[c] ** 2
    return ra


def tien_xu_ly_mo_rong() -> ColumnTransformer:
    """Bộ tiền xử lý cho 16 cột số (8 cột gốc và 8 cột bình phương)."""
    bo = ColumnTransformer([
        ("so", StandardScaler(), COT_SO + [c + "_bp" for c in COT_SO]),
        ("pl", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DAC_TRUNG_PL)])
    bo.set_output(transform="pandas")
    return bo


def phi_tuyen(X_tr, X_te, y_tr, y_te, du_goc) -> None:
    """Quan hệ cong: thêm số hạng bình phương rồi đo lại bằng chính mô hình tuyến tính."""
    mo = Pipeline([("tien", tien_xu_ly_mo_rong()), ("mo_hinh", LinearRegression())])
    mo.fit(them_binh_phuong(X_tr), y_tr)
    d = mo.predict(them_binh_phuong(X_te))
    print("Thêm 8 cột bình phương: MAE %.3f | RMSE %.3f | R2 %.4f"
          % (mean_absolute_error(y_te, d), root_mean_squared_error(y_te, d),
             r2_score(y_te, d)))
    print("  ", ba_con_so(du_goc, d, y_te))


def mot_dong_ngoai_lai(X_tr, X_te, y_tr, y_te, tien, du_goc) -> None:
    """Thêm đúng một dòng có giá trị canh 99999 vào tập huấn luyện."""
    X_xau = pd.concat([X_tr, X_tr.iloc[[0]]], ignore_index=True)
    X_xau.loc[len(X_xau) - 1, "tong_thoi_luong_phut"] = 99999
    y_xau = pd.concat([y_tr, pd.Series([0.0])], ignore_index=True)
    mo = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())])
    mo.fit(X_xau, y_xau)
    hs = pd.Series(mo.named_steps["mo_hinh"].coef_,
                   index=mo.named_steps["tien"].get_feature_names_out())
    print("Hệ số của tong_thoi_luong_phut sau khi thêm một dòng: %+.4f"
          % hs["so__tong_thoi_luong_phut"])
    print("  ", ba_con_so(du_goc, mo.predict(X_te), y_te))


def chan_ve_mien(y_te, du_goc) -> None:
    """Chặn dự đoán về miền hợp lệ của biến mục tiêu, tức thang điểm từ 0,0 tới 10,0."""
    ngoai = (du_goc < 0.0) | (du_goc > 10.0)
    print("Dự đoán rơi ngoài thang điểm: %d/%d | nhỏ nhất %.2f | lớn nhất %.2f"
          % (int(ngoai.sum()), len(du_goc), du_goc.min(), du_goc.max()))
    chan = np.clip(du_goc, 0.0, 10.0)
    print("Sau khi chặn: MAE %.4f (trước %.4f) | R2 %.4f (trước %.4f)"
          % (mean_absolute_error(y_te, chan), mean_absolute_error(y_te, du_goc),
             r2_score(y_te, chan), r2_score(y_te, du_goc)))
    print("  ", ba_con_so(du_goc, chan, y_te))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    goc = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())]).fit(X_tr, y_tr)
    du_goc = goc.predict(X_te)              # vectơ dự đoán gốc, dùng lại ở cả ba phép thử
    print("Mô hình tuyến tính gốc: R2 %.4f\n" % r2_score(y_te, du_goc))
    print("-- Giới hạn 1: quan hệ phi tuyến --")
    phi_tuyen(X_tr, X_te, y_tr, y_te, du_goc)
    print("\n-- Giới hạn 2: một dòng ngoại lai trong tập huấn luyện --")
    mot_dong_ngoai_lai(X_tr, X_te, y_tr, y_te, tien, du_goc)
    print("\n-- Giới hạn 3: dự đoán rơi ngoài miền hợp lệ --")
    chan_ve_mien(y_te, du_goc)


if __name__ == "__main__":
    main()
