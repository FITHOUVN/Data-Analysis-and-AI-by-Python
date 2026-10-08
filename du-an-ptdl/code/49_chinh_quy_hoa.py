"""Chính quy hóa L2, siêu tham số C, và vì sao chính quy hóa đòi cùng thang đo (mục 6.2.4)."""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
          "hinh_thuc_hoc", "thiet_bi_chinh"]
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
MUC_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def quet_C(tien, X_tr, X_te, y_tr, y_te) -> pd.DataFrame:
    """Sáu mức C, mỗi mức một dòng: độ lớn hệ số và bốn độ đo."""
    dong = []
    for C in MUC_C:
        mo = Pipeline([("tien", clone(tien)),
                       ("mo_hinh", LogisticRegression(C=C, max_iter=1000))])
        mo.fit(X_tr, y_tr)
        w = mo.named_steps["mo_hinh"].coef_[0]
        nh = mo.predict(X_te)
        dong.append({"C": C,
                     "tong_tri_tuyet_doi": round(float(np.abs(w).sum()), 3),
                     "he_so_lon_nhat": round(float(np.abs(w).max()), 4),
                     "acc_huan_luyen": round(accuracy_score(y_tr, mo.predict(X_tr)), 4),
                     "acc_kiem_tra": round(accuracy_score(y_te, nh), 4),
                     "recall": round(recall_score(y_te, nh), 4),
                     "F1": round(f1_score(y_te, nh), 4),
                     "AUC": round(roc_auc_score(y_te, mo.predict_proba(X_te)[:, 1]), 4)})
    return pd.DataFrame(dong)


def tien_xu_ly_tho() -> ColumnTransformer:
    """Bộ tiền xử lý giữ nguyên thang gốc tám cột số, chỉ mã hóa cột phân loại."""
    return ColumnTransformer([
        ("so", "passthrough", COT_SO),
        ("pl", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COT_PL)])


def ba_con_so(nhan_goc, nhan_moi, p_goc, p_moi, y_te) -> str:
    """Ba con số bắt buộc của mọi so sánh 'có đổi hay không', bản cho phân loại."""
    return ("đổi %d/%d nhãn | lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
            % (int((nhan_goc != nhan_moi).sum()), len(nhan_goc),
               float(np.abs(p_goc - p_moi).max()),
               f1_score(y_te, nhan_moi) - f1_score(y_te, nhan_goc)))


def bo_chuan_hoa_z(tien, X_tr, X_te, y_tr, y_te) -> None:
    """So bản có chuẩn hóa z với bản thang gốc, và bắt cảnh báo không hội tụ."""
    chuan = Pipeline([("tien", clone(tien)),
                      ("mo_hinh", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
    with warnings.catch_warnings(record=True) as ghi:
        warnings.simplefilter("always")
        tho = Pipeline([("tien", tien_xu_ly_tho()),
                        ("mo_hinh", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
        for w in ghi:
            print("  [%s] %s" % (w.category.__name__, str(w.message).splitlines()[0]))
    print("  số vòng lặp: thang gốc %d | đã chuẩn hóa z %d"
          % (tho.named_steps["mo_hinh"].n_iter_[0],
             chuan.named_steps["mo_hinh"].n_iter_[0]))
    p_c, p_t = chuan.predict_proba(X_te)[:, 1], tho.predict_proba(X_te)[:, 1]
    print("  ", ba_con_so(chuan.predict(X_te), tho.predict(X_te), p_c, p_t, y_te))
    print("  bản thang gốc: accuracy %.4f | recall %.4f | F1 %.4f | AUC %.4f"
          % (accuracy_score(y_te, tho.predict(X_te)),
             recall_score(y_te, tho.predict(X_te)),
             f1_score(y_te, tho.predict(X_te)), roc_auc_score(y_te, p_t)))
    print("  số ca xác suất lệch quá 0,01: %d/%d"
          % (int((np.abs(p_c - p_t) > 0.01).sum()), len(p_c)))
    hs = pd.Series(tho.named_steps["mo_hinh"].coef_[0],
                   index=tho.named_steps["tien"].get_feature_names_out())
    print("  bốn hệ số cột số lớn nhất của bản thang gốc:")
    print(hs[["so__" + c for c in COT_SO]]
          .sort_values(key=abs, ascending=False).head(4).round(6).to_string())


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Sáu mức chính quy hóa --")
    print(quet_C(tien, X_tr, X_te, y_tr, y_te).to_string(index=False))
    print("\n-- Bỏ chuẩn hóa z khỏi bộ tiền xử lý --")
    bo_chuan_hoa_z(tien, X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    main()
