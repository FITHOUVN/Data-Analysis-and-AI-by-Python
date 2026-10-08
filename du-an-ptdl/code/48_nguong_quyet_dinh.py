"""Từ xác suất sang nhãn: ngưỡng quyết định và cái giá của từng mức (mục 6.2.3)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
MUC_NGUONG = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def bang_theo_nguong(y_te, xac_suat) -> pd.DataFrame:
    """Bốn ô của ma trận nhầm lẫn và ba độ đo, cho từng mức ngưỡng."""
    dong = []
    for nguong in MUC_NGUONG:
        nhan = (xac_suat >= nguong).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, nhan).ravel()
        dong.append({"nguong": nguong, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(precision_score(y_te, nhan,
                                                        zero_division=0), 4),
                     "recall": round(recall_score(y_te, nhan), 4),
                     "F1": round(f1_score(y_te, nhan), 4),
                     "accuracy": round(accuracy_score(y_te, nhan), 4)})
    return pd.DataFrame(dong)


def nguong_toi_uu_F1(y_te, xac_suat) -> None:
    """Ngưỡng cho F1 lớn nhất, lấy từ đường precision-recall."""
    pre, rec, nguong = precision_recall_curve(y_te, xac_suat)
    f1 = 2 * pre * rec / np.clip(pre + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    print("Số ngưỡng ứng viên: %d | F1 lớn nhất %.4f tại ngưỡng %.4f"
          " (precision %.4f, recall %.4f)"
          % (len(nguong), f1[i], nguong[i], pre[i], rec[i]))


def nguong_dat_recall(y_te, xac_suat, muc=0.90) -> None:
    """Ngưỡng cao nhất còn đạt mức recall yêu cầu, và cái giá kèm theo."""
    chon = None
    for nguong in np.arange(0.99, 0.0, -0.01):
        nhan = (xac_suat >= nguong).astype(int)
        if recall_score(y_te, nhan) >= muc:
            chon = (nguong, nhan)
            break
    nguong, nhan = chon
    tn, fp, fn, tp = confusion_matrix(y_te, nhan).ravel()
    print("Ngưỡng cao nhất còn đạt recall >= %.2f: %.2f"
          " | recall %.4f | precision %.4f | TP %d FP %d FN %d"
          % (muc, nguong, recall_score(y_te, nhan),
             precision_score(y_te, nhan), tp, fp, fn))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    mo = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
    xac_suat = mo.predict_proba(X_te)[:, 1]
    print("predict_proba trả về mảng", mo.predict_proba(X_te).shape,
          "| hai lớp:", mo.classes_.tolist())
    print("Xác suất lớp dương: nhỏ nhất %.6f | lớn nhất %.6f | trung bình %.4f"
          % (xac_suat.min(), xac_suat.max(), xac_suat.mean()))
    print("Số ca xác suất nằm trong [0,40; 0,60]:",
          int(((xac_suat >= 0.4) & (xac_suat <= 0.6)).sum()))
    print("hồi quy logistic: predict trùng khít (predict_proba >= 0,5):",
          bool((mo.predict(X_te) == (xac_suat >= 0.5).astype(int)).all()))
    print("Năm xác suất đầu:", np.round(xac_suat[:5], 4).tolist())
    print("Năm nhãn thực đầu:", y_te.to_numpy()[:5].tolist())
    print("\n-- Bảy mức ngưỡng trên cùng một vectơ xác suất --")
    print(bang_theo_nguong(y_te, xac_suat).to_string(index=False))
    print()
    nguong_toi_uu_F1(y_te, xac_suat)
    nguong_dat_recall(y_te, xac_suat)


if __name__ == "__main__":
    main()
