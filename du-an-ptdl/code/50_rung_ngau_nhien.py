"""Từ một cây đến một rừng, và bốn siêu tham số của rừng (mục 6.3.1, 6.3.2)."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
BA_MO_HINH = (
    ("cây không giới hạn", DecisionTreeClassifier(random_state=42)),
    ("cây sâu 5 tầng", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("rừng 200 cây", RandomForestClassifier(n_estimators=200, random_state=42)),
)
BON_CAU_HINH = (
    ("rừng 200 cây", {}),
    ("rừng max_depth=8", {"max_depth": 8}),
    ("rừng min_samples_leaf=20", {"min_samples_leaf": 20}),
    ("rừng max_features=None", {"max_features": None}),
)


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def mot_luot(tien, X_tr, X_te, y_tr, y_te, ten, goc) -> dict:
    """Khớp một mô hình rồi trả về một dòng kết quả, đo trên cả hai tập."""
    mo = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(goc))]).fit(X_tr, y_tr)
    nh = mo.predict(X_te)
    return {"mo_hinh": ten,
            "acc_huan_luyen": round(accuracy_score(y_tr, mo.predict(X_tr)), 4),
            "acc_kiem_tra": round(accuracy_score(y_te, nh), 4),
            "precision": round(precision_score(y_te, nh), 4),
            "recall": round(recall_score(y_te, nh), 4),
            "F1": round(f1_score(y_te, nh), 4),
            "AUC": round(roc_auc_score(y_te, mo.predict_proba(X_te)[:, 1]), 4)}


def mot_cay_so_voi_rung(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Hai cây lẻ và một rừng, cùng tập dữ liệu, cùng bộ tiền xử lý."""
    dong = [mot_luot(tien, X_tr, X_te, y_tr, y_te, ten, goc)
            for ten, goc in BA_MO_HINH]
    print(pd.DataFrame(dong).to_string(index=False))
    for ten, goc in BA_MO_HINH:
        mo = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(goc))]).fit(X_tr, y_tr)
        print("  %-20s ma trận nhầm lẫn %s"
              % (ten, confusion_matrix(y_te, mo.predict(X_te)).tolist()))


def trong_long_rung(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Đo từng cây trong rừng để thấy phép lấy trung bình làm được gì."""
    mo = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", RandomForestClassifier(n_estimators=200,
                                                      random_state=42))]).fit(X_tr, y_tr)
    # Bộ tiền xử lý trả về DataFrame, còn các cây bên trong rừng được khớp trên mảng
    # numpy, nên phải bỏ tên cột đi; không bỏ thì mỗi lời gọi predict in một UserWarning.
    Z_te = mo.named_steps["tien"].transform(X_te).to_numpy()
    cay = mo.named_steps["mo_hinh"].estimators_
    f1_le = np.array([f1_score(y_te, c.predict(Z_te)) for c in cay])
    f1_rung = f1_score(y_te, mo.predict(X_te))
    print("F1 của 200 cây lẻ: nhỏ nhất %.4f | lớn nhất %.4f | trung bình %.4f"
          % (f1_le.min(), f1_le.max(), f1_le.mean()))
    print("F1 của cả rừng: %.4f | số cây lẻ vượt được cả rừng: %d/200"
          % (f1_rung, int((f1_le > f1_rung).sum())))
    print("Số giá trị xác suất khác nhau: cả rừng %d | một cây lẻ %d"
          % (len(np.unique(mo.predict_proba(X_te)[:, 1])),
             len(np.unique(cay[0].predict_proba(Z_te)[:, 1]))))


def quet_so_cay(tien, X_tr, X_te, y_tr, y_te) -> pd.DataFrame:
    """Năm mức số cây, kèm thời gian khớp."""
    dong = []
    for k in (10, 50, 100, 200, 500):
        t0 = time.perf_counter()
        mo = Pipeline([("tien", clone(tien)),
                       ("mo_hinh", RandomForestClassifier(n_estimators=k,
                                                          random_state=42))])
        mo.fit(X_tr, y_tr)
        giay = time.perf_counter() - t0
        nh = mo.predict(X_te)
        dong.append({"n_estimators": k,
                     "accuracy": round(accuracy_score(y_te, nh), 4),
                     "precision": round(precision_score(y_te, nh), 4),
                     "recall": round(recall_score(y_te, nh), 4),
                     "F1": round(f1_score(y_te, nh), 4),
                     "AUC": round(roc_auc_score(y_te, mo.predict_proba(X_te)[:, 1]), 4),
                     "giay_khop": round(giay, 2)})
    return pd.DataFrame(dong)


def quet_cau_hinh(tien, X_tr, X_te, y_tr, y_te) -> pd.DataFrame:
    """Rừng có cần chặn quá khớp như cây lẻ hay không: bốn cấu hình."""
    dong = []
    for ten, tham_so in BON_CAU_HINH:
        goc = RandomForestClassifier(n_estimators=200, random_state=42, **tham_so)
        dong.append(mot_luot(tien, X_tr, X_te, y_tr, y_te, ten, goc))
    return pd.DataFrame(dong)


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Hai cây lẻ so với một rừng --")
    mot_cay_so_voi_rung(tien, X_tr, X_te, y_tr, y_te)
    print("\n-- Trong lòng rừng: 200 cây lẻ --")
    trong_long_rung(tien, X_tr, X_te, y_tr, y_te)
    print("\n-- Số cây --")
    print(quet_so_cay(tien, X_tr, X_te, y_tr, y_te).to_string(index=False))
    print("\n-- Chặn quá khớp cho rừng, và tắt nguồn ngẫu nhiên thứ hai --")
    print(quet_cau_hinh(tien, X_tr, X_te, y_tr, y_te).to_string(index=False))


if __name__ == "__main__":
    main()
