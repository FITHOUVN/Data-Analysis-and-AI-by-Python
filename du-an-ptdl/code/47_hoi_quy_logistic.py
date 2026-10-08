"""Vì sao không dùng hồi quy tuyến tính, và huấn luyện hồi quy logistic (mục 6.2.1, 6.2.2)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
          "hinh_thuc_hoc", "thiet_bi_chinh"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def dung_tuyen_tinh(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Khớp hồi quy tuyến tính lên nhãn 0/1 để thấy nó sai ở đâu."""
    mo = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())])
    mo.fit(X_tr, y_tr)
    d = mo.predict(X_te)
    ngoai = (d < 0.0) | (d > 1.0)
    print("Đầu ra hồi quy tuyến tính: nhỏ nhất %.4f | lớn nhất %.4f" % (d.min(), d.max()))
    print("Số giá trị rơi ngoài [0, 1]: %d/%d (dưới 0: %d | trên 1: %d)"
          % (int(ngoai.sum()), len(d), int((d < 0).sum()), int((d > 1).sum())))
    nhan = (d >= 0.5).astype(int)
    print("Cắt ở 0,5: accuracy %.4f | precision %.4f | recall %.4f | F1 %.4f"
          % (accuracy_score(y_te, nhan), precision_score(y_te, nhan),
             recall_score(y_te, nhan), f1_score(y_te, nhan)))
    print("Ma trận nhầm lẫn:", confusion_matrix(y_te, nhan).tolist())


def khop_logistic(tien, X_tr, y_tr) -> Pipeline:
    """Hồi quy logistic trong một Pipeline hai chặng, lời gọi fit duy nhất."""
    mo = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", LogisticRegression(max_iter=1000))])
    mo.fit(X_tr, y_tr)
    return mo


def do_theo_cap(mo, X_tr, X_te, y_tr, y_te) -> pd.DataFrame:
    """Năm độ đo trên cả hai tập, để đọc theo cặp như Chương 5 đã chốt."""
    dong = []
    for nhan, X, y in (("huấn luyện", X_tr, y_tr), ("kiểm tra", X_te, y_te)):
        nh = mo.predict(X)
        dong.append({"tập": nhan,
                     "accuracy": round(accuracy_score(y, nh), 4),
                     "precision": round(precision_score(y, nh), 4),
                     "recall": round(recall_score(y, nh), 4),
                     "F1": round(f1_score(y, nh), 4),
                     "AUC": round(roc_auc_score(y, mo.predict_proba(X)[:, 1]), 4)})
    return pd.DataFrame(dong)


def bang_he_so(mo) -> pd.DataFrame:
    """Hệ số, tỉ số cược tương ứng, xếp theo giá trị tuyệt đối giảm dần."""
    ten = mo.named_steps["tien"].get_feature_names_out()
    hs = pd.Series(mo.named_steps["mo_hinh"].coef_[0], index=ten)
    hs = hs.sort_values(key=abs, ascending=False)
    return pd.DataFrame({"he_so": hs.round(4),
                         "ti_so_cuoc": np.exp(hs).round(4)})


def doc_he_so_chan(mo, X_tr) -> None:
    """Hệ số chặn một mình không là xác suất của ai: phải cộng hệ số của sáu mức cụ thể."""
    ten = mo.named_steps["tien"].get_feature_names_out()
    hs = pd.Series(mo.named_steps["mo_hinh"].coef_[0], index=ten)
    b = float(mo.named_steps["mo_hinh"].intercept_[0])
    muc = {c: X_tr[c].mode()[0] for c in COT_PL}
    tong = float(sum(hs["pl__%s_%s" % (c, m)] for c, m in muc.items()))
    z = b + tong
    print("Hệ số chặn %.4f -> tỉ số cược %.4f; nó ứng với điểm mà cả %d cột một nóng bằng 0"
          % (b, np.exp(b), sum(1 for t in ten if t.startswith("pl__"))))
    print("Sáu mức hay gặp nhất: %s" % ", ".join(muc.values()))
    print("Tổng sáu hệ số %.4f -> z = %.4f -> xác suất của 'sinh viên trung bình' %.4f"
          % (tong, z, 1 / (1 + np.exp(-z))))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Hồi quy tuyến tính trên nhãn 0/1 --")
    dung_tuyen_tinh(tien, X_tr, X_te, y_tr, y_te)
    mo = khop_logistic(tien, X_tr, y_tr)
    print("\n-- Hồi quy logistic, năm độ đo trên cả hai tập --")
    print(do_theo_cap(mo, X_tr, X_te, y_tr, y_te).to_string(index=False))
    print("Số vòng lặp tới khi hội tụ:", mo.named_steps["mo_hinh"].n_iter_[0])
    print("Ma trận nhầm lẫn trên tập kiểm tra:",
          confusion_matrix(y_te, mo.predict(X_te)).tolist())
    print("\n-- Hệ số chặn và mười hệ số lớn nhất --")
    print("Hệ số chặn: %.4f | số hệ số: %d"
          % (mo.named_steps["mo_hinh"].intercept_[0],
             len(mo.named_steps["mo_hinh"].coef_[0])))
    print(bang_he_so(mo).head(10).to_string())
    doc_he_so_chan(mo, X_tr)
    joblib.dump(mo, THU_MUC_MO_HINH / "ch06-hoi-quy-logistic.joblib")
    print("\nĐã lưu mô hình vào models/ch06-hoi-quy-logistic.joblib")


if __name__ == "__main__":
    main()
