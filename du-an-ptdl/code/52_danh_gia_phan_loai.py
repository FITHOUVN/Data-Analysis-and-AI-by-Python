"""Bốn độ đo, ma trận nhầm lẫn, AUC, và phép so hai mô hình theo hậu quả (mục 6.4)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
MO_HINH = {
    "mốc luôn đoán Đạt": DummyClassifier(strategy="most_frequent"),
    "hồi quy logistic": LogisticRegression(max_iter=1000),
    "rừng ngẫu nhiên": RandomForestClassifier(n_estimators=200, random_state=42),
}


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def khop_tat_ca(tien, X_tr, X_te, y_tr, y_te):
    """Khớp ba mô hình trên cùng tập huấn luyện, đo trên cùng tập kiểm tra."""
    dong, nhan, xac_suat = [], {}, {}
    for ten, goc in MO_HINH.items():
        mo = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(goc))]).fit(X_tr, y_tr)
        nh = mo.predict(X_te)
        ps = mo.predict_proba(X_te)[:, 1]
        nhan[ten], xac_suat[ten] = nh, ps
        tn, fp, fn, tp = confusion_matrix(y_te, nh).ravel()
        dong.append({"mo_hinh": ten, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "accuracy": round(accuracy_score(y_te, nh), 4),
                     "precision": round(precision_score(y_te, nh, zero_division=0), 4),
                     "recall": round(recall_score(y_te, nh), 4),
                     "F1": round(f1_score(y_te, nh), 4),
                     "AUC": round(roc_auc_score(y_te, ps), 4)})
    return pd.DataFrame(dong), nhan, xac_suat


def ba_con_so(nhan, xac_suat, y_te, chuan="hồi quy logistic") -> str:
    """Ba con số bắt buộc khi so một mô hình với mô hình chuẩn."""
    khac = [t for t in nhan if t != chuan and t in MO_HINH][-1]
    return ("đổi %d/%d nhãn | lệch xác suất lớn nhất %.4f"
            " | lệch F1 %+.4f | lệch AUC %+.4f"
            % (int((nhan[chuan] != nhan[khac]).sum()), len(y_te),
               float(np.abs(xac_suat[chuan] - xac_suat[khac]).max()),
               f1_score(y_te, nhan[khac]) - f1_score(y_te, nhan[chuan]),
               roc_auc_score(y_te, xac_suat[khac])
               - roc_auc_score(y_te, xac_suat[chuan])))


def ai_dung_khi_hai_mo_hinh_lech(nhan, y_te) -> None:
    """Trong số ca hai mô hình đáp khác nhau, mỗi bên đúng bao nhiêu."""
    nh_lg, nh_rf = nhan["hồi quy logistic"], nhan["rừng ngẫu nhiên"]
    thuc = y_te.to_numpy()
    print("chỉ hồi quy logistic báo dương: %d ca, đúng %d ca"
          % (int(((nh_lg == 1) & (nh_rf == 0)).sum()),
             int(thuc[(nh_lg == 1) & (nh_rf == 0)].sum())))
    print("chỉ rừng ngẫu nhiên báo dương: %d ca, đúng %d ca"
          % (int(((nh_lg == 0) & (nh_rf == 1)).sum()),
             int(thuc[(nh_lg == 0) & (nh_rf == 1)].sum())))


def hieu_qua_nhieu_lan_chia(X, y, tien, so_lan=10) -> None:
    """Đo HIỆU bốn độ đo qua từng lần chia: hiệu đổi dấu hay không mới là câu trả lời."""
    cot = {k: [] for k in ("recall", "precision", "F1", "AUC", "FN", "FP")}
    for hat in range(so_lan):
        A, B, a, b = train_test_split(X, y, test_size=0.2, random_state=hat, stratify=y)
        diem = {}
        for ten in ("hồi quy logistic", "rừng ngẫu nhiên"):
            mo = Pipeline([("tien", clone(tien)),
                           ("mo_hinh", clone(MO_HINH[ten]))]).fit(A, a)
            nh, ps = mo.predict(B), mo.predict_proba(B)[:, 1]
            ma = confusion_matrix(b, nh)
            diem[ten] = {"recall": recall_score(b, nh),
                         "precision": precision_score(b, nh),
                         "F1": f1_score(b, nh), "AUC": roc_auc_score(b, ps),
                         "FN": int(ma[1, 0]), "FP": int(ma[0, 1])}
        for k in cot:
            cot[k].append(diem["hồi quy logistic"][k] - diem["rừng ngẫu nhiên"][k])
        print("  hạt %d: logistic FN %2d FP %2d recall %.4f | rừng FN %2d FP %2d"
              " recall %.4f"
              % (hat, diem["hồi quy logistic"]["FN"], diem["hồi quy logistic"]["FP"],
                 diem["hồi quy logistic"]["recall"], diem["rừng ngẫu nhiên"]["FN"],
                 diem["rừng ngẫu nhiên"]["FP"], diem["rừng ngẫu nhiên"]["recall"]))
    for k in ("recall", "precision", "F1", "AUC", "FN", "FP"):
        h = np.array(cot[k], dtype=float)
        print("hiệu %-9s dương %2d/%d | %+.4f … %+.4f | trung bình %+.4f"
              % (k, int((h > 0).sum()), so_lan, h.min(), h.max(), h.mean()))


def chinh_nguong_cho_rung(xac_suat, nhan, y_te) -> None:
    """Hạ ngưỡng của rừng cho tới khi nó đạt đúng recall của hồi quy logistic."""
    muc = recall_score(y_te, nhan["hồi quy logistic"])
    ps = xac_suat["rừng ngẫu nhiên"]
    for nguong in np.arange(0.99, 0.0, -0.005):
        nh = (ps >= nguong).astype(int)
        if recall_score(y_te, nh) >= muc:
            tn, fp, fn, tp = confusion_matrix(y_te, nh).ravel()
            print("Rừng ở ngưỡng %.3f: recall %.4f | precision %.4f | F1 %.4f"
                  " | TP %d FP %d FN %d"
                  % (nguong, recall_score(y_te, nh), precision_score(y_te, nh),
                     f1_score(y_te, nh), tp, fp, fn))
            return


def diem_tren_duong_roc(xac_suat, y_te) -> None:
    """Hai điểm đáng đọc của mỗi đường ROC: đỉnh Youden và mức FPR 0,10."""
    for ten in ("hồi quy logistic", "rừng ngẫu nhiên"):
        fpr, tpr, nguong = roc_curve(y_te, xac_suat[ten])
        i = int(np.argmax(tpr - fpr))
        j = int(np.searchsorted(fpr, 0.10, side="right") - 1)
        print("%-17s AUC %.4f | điểm xa đường chéo nhất: ngưỡng %.4f, TPR %.4f,"
              " FPR %.4f | tại FPR <= 0,10: TPR %.4f"
              % (ten, roc_auc_score(y_te, xac_suat[ten]), nguong[i], tpr[i], fpr[i],
                 tpr[j]))


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    bang, nhan, xac_suat = khop_tat_ca(tien, X_tr, X_te, y_tr, y_te)
    print(bang.to_string(index=False))
    print("\n-- Báo cáo phân loại của hồi quy logistic --")
    print(classification_report(y_te, nhan["hồi quy logistic"],
                                target_names=["Đạt", "Không đạt"], digits=4))
    print("-- Ba con số: rừng ngẫu nhiên so với hồi quy logistic --")
    print(ba_con_so(nhan, xac_suat, y_te))
    ai_dung_khi_hai_mo_hinh_lech(nhan, y_te)
    print("\n-- Hai điểm đáng đọc trên mỗi đường ROC --")
    diem_tren_duong_roc(xac_suat, y_te)
    print("\n-- Rừng chỉnh ngưỡng để đạt recall của hồi quy logistic --")
    chinh_nguong_cho_rung(xac_suat, nhan, y_te)
    print("\n-- Hiệu (hồi quy logistic trừ rừng) trên mười lần chia phân tầng --")
    hieu_qua_nhieu_lan_chia(X, y, tien)
    bang.to_csv(VAO / "ch06-so-sanh-mo-hinh.csv", index=False, encoding="utf-8")
    print("\nĐã ghi bảng so sánh vào data/processed/ch06-so-sanh-mo-hinh.csv")


if __name__ == "__main__":
    main()
