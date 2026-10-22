"""Hai hướng mở rộng, đo trên bài toán của dự án (mục 11.4.1 và 11.4.2).

Chạy từ gốc dự án:
    python code/81_so_huong_nang_cao.py

So hồi quy logistic của dự án với hai hướng hay được khuyên: máy tăng cường gradient
(hướng cổ điển) và một mạng nhiều tầng (hướng học sâu). Phép so dùng MƯỜI lần chia, vì
một lần chia không phân biệt được chênh lệch thật với dao động của phép chia tập
(mục 7.5.1). Thời gian khớp phụ thuộc máy nên in cả dải.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, TunedThresholdClassifierCV,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
CHUA_CO_O_TUAN_8 = ["diem_giua_ky"]
SO_LAN_CHIA = 10

UNG_VIEN = {
    "logistic": lambda: LogisticRegression(max_iter=1000),
    "tăng cường gradient": lambda: HistGradientBoostingClassifier(random_state=42),
    "mạng hai tầng ẩn": lambda: MLPClassifier(hidden_layer_sizes=(64, 32),
                                              max_iter=2000, random_state=42),
}


def bo_tien_theo_cot(bang_goc, cot):
    so = [c for c in cot if pd.api.types.is_numeric_dtype(bang_goc[c])]
    pl = [c for c in cot if c not in so]
    return ColumnTransformer([("so", StandardScaler(), so),
                              ("pl", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), pl)])


def main():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    cot = [c for c in X.columns if c not in CHUA_CO_O_TUAN_8]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    print("-- 11.4 một lần chia duy nhất, đúng lần chia của sách (random_state=42)")
    for ten, tao in UNG_VIEN.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X[cot], y, test_size=0.2,
                                                  random_state=42, stratify=y)
        ong = Pipeline([("tien", bo_tien_theo_cot(bang, cot)), ("mo_hinh", tao())])
        mot = TunedThresholdClassifierCV(ong, scoring="f1", cv=skf, refit=True,
                                         random_state=42).fit(X_tr, y_tr)
        nhan = mot.predict(X_te)
        print("  %-20s F1 %.4f | recall %.4f | AUC %.4f"
              % (ten, f1_score(y_te, nhan), recall_score(y_te, nhan),
                 roc_auc_score(y_te, mot.predict_proba(X_te)[:, 1])))

    diem = {ten: {"f1": [], "rec": [], "auc": [], "giay": []} for ten in UNG_VIEN}
    for hat in range(SO_LAN_CHIA):
        X_tr, X_te, y_tr, y_te = train_test_split(X[cot], y, test_size=0.2,
                                                  random_state=hat, stratify=y)
        for ten, tao in UNG_VIEN.items():
            ong = Pipeline([("tien", bo_tien_theo_cot(bang, cot)), ("mo_hinh", tao())])
            bat_dau = time.perf_counter()
            mo_hinh = TunedThresholdClassifierCV(ong, scoring="f1", cv=skf, refit=True,
                                                 random_state=42).fit(X_tr, y_tr)
            diem[ten]["giay"].append(time.perf_counter() - bat_dau)
            nhan = mo_hinh.predict(X_te)
            diem[ten]["f1"].append(f1_score(y_te, nhan))
            diem[ten]["rec"].append(recall_score(y_te, nhan))
            diem[ten]["auc"].append(roc_auc_score(y_te, mo_hinh.predict_proba(X_te)[:, 1]))

    print("-- 11.4 mười lần chia, cùng cột, cùng cách chọn ngưỡng")
    for ten, d in diem.items():
        print("  %-20s F1 trung bình %.4f | thời gian khớp %.1f tới %.1f giây"
              % (ten, np.mean(d["f1"]), min(d["giay"]), max(d["giay"])))
    # Đếm số lần DƯƠNG và số lần ÂM riêng, không lấy một trừ cái kia: hiệu bằng đúng 0
    # không thuộc nhóm nào, và nếu nó xảy ra ở mọi lần chia thì phép so không nói gì.
    for ten in ["tăng cường gradient", "mạng hai tầng ẩn"]:
        for do_do in ["f1", "rec", "auc"]:
            hieu = (np.array(diem[ten][do_do])
                    - np.array(diem["logistic"][do_do]))
            print("  %-20s hiệu %-4s dương %2d/%d · âm %2d/%d"
                  % (ten, do_do, (hieu > 0).sum(), SO_LAN_CHIA,
                     (hieu < 0).sum(), SO_LAN_CHIA))
            print("       trung bình %+.4f | từ %+.4f tới %+.4f"
                  % (hieu.mean(), hieu.min(), hieu.max()))
    # Thứ tự thời gian khớp: so THEO TỪNG LẦN CHIA, vì ba dải thời gian chồng nhau nên
    # dải không đỡ được một khẳng định về thứ tự.
    giu = sum(1 for i in range(SO_LAN_CHIA)
              if diem["logistic"]["giay"][i] < diem["tăng cường gradient"]["giay"][i]
              < diem["mạng hai tầng ẩn"]["giay"][i])
    print("  thứ tự thời gian khớp logistic < tăng cường gradient < mạng: giữ %d/%d lần chia"
          % (giu, SO_LAN_CHIA))


if __name__ == "__main__":
    main()
