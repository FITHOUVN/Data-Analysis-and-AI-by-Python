"""Bảng so sánh cùng điều kiện, đường cong học, và chốt mô hình cuối cùng (mục 7.5)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as OngLayMau
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, TunedThresholdClassifierCV,
                                     cross_validate, learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
BON_COT_YEU = ["nganh", "nhom_tuoi", "so_lan_dien_dan", "gioi_tinh"]
GIA_FP, GIA_FN = 0.25, 45.0


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def bo_tien_theo_cot(bang_goc, cot):
    so = [c for c in cot if pd.api.types.is_numeric_dtype(bang_goc[c])]
    pl = [c for c in cot if c not in so]
    return ColumnTransformer([("so", StandardScaler(), so),
                              ("pl", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), pl)])


def sau_cau_hinh(X, tien, skf):
    """Sáu cấu hình đáng so, mỗi cấu hình là một pipeline hoàn chỉnh."""
    lg = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", LogisticRegression(max_iter=1000))])
    rung = Pipeline([("tien", clone(tien)),
                     ("mo_hinh", RandomForestClassifier(n_estimators=200,
                                                        random_state=42))])
    cot_10 = [c for c in X.columns if c not in BON_COT_YEU]
    rung10 = Pipeline([("tien", bo_tien_theo_cot(X, cot_10)),
                       ("mo_hinh", RandomForestClassifier(n_estimators=200,
                                                          random_state=42))])
    return {
        "A. logistic, ngưỡng 0,5": (lg, list(X.columns)),
        "B. logistic + ngưỡng chọn bằng XTC":
            (TunedThresholdClassifierCV(lg, scoring="f1", cv=skf, random_state=42),
             list(X.columns)),
        "C. logistic + SMOTE":
            (OngLayMau([("tien", clone(tien)), ("lay_mau", SMOTE(random_state=42)),
                        ("mo_hinh", LogisticRegression(max_iter=1000))]),
             list(X.columns)),
        "D. rừng 200 cây, ngưỡng 0,5": (rung, list(X.columns)),
        "E. rừng + ngưỡng chọn bằng XTC":
            (TunedThresholdClassifierCV(rung, scoring="f1", cv=skf, random_state=42),
             list(X.columns)),
        "F. rừng 10 cột + ngưỡng chọn bằng XTC":
            (TunedThresholdClassifierCV(rung10, scoring="f1", cv=skf, random_state=42),
             cot_10),
    }


def bang_so_sanh(cach, X_tr, y_tr, skf):
    """Mọi cấu hình đi qua ĐÚNG MỘT phép xác thực chéo, nên các dòng so được với nhau."""
    diem = {}
    for ten, (p, cot) in cach.items():
        r = cross_validate(clone(p), X_tr[cot], y_tr, cv=skf,
                           scoring=("f1", "recall", "precision", "roc_auc"))
        diem[ten] = {k: r["test_" + k] for k in ("f1", "recall", "precision", "roc_auc")}
        print("  %s" % ten)
        print("    F1 %.4f (sd %.4f) | rec %.4f | pre %.4f | AUC %.4f"
              % (r["test_f1"].mean(), r["test_f1"].std(ddof=1),
                 r["test_recall"].mean(), r["test_precision"].mean(),
                 r["test_roc_auc"].mean()))
    return diem


def hieu_ghep_cap(diem, moc, do_do="f1"):
    """So theo hiệu từng phần, không so hai khoảng trung bình ± độ lệch chuẩn."""
    print("  hiệu %s theo từng phần, %s trừ từng cấu hình còn lại:" % (do_do, moc[:2]))
    for ten in diem:
        if ten == moc:
            continue
        h = diem[moc][do_do] - diem[ten][do_do]
        print("    trừ %-4s dương %d/5 | %s | trung bình %+.4f"
              % (ten[:2], int((h > 0).sum()),
                 " · ".join("%+.4f" % v for v in h), h.mean()))


def duong_cong_hoc(p, X_tr, y_tr, skf, ten):
    """F1 theo cỡ tập huấn luyện: dữ liệu để quyết định có nên đi thu thêm dòng hay không."""
    co, diem_tr, diem_v = learning_curve(clone(p), X_tr, y_tr, cv=skf, scoring="f1",
                                         train_sizes=np.linspace(0.1, 1.0, 6))
    print("  %s" % ten)
    for n, a, b in zip(co, diem_tr.mean(axis=1), diem_v.mean(axis=1)):
        print("    n=%-5d F1 trên phần dùng để khớp %.4f" % (n, a))
        print("          trên phần dùng để đo %.4f | khoảng cách %.4f"
              % (b, a - b))


def chot_mo_hinh(X_tr, y_tr, X_te, y_te, tien, skf):
    """Khớp cấu hình đã chọn trên cả tập huấn luyện rồi đo ĐÚNG MỘT LẦN."""
    lg = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", LogisticRegression(max_iter=1000))])
    cuoi = TunedThresholdClassifierCV(lg, scoring="f1", cv=skf, refit=True,
                                      random_state=42).fit(X_tr, y_tr)
    nh, ps = cuoi.predict(X_te), cuoi.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, nh).ravel()
    print("  ngưỡng %.4f | acc %.4f | pre %.4f | rec %.4f"
          % (cuoi.best_threshold_, accuracy_score(y_te, nh),
             precision_score(y_te, nh), recall_score(y_te, nh)))
    print("    F1 %.4f | AUC %.4f | ma %s"
          % (f1_score(y_te, nh), roc_auc_score(y_te, ps),
             confusion_matrix(y_te, nh).tolist()))
    goc = clone(lg).fit(X_tr, y_tr)
    nh6, ps6 = goc.predict(X_te), goc.predict_proba(X_te)[:, 1]
    tn6, fp6, fn6, tp6 = confusion_matrix(y_te, nh6).ravel()
    print("  ba con số so với mô hình Chương 6: đổi %d/%d nhãn"
          % (int((nh != nh6).sum()), len(y_te)))
    print("    lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
          % (float(np.abs(ps - ps6).max()),
             f1_score(y_te, nh) - f1_score(y_te, nh6)))
    print("  bỏ sót %d thay vì %d | báo động sai %d thay vì %d"
          % (fn, fn6, fp, fp6))
    print("    tổn thất %.2f giờ thay vì %.2f giờ"
          % (fp * GIA_FP + fn * GIA_FN, fp6 * GIA_FP + fn6 * GIA_FN))
    return cuoi


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cach = sau_cau_hinh(X, tien, skf)

    print("-- 7.5.1 sáu cấu hình, cùng StratifiedKFold(5, shuffle=True, random_state=42)"
          " trên 2.409 dòng --")
    diem = bang_so_sanh(cach, X_tr, y_tr, skf)
    print()
    hieu_ghep_cap(diem, "B. logistic + ngưỡng chọn bằng XTC", "f1")
    print()
    hieu_ghep_cap(diem, "D. rừng 200 cây, ngưỡng 0,5", "roc_auc")

    print("\n-- 7.5.1 đường cong học, F1 --")
    duong_cong_hoc(cach["A. logistic, ngưỡng 0,5"][0], X_tr, y_tr, skf,
                   "A. logistic, ngưỡng 0,5")
    duong_cong_hoc(cach["B. logistic + ngưỡng chọn bằng XTC"][0], X_tr, y_tr, skf,
                   "B. logistic + ngưỡng chọn bằng XTC")

    print("\n-- 7.5.2 chốt cấu hình B và đo một lần trên 603 dòng kiểm tra --")
    cuoi = chot_mo_hinh(X_tr, y_tr, X_te, y_te, tien, skf)
    joblib.dump(cuoi, THU_MUC_MO_HINH / "ch07-mo-hinh-chot.joblib")
    bang = pd.DataFrame([{"cau_hinh": ten,
                          "F1_xac_thuc_cheo": round(float(d["f1"].mean()), 4),
                          "sd": round(float(d["f1"].std(ddof=1)), 4),
                          "recall": round(float(d["recall"].mean()), 4),
                          "precision": round(float(d["precision"].mean()), 4),
                          "AUC": round(float(d["roc_auc"].mean()), 4)}
                         for ten, d in diem.items()])
    bang.to_csv(VAO / "ch07-so-sanh-cau-hinh.csv", index=False, encoding="utf-8")
    print("  đã lưu models/ch07-mo-hinh-chot.joblib"
          " và data/processed/ch07-so-sanh-cau-hinh.csv")


if __name__ == "__main__":
    main()
