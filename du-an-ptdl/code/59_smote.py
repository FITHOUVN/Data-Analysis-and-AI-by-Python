"""Lấy mẫu lại và SMOTE: đặt đúng chỗ, đặt sai chỗ, và bẫy cột một nóng (mục 7.4.3)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as OngLayMau
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def bon_do_do(ten, y_thuc, nh, ps):
    print("  %s" % ten)
    print("    rec %.4f | pre %.4f | F1 %.4f | AUC %.4f"
          % (recall_score(y_thuc, nh), precision_score(y_thuc, nh),
             f1_score(y_thuc, nh), roc_auc_score(y_thuc, ps)))


def cach_dung(tien):
    """SMOTE là bước GIỮA của một imblearn Pipeline, nên nó chỉ chạy
    khi fit."""
    return OngLayMau([
        ("tien", clone(tien)),
        ("lay_mau", SMOTE(random_state=42)),
        ("mo_hinh", LogisticRegression(max_iter=1000))])


def dat_dung_cho(X_tr, y_tr, X_te, y_te, tien, skf):
    ong = cach_dung(tien)
    d = cross_val_score(ong, X_tr, y_tr, cv=skf, scoring="f1")
    print("  F1 năm phần: %s"
          % " · ".join("%.4f" % v for v in d))
    print("    trung bình %.4f | sd %.4f" % (d.mean(), d.std(ddof=1)))
    ong.fit(X_tr, y_tr)
    bon_do_do("đặt đúng chỗ, đo trên tập kiểm tra", y_te, ong.predict(X_te),
              ong.predict_proba(X_te)[:, 1])
    return d.mean(), f1_score(y_te, ong.predict(X_te))


def dat_sai_cho_truoc_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, skf, f1_dung):
    """Bơm dữ liệu một lần cho cả tập huấn luyện rồi mới xác thực chéo: con số thành vô nghĩa."""
    bo = clone(tien).fit(X_tr)
    A2, a2 = SMOTE(random_state=42).fit_resample(bo.transform(X_tr), y_tr)
    print("  sau khi bơm: %d dòng, dương %d" % (len(a2), int(a2.sum())))
    d = cross_val_score(Pipeline([("mo_hinh", LogisticRegression(max_iter=1000))]),
                        A2, a2, cv=skf, scoring="f1")
    print("  F1 năm phần trên dữ liệu đã bơm:")
    print("    %s | trung bình %.4f"
          % (" · ".join("%.4f" % v for v in d), d.mean()))
    print("  LẠM PHÁT so với cách đặt đúng chỗ: %+.4f"
          % (d.mean() - f1_dung))
    mo = LogisticRegression(max_iter=1000).fit(A2, a2)
    bon_do_do("cùng mô hình đó, đo trên 603 dòng thật", y_te, mo.predict(bo.transform(X_te)),
              mo.predict_proba(bo.transform(X_te))[:, 1])
    return A2, bo


def dat_sai_cho_truoc_khi_chia(X, y, X_te, y_te, tien, f1_dung):
    """Bơm cả bảng trước khi chia tập: dòng sinh ra lọt vào tập kiểm tra."""
    bo = clone(tien).fit(X)
    Ab, yb = SMOTE(random_state=42).fit_resample(bo.transform(X), y)
    print("  cả bảng sau khi bơm: %d dòng, dương %d" % (len(yb), int(yb.sum())))
    chi_so = np.arange(len(yb))
    i_tr, i_te = train_test_split(chi_so, test_size=0.2, random_state=42, stratify=yb)
    mo = LogisticRegression(max_iter=1000).fit(np.asarray(Ab)[i_tr], yb[i_tr])
    nh = mo.predict(np.asarray(Ab)[i_te])
    ps = mo.predict_proba(np.asarray(Ab)[i_te])[:, 1]
    bon_do_do("đo trên 'tập kiểm tra' của dữ liệu đã bơm", yb[i_te], nh, ps)
    print("  trong đó %d/%d dòng là dòng SINH RA, không phải sinh viên thật"
          % (int((i_te >= len(y)).sum()), len(i_te)))
    print("  LẠM PHÁT của con số báo cáo: %+.4f" % (f1_score(yb[i_te], nh) - f1_dung))
    A_te = np.asarray(bo.transform(X_te))   # mô hình được khớp trên mảng numpy, bỏ tên cột
    bon_do_do("cùng mô hình đó, đo trên 603 dòng thật", y_te,
              mo.predict(A_te), mo.predict_proba(A_te)[:, 1])


def bay_mot_nong(A2, bo, so_dong_goc):
    """SMOTE nội suy giữa hai dòng, nên ô một nóng nhận giá trị không phải 0 hay 1."""
    ten = list(bo.get_feature_names_out())
    cot_pl = [i for i, t in enumerate(ten) if t.startswith("pl__")]
    moi = np.asarray(A2)[so_dong_goc:]
    g = moi[:, cot_pl]
    le = ((g > 0.001) & (g < 0.999)).any(axis=1)
    print("  %d dòng sinh thêm | %d dòng (%.1f%%) có ô một nóng không phải 0 hay 1"
          % (len(moi), int(le.sum()), 100 * le.mean()))
    vi_du = g[le][0]
    print("  ví dụ ba ô lạ của dòng sinh thêm đầu tiên:",
          [round(float(v), 4) for v in vi_du[(vi_du > 0.001) & (vi_du < 0.999)][:3]])
    tong = g.sum(axis=1)
    print("  tổng 22 ô một nóng của mỗi dòng sinh thêm: %.4f … %.4f (sáu biến phân loại)"
          % (tong.min(), tong.max()))


def bay_cach(X_tr, y_tr, tien, skf, cot_phan_loai):
    """Bảy cách xử lý mất cân bằng, cùng một phép xác thực chéo, cùng một độ đo."""
    cach = {
        "không làm gì": OngLayMau([("tien", clone(tien)),
                                   ("mo_hinh", LogisticRegression(max_iter=1000))]),
        "class_weight=balanced": OngLayMau([("tien", clone(tien)),
            ("mo_hinh", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "RandomOverSampler": OngLayMau([("tien", clone(tien)),
            ("lay_mau", RandomOverSampler(random_state=42)),
            ("mo_hinh", LogisticRegression(max_iter=1000))]),
        "RandomUnderSampler": OngLayMau([("tien", clone(tien)),
            ("lay_mau", RandomUnderSampler(random_state=42)),
            ("mo_hinh", LogisticRegression(max_iter=1000))]),
        "SMOTE": cach_dung(tien),
        "SMOTE sampling_strategy=0.5": OngLayMau([("tien", clone(tien)),
            ("lay_mau", SMOTE(random_state=42, sampling_strategy=0.5)),
            ("mo_hinh", LogisticRegression(max_iter=1000))]),
        "SMOTENC trước bộ tiền xử lý": OngLayMau([
            ("lay_mau", SMOTENC(categorical_features=cot_phan_loai, random_state=42)),
            ("tien", clone(tien)),
            ("mo_hinh", LogisticRegression(max_iter=1000))]),
    }
    diem = {}
    for ten, p in cach.items():
        r = cross_validate(p, X_tr, y_tr, cv=skf,
                           scoring=("f1", "recall", "precision", "roc_auc"))
        diem[ten] = r["test_f1"]
        print("  %-28s F1 %.4f (sd %.4f) | rec %.4f | pre %.4f | AUC %.4f"
              % (ten, r["test_f1"].mean(), r["test_f1"].std(ddof=1),
                 r["test_recall"].mean(), r["test_precision"].mean(),
                 r["test_roc_auc"].mean()))
    for a, b in (("SMOTE", "không làm gì"), ("class_weight=balanced", "không làm gì"),
                 ("SMOTE", "class_weight=balanced"),
                 ("SMOTE", "SMOTENC trước bộ tiền xử lý")):
        h = diem[a] - diem[b]
        print("  hiệu F1 từng phần, %s trừ %s: %s | dương %d/5"
              % (a, b, " · ".join("%+.4f" % v for v in h), int((h > 0).sum())))


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("-- (1) ĐÚNG: SMOTE là một bước của imblearn Pipeline --")
    f1_cv, f1_kt = dat_dung_cho(X_tr, y_tr, X_te, y_te, tien, skf)

    print("\n-- (2) SAI: bơm dữ liệu trước khi xác thực chéo --")
    A2, bo = dat_sai_cho_truoc_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, skf, f1_cv)

    print("\n-- (3) SAI NẶNG: bơm dữ liệu trước khi chia tập --")
    dat_sai_cho_truoc_khi_chia(X, y, X_te, y_te, tien, f1_kt)

    print("\n-- bẫy cột một nóng --")
    bay_mot_nong(A2, bo, len(y_tr))

    print("\n-- bảy cách xử lý mất cân bằng, StratifiedKFold(5, shuffle=True, random_state=42) --")
    cot_phan_loai = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    bay_cach(X_tr, y_tr, tien, skf, cot_phan_loai)


if __name__ == "__main__":
    main()
