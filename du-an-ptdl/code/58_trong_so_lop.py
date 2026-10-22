"""Mất cân bằng gây hại ở đâu, và trọng số lớp so với đổi ngưỡng (mục 7.4.1 và 7.4.2)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]


def nap_goi():
    """Bảng đầy đủ, bộ tiền xử lý, và phép chia phân tầng của Chương 6."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def ong(tien, mo_hinh):
    return Pipeline([("tien", clone(tien)), ("mo_hinh", clone(mo_hinh))])


def lam_ngheo_lop_duong(X_tr, y_tr, X_te, y_te, tien):
    """Giữ nguyên tập kiểm tra, chỉ làm lớp dương của tập huấn luyện thưa dần."""
    rng = np.random.default_rng(42)
    duong = y_tr[y_tr == 1].index.to_numpy()
    am = y_tr[y_tr == 0].index.to_numpy()
    print("  tỉ lệ | n dương | recall | precision |   F1   |  AUC   | TP | FN"
          " | ngưỡng cho recall 0,60")
    for ti in (0.179, 0.10, 0.05, 0.02, 0.01):
        if ti >= 0.179:
            giu = y_tr.index.to_numpy()
        else:
            k = int(round(ti * len(am) / (1 - ti)))
            giu = np.concatenate([am, rng.choice(duong, size=k, replace=False)])
        A, a = X_tr.loc[giu], y_tr.loc[giu]
        mo = ong(tien, LogisticRegression(max_iter=1000)).fit(A, a)
        nh, ps = mo.predict(X_te), mo.predict_proba(X_te)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_te, nh).ravel()
        ng = next((t for t in np.arange(0.99, 0.0, -0.005)
                   if recall_score(y_te, (ps >= t).astype(int)) >= 0.60), float("nan"))
        print("  %.3f | %7d | %.4f | %9.4f | %.4f | %.4f | %2d | %2d | %.3f"
              % (a.mean(), int(a.sum()), recall_score(y_te, nh),
                 precision_score(y_te, nh, zero_division=0), f1_score(y_te, nh),
                 roc_auc_score(y_te, ps), tp, fn, ng))


def mot_dong(ten, y_te, nh, ps):
    print("  %-44s acc %.4f | pre %.4f | rec %.4f | F1 %.4f | AUC %.4f | ma %s"
          % (ten, accuracy_score(y_te, nh), precision_score(y_te, nh),
             recall_score(y_te, nh), f1_score(y_te, nh), roc_auc_score(y_te, ps),
             confusion_matrix(y_te, nh).tolist()))


def nguong_dat_muc_recall(ps, y_te, muc):
    """Ngưỡng cao nhất mà tại đó recall vẫn đạt mức đã cho."""
    return next(t for t in np.arange(0.999, 0.0, -0.001)
                if recall_score(y_te, (ps >= t).astype(int)) >= muc)


def trong_so_so_voi_nguong(ten_mo, tao, X_tr, y_tr, X_te, y_te, tien):
    """Cân bằng trọng số lớp có khác gì hạ ngưỡng về cùng mức recall hay không."""
    goc = ong(tien, tao(None)).fit(X_tr, y_tr)
    bal = ong(tien, tao("balanced")).fit(X_tr, y_tr)
    p0, pw = goc.predict_proba(X_te)[:, 1], bal.predict_proba(X_te)[:, 1]
    nh_w = bal.predict(X_te)
    mot_dong("%s, class_weight=balanced" % ten_mo, y_te, nh_w, pw)
    muc = recall_score(y_te, nh_w)
    ng = nguong_dat_muc_recall(p0, y_te, muc)
    nh_ng = (p0 >= ng).astype(int)
    mot_dong("%s, bản gốc hạ ngưỡng về %.3f" % (ten_mo, ng), y_te, nh_ng, p0)
    print("  ba con số (bản trọng số trừ bản hạ ngưỡng): đổi %d/%d nhãn"
          " | lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
          % (int((nh_w != nh_ng).sum()), len(y_te), float(np.abs(pw - p0).max()),
             f1_score(y_te, nh_w) - f1_score(y_te, nh_ng)))
    print("  tương quan thứ bậc giữa hai vectơ xác suất: %.6f"
          % pd.Series(p0).corr(pd.Series(pw), method="spearman"))
    return goc, bal


def hieu_qua_nhieu_lan_chia(X, y, tien, tao, ten_mo, so_lan=10):
    """Hiệu bốn độ đo, bản cân bằng trọng số trừ bản nguyên, qua nhiều lần chia."""
    gom = {k: [] for k in ("recall", "precision", "F1", "AUC")}
    for hat in range(so_lan):
        A, B, a, b = train_test_split(X, y, test_size=0.2, random_state=hat, stratify=y)
        diem = {}
        for nhan, w in (("goc", None), ("bal", "balanced")):
            mo = ong(tien, tao(w)).fit(A, a)
            nh, ps = mo.predict(B), mo.predict_proba(B)[:, 1]
            diem[nhan] = {"recall": recall_score(b, nh),
                          "precision": precision_score(b, nh),
                          "F1": f1_score(b, nh), "AUC": roc_auc_score(b, ps)}
        for k in gom:
            gom[k].append(diem["bal"][k] - diem["goc"][k])
    print("  %s" % ten_mo)
    for k, v in gom.items():
        v = np.array(v)
        print("    hiệu %-10s dương %2d/%d | %+.4f … %+.4f | trung bình %+.4f"
              % (k, int((v > 0).sum()), so_lan, v.min(), v.max(), v.mean()))


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("huấn luyện %d dòng (dương %d, %.2f%%) | kiểm tra %d dòng (dương %d)\n"
          % (len(y_tr), int(y_tr.sum()), 100 * y_tr.mean(), len(y_te), int(y_te.sum())))

    print("-- 7.4.1 làm lớp dương của tập huấn luyện thưa dần --")
    lam_ngheo_lop_duong(X_tr, y_tr, X_te, y_te, tien)

    print("\n-- 7.4.2 trọng số lớp so với đổi ngưỡng --")
    tao_lg = lambda w: LogisticRegression(max_iter=1000, class_weight=w)
    goc, bal = trong_so_so_voi_nguong("hồi quy logistic", tao_lg,
                                      X_tr, y_tr, X_te, y_te, tien)
    print("  hệ số chặn: bản gốc %+.4f | bản cân bằng %+.4f | tương quan 30 hệ số %.6f"
          % (goc.named_steps["mo_hinh"].intercept_[0],
             bal.named_steps["mo_hinh"].intercept_[0],
             float(np.corrcoef(goc.named_steps["mo_hinh"].coef_[0],
                               bal.named_steps["mo_hinh"].coef_[0])[0, 1])))
    print()
    tao_rf = lambda w: RandomForestClassifier(n_estimators=200, random_state=42,
                                              class_weight=w)
    trong_so_so_voi_nguong("rừng 200 cây", tao_rf, X_tr, y_tr, X_te, y_te, tien)

    print("\n-- hiệu qua mười lần chia phân tầng --")
    hieu_qua_nhieu_lan_chia(X, y, tien, tao_lg, "hồi quy logistic")
    hieu_qua_nhieu_lan_chia(X, y, tien, tao_rf, "rừng 200 cây")


if __name__ == "__main__":
    main()
