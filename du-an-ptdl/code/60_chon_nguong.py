"""Chọn ngưỡng quyết định: trên một tập kiểm định, và bằng TunedThresholdClassifierCV (mục 7.4.4)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, TunedThresholdClassifierCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
GIA_FP = 0.25      # giờ của cố vấn cho một ca báo động sai, giả thiết của Ví dụ 6.3
GIA_FN = 45.0      # giờ của sinh viên cho một ca bỏ sót


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def ong(tien):
    return Pipeline([("tien", clone(tien)),
                     ("mo_hinh", LogisticRegression(max_iter=1000))])


def mot_dong(ten, y_te, nh):
    tn, fp, fn, tp = confusion_matrix(y_te, nh).ravel()
    print("  %-30s rec %.4f | pre %.4f | F1 %.4f | TP %3d FP %3d FN %3d | tổn thất %.2f giờ"
          % (ten, recall_score(y_te, nh), precision_score(y_te, nh), f1_score(y_te, nh),
             tp, fp, fn, fp * GIA_FP + fn * GIA_FN))


def nguong_tren_tap_kiem_dinh(X_tr, y_tr, X_te, y_te, tien, hat):
    """Quy trình bốn bước: tách kiểm định, quét ngưỡng, khớp lại, đo một lần."""
    A, V, a, v = train_test_split(X_tr, y_tr, test_size=0.25, random_state=hat,
                                  stratify=y_tr)
    mo = ong(tien).fit(A, a)
    pv = mo.predict_proba(V)[:, 1]
    luoi = np.arange(0.01, 1.00, 0.01)
    diem = [f1_score(v, (pv >= t).astype(int)) for t in luoi]
    t_tot = float(luoi[int(np.argmax(diem))])
    cuoi = ong(tien).fit(X_tr, y_tr)
    nh = (cuoi.predict_proba(X_te)[:, 1] >= t_tot).astype(int)
    print("  giá trị khởi tạo %-3d | kiểm định %d dòng | ngưỡng %.2f | F1 kiểm định %.4f"
          " | F1 kiểm tra %.4f | rec %.4f"
          % (hat, len(v), t_tot, max(diem), f1_score(y_te, nh), recall_score(y_te, nh)))
    return t_tot


def nguong_bang_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, hat, diem_chon="f1",
                              ten_diem="f1"):
    """TunedThresholdClassifierCV: ngưỡng chọn bằng xác thực chéo, mô hình khớp lại."""
    cv = StratifiedKFold(5, shuffle=True, random_state=hat)
    do = TunedThresholdClassifierCV(ong(tien), scoring=diem_chon, cv=cv, refit=True,
                                    random_state=42).fit(X_tr, y_tr)
    nh = do.predict(X_te)
    print("  giá trị khởi tạo %-3d | độ đo %-10s | ngưỡng %.4f"
          % (hat, ten_diem, do.best_threshold_))
    print("    best_score_ %.4f | F1 kiểm tra %.4f"
          % (do.best_score_, f1_score(y_te, nh)))
    print("    rec %.4f | pre %.4f"
          % (recall_score(y_te, nh), precision_score(y_te, nh)))
    return do


def ton_that_gio(y_thuc, nhan):
    """Điểm càng cao càng tốt, nên tổn thất mang dấu âm."""
    tn, fp, fn, tp = confusion_matrix(y_thuc, nhan).ravel()
    return -(fp * GIA_FP + fn * GIA_FN)


def nguon_luc_huu_han(X_tr, y_tr, X_te, y_te, tien):
    """Khi mỗi đợt chỉ gọi được k sinh viên thì chính k quyết định ngưỡng."""
    mo = ong(tien).fit(X_tr, y_tr)
    ps = mo.predict_proba(X_te)[:, 1]
    thu_tu = np.argsort(-ps, kind="stable")
    thuc = y_te.to_numpy()
    for k in (40, 95, 150, 379):
        chon = thu_tu[:k]
        dung = int(thuc[chon].sum())
        print("  k=%-4d | ngưỡng tương ứng %.4f | bắt đúng %3d/%3d | recall %.4f"
              " | precision %.4f"
              % (k, ps[thu_tu[k - 1]], dung, k, dung / thuc.sum(), dung / k))


def hieu_qua_nhieu_lan_chia(X, y, tien, so_lan=10):
    """Hiệu ba độ đo, bản chọn ngưỡng bằng xác thực chéo trừ bản để ngưỡng 0,5."""
    gom = {k: [] for k in ("F1", "recall", "precision")}
    nguong = []
    for hat in range(so_lan):
        A, B, a, b = train_test_split(X, y, test_size=0.2, random_state=hat, stratify=y)
        goc = ong(tien).fit(A, a)
        do = TunedThresholdClassifierCV(ong(tien), scoring="f1",
                                        cv=StratifiedKFold(5, shuffle=True,
                                                           random_state=42),
                                        random_state=42).fit(A, a)
        nguong.append(do.best_threshold_)
        diem = {}
        for nhan, nh in (("goc", goc.predict(B)), ("tt", do.predict(B))):
            diem[nhan] = {"F1": f1_score(b, nh), "recall": recall_score(b, nh),
                          "precision": precision_score(b, nh)}
        for k in gom:
            gom[k].append(diem["tt"][k] - diem["goc"][k])
    print("  ngưỡng chọn được ở mười lần chia: %s" % " · ".join("%.4f" % t for t in nguong))
    for k, v in gom.items():
        v = np.array(v)
        print("  hiệu %-10s dương %2d/%d | %+.4f … %+.4f | trung bình %+.4f"
              % (k, int((v > 0).sum()), so_lan, v.min(), v.max(), v.mean()))


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    f2 = make_scorer(fbeta_score, beta=2)
    diem_ton_that = make_scorer(ton_that_gio, greater_is_better=True)

    print("-- (a) ngưỡng chọn trên MỘT tập kiểm định, năm hạt khác nhau --")
    for hat in (42, 0, 1, 2, 7):
        nguong_tren_tap_kiem_dinh(X_tr, y_tr, X_te, y_te, tien, hat)

    print("\n-- (b) ngưỡng chọn bằng xác thực chéo, năm hạt khác nhau --")
    for hat in (42, 0, 1, 2, 7):
        nguong_bang_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, hat)

    print("\n-- (c) đổi độ đo dùng để chọn ngưỡng --")
    do_f1 = nguong_bang_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, 42, "f1", "f1")
    do_f2 = nguong_bang_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, 42, f2, "F2")
    do_ton = nguong_bang_xac_thuc_cheo(X_tr, y_tr, X_te, y_te, tien, 42,
                                       diem_ton_that, "tổn thất")
    print("  tổn thất trung bình mỗi phần của bản chọn theo tổn thất: %.1f giờ"
          % (-do_ton.best_score_))
    goc = ong(tien).fit(X_tr, y_tr)
    mot_dong("ngưỡng 0,5 (Chương 6)", y_te, goc.predict(X_te))
    for ten, do in (("ngưỡng chọn theo F1", do_f1), ("ngưỡng chọn theo F2", do_f2),
                    ("ngưỡng chọn theo tổn thất", do_ton)):
        mot_dong("%s (%.4f)" % (ten, do.best_threshold_), y_te, do.predict(X_te))

    print("\n-- (d) khi nguồn lực hữu hạn: k cuộc gọi mỗi đợt --")
    nguon_luc_huu_han(X_tr, y_tr, X_te, y_te, tien)

    print("\n-- (e) hiệu qua mười lần chia: chọn ngưỡng bằng xác thực chéo trừ ngưỡng 0,5 --")
    hieu_qua_nhieu_lan_chia(X, y, tien)


if __name__ == "__main__":
    main()
