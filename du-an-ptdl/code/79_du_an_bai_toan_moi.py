"""Bài toán của dự án: cảnh báo trước khi có điểm giữa kỳ (mục 11.2.4).

Chạy từ gốc dự án:
    python code/79_du_an_bai_toan_moi.py

Chương trình làm hai việc, theo đúng thứ tự ấy:
  1. HIỆU CHUẨN: khớp lại cấu hình đã chốt ở mục 7.5.2 trên đủ 14 cột và kiểm rằng nó
     tái lập đúng bộ số đã in trong sách. Nếu phép `assert` này trượt thì mọi con số của
     bài toán mới cũng không đáng tin, nên nó đứng trước.
  2. Khớp cùng cấu hình ấy trên 13 cột, tức bỏ `diem_giua_ky` vì cột này chưa tồn tại tại
     thời điểm ra quyết định của bài toán mới.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, TunedThresholdClassifierCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
CHUA_CO_O_TUAN_8 = ["diem_giua_ky"]
GIA_FP, GIA_FN = 0.25, 45.0       # đúng mức đã chốt ở mục 6.4.4
SO_LAN_CHIA = 10


def bo_tien_theo_cot(bang_goc, cot):
    """Đúng bộ tiền xử lý của mục 4.4.1, nhưng dựng lại theo danh sách cột được truyền vào."""
    so = [c for c in cot if pd.api.types.is_numeric_dtype(bang_goc[c])]
    pl = [c for c in cot if c not in so]
    return ColumnTransformer([("so", StandardScaler(), so),
                              ("pl", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), pl)])


def khop_va_do(bang, X, y, cot, skf, hat=42):
    """Khớp hồi quy logistic kèm ngưỡng chọn bằng xác thực chéo, rồi đo một lần."""
    X_tr, X_te, y_tr, y_te = train_test_split(X[cot], y, test_size=0.2,
                                              random_state=hat, stratify=y)
    ong = Pipeline([("tien", bo_tien_theo_cot(bang, cot)),
                    ("mo_hinh", LogisticRegression(max_iter=1000))])
    mo_hinh = TunedThresholdClassifierCV(ong, scoring="f1", cv=skf, refit=True,
                                         random_state=42).fit(X_tr, y_tr)
    nhan = mo_hinh.predict(X_te)
    xac_suat = mo_hinh.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, nhan).ravel()
    return {"so_cot": len(cot), "nguong": mo_hinh.best_threshold_,
            "acc": accuracy_score(y_te, nhan), "pre": precision_score(y_te, nhan),
            "rec": recall_score(y_te, nhan), "f1": f1_score(y_te, nhan),
            "auc": roc_auc_score(y_te, xac_suat), "bo_sot": int(fn),
            "bao_dong_sai": int(fp), "ma": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "ton_that": fn * GIA_FN + fp * GIA_FP}


def in_dong(ten, kq):
    print("  %-21s %2d cột | ngưỡng %.4f"
          % (ten, kq["so_cot"], kq["nguong"]))
    print("       F1 %.4f | rec %.4f | AUC %.4f | bỏ sót %d"
          % (kq["f1"], kq["rec"], kq["auc"], kq["bo_sot"]))
    print("       pre %.4f | báo động sai %d"
          % (kq["pre"], kq["bao_dong_sai"]))


def main():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    cot_day_du = list(X.columns)
    cot_du_an = [c for c in cot_day_du if c not in CHUA_CO_O_TUAN_8]

    print("-- 11.2.4 hiệu chuẩn: cấu hình đã chốt ở mục 7.5.2, khớp lại ở đây")
    cu = khop_va_do(bang, X, y, cot_day_du, skf)
    in_dong("mô hình của Chương 7", cu)
    # Khóa GIÁ TRỊ, không khóa kết luận: bốn số dưới đây là bốn số đã in ở mục 7.5.2.
    assert round(cu["nguong"], 4) == 0.2727, cu["nguong"]
    assert round(cu["f1"], 4) == 0.7266, cu["f1"]
    assert round(cu["rec"], 4) == 0.8611, cu["rec"]
    assert cu["ma"] == [[440, 55], [15, 93]], cu["ma"]
    print("     hiệu chuẩn ĐẠT: bốn giá trị trùng bộ số của mục 7.5.2")

    print("-- 11.2.4 bài toán của dự án: bỏ %s" % ", ".join(CHUA_CO_O_TUAN_8))
    moi = khop_va_do(bang, X, y, cot_du_an, skf)
    in_dong("mô hình của dự án", moi)
    print("  ba con số của phép so: F1 %+.4f | recall %+.4f | AUC %+.4f"
          % (moi["f1"] - cu["f1"], moi["rec"] - cu["rec"],
             moi["auc"] - cu["auc"]))
    print("  theo đơn vị nghiệp vụ: bỏ sót %d thay vì %d sinh viên trên 603 dòng kiểm tra,"
          " tổn thất %.2f giờ thay vì %.2f giờ"
          % (moi["bo_sot"], cu["bo_sot"], moi["ton_that"], cu["ton_that"]))

    # Bước 6 của Quy trình 11.1 không cho kết luận từ MỘT lần chia, nên đo hiệu qua
    # mười lần chia và xem hiệu có đổi dấu hay không, theo từng độ đo.
    print("-- 11.2.4 phép thử dấu: hiệu (dự án trừ Chương 7) qua %d lần chia" % SO_LAN_CHIA)
    hieu = {k: [] for k in ("f1", "rec", "auc", "bo_sot")}
    for hat in range(SO_LAN_CHIA):
        a = khop_va_do(bang, X, y, cot_day_du, skf, hat)
        b = khop_va_do(bang, X, y, cot_du_an, skf, hat)
        for k in hieu:
            hieu[k].append(b[k] - a[k])
    for k, dau in (("f1", "âm"), ("rec", "âm"), ("auc", "âm"), ("bo_sot", "dương")):
        d = np.array(hieu[k], dtype=float)
        cung_dau = int((d < 0).sum()) if dau == "âm" else int((d > 0).sum())
        print("  hiệu %-7s %s %2d/%d | trung bình %+.4f | từ %+.4f tới %+.4f"
              % (k, dau, cung_dau, SO_LAN_CHIA, d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
