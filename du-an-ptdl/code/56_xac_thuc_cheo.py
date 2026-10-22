"""Xác thực chéo k phần: dao động của một lần chia, cách chia, và cách đọc kết quả (mục 7.3)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
MO_HINH = {"hồi quy tuyến tính": LinearRegression(),
           "rừng 200 cây": RandomForestRegressor(n_estimators=200, random_state=42)}


def nap_goi():
    """Gộp lại toàn bộ 3.012 dòng: xác thực chéo tự chia tập, không cần chia trước."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").squeeze("columns")
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").squeeze("columns")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return pd.concat([X_tr, X_te]), pd.concat([y_tr, y_te]), tien


def ong(tien, mo_hinh):
    """Bộ tiền xử lý luôn được khớp LẠI bên trong từng phần, nên không rò rỉ."""
    return Pipeline([("tien", clone(tien)), ("mo_hinh", clone(mo_hinh))])


def dao_dong_mot_lan_chia(X, y, tien, so_lan=10):
    """R² của từng mô hình qua mười lần chia tập khác nhau: biên độ của một MỨC."""
    print("-- mười lần chia tập, mỗi lần một con số R² --")
    for ten, m in MO_HINH.items():
        d = []
        for hat in range(so_lan):
            A, B, a, b = train_test_split(X, y, test_size=0.2,
                                          random_state=hat)
            d.append(r2_score(b, ong(tien, m).fit(A, a).predict(B)))
        d = np.array(d)
        print("  %s" % ten)
        for i in range(0, len(d), 5):
            print("    " + " ".join("%.4f" % x for x in d[i:i + 5]))
        print("    %.4f … %.4f | rộng %.4f"
              % (d.min(), d.max(), d.max() - d.min()))


def xac_thuc_cheo(X, y, tien, cv, nhan):
    """Năm con số R², trung bình và độ lệch chuẩn mẫu của chúng."""
    diem = {}
    for ten, m in MO_HINH.items():
        d = cross_val_score(ong(tien, m), X, y, cv=cv, scoring="r2")
        diem[ten] = d
        print("  %s" % ten)
        print("    " + " · ".join("%.4f" % v for v in d))
        print("    TB %.4f | sd %.4f" % (d.mean(), d.std(ddof=1)))
    h = diem["rừng 200 cây"] - diem["hồi quy tuyến tính"]
    # Các tập huấn luyện của CV chồng lấn; p dưới đây chỉ minh họa cách tính,
    # không phải kiểm định xác nhận trên các quan sát độc lập.
    t, p = stats.ttest_rel(diem["rừng 200 cây"], diem["hồi quy tuyến tính"])
    print("  hiệu ghép cặp theo từng phần (%s):" % nhan)
    print("    " + " · ".join("%+.4f" % v for v in h))
    print("  dương %d/%d phần | trung bình %+.4f | t %.3f | p %.4f"
          % (int((h > 0).sum()), len(h), h.mean(), t, p))
    return diem


def khoang_mot_do_lech(diem):
    """Hai khoảng trung bình ± một độ lệch chuẩn có chồng nhau hay không."""
    for ten, d in diem.items():
        print("  %-18s trung bình ± sd = [%.4f ; %.4f]"
              % (ten, d.mean() - d.std(ddof=1), d.mean() + d.std(ddof=1)))
    a = diem["hồi quy tuyến tính"]
    b = diem["rừng 200 cây"]
    lo = max(a.mean() - a.std(ddof=1), b.mean() - b.std(ddof=1))
    hi = min(a.mean() + a.std(ddof=1), b.mean() + b.std(ddof=1))
    print("  phần chồng nhau: [%.4f ; %.4f], rộng %.4f" % (lo, hi, hi - lo))


def ti_le_lop_tung_phan(X, y, cv, nhan):
    """Tỉ lệ lớp dương của từng phần: đây là lý do phải phân tầng."""
    ti_le = [y.iloc[te].mean() for _, te in cv.split(X, y)]
    print("  %-34s %s | dải %.4f"
          % (nhan, " · ".join("%.4f" % v for v in ti_le), max(ti_le) - min(ti_le)))


def main() -> None:
    X, y, tien = nap_goi()
    print("gộp lại %d dòng, %d cột đặc trưng\n" % (X.shape[0], X.shape[1]))
    dao_dong_mot_lan_chia(X, y, tien)

    print("\n-- xác thực chéo 5 phần, KFold(n_splits=5, shuffle=True, random_state=42) --")
    diem = xac_thuc_cheo(X, y, tien, KFold(n_splits=5, shuffle=True, random_state=42),
                         "rừng trừ tuyến tính")
    print("\n-- hai khoảng một độ lệch chuẩn --")
    khoang_mot_do_lech(diem)

    print("\n-- cùng dữ liệu, 10 phần --")
    xac_thuc_cheo(X, y, tien, KFold(n_splits=10, shuffle=True, random_state=42),
                  "10 phần")

    print("\n-- cùng dữ liệu, 5 phần KHÔNG xáo trộn --")
    xac_thuc_cheo(X, y, tien, KFold(n_splits=5), "5 phần không xáo trộn")

    print("\n-- tỉ lệ lớp dương của từng phần, bài toán phân loại --")
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    Xc = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    yc = (bang["ket_qua"] == "Không đạt").astype(int)
    print("  tỉ lệ trên cả bảng: %.4f" % yc.mean())
    ti_le_lop_tung_phan(Xc, yc, KFold(5, shuffle=True, random_state=42),
                        "KFold(5, shuffle=True)")
    ti_le_lop_tung_phan(Xc, yc, KFold(5), "KFold(5) không xáo trộn")
    ti_le_lop_tung_phan(Xc, yc, StratifiedKFold(5, shuffle=True, random_state=42),
                        "StratifiedKFold(5, shuffle=True)")


if __name__ == "__main__":
    main()
