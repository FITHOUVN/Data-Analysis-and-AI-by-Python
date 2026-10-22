"""Ghép tinh chỉnh siêu tham số với xác thực chéo: GridSearchCV và RandomizedSearchCV (mục 7.3.4)."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
SAU = [4, 5, 6, 7, 8, 10, 12]
LA = [1, 5, 10, 20, 30, 50]
LUOI = {"cay__max_depth": SAU, "cay__min_samples_leaf": LA}


def nap_goi():
    """Nạp gói dữ liệu hồi quy của Chương 4 và bộ tiền xử lý đã khớp."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").squeeze("columns")
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").squeeze("columns")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def ong_cay(tien, **tham_so):
    return Pipeline([("tien", clone(tien)),
                     ("cay", DecisionTreeRegressor(random_state=42, **tham_so))])


def do_bang_luoi(tien, X_tr, y_tr, X_te, y_te):
    """GridSearchCV: chọn bằng xác thực chéo, khớp lại, rồi đo một lần."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.perf_counter()
    do = GridSearchCV(ong_cay(tien), LUOI, cv=kf, scoring="r2", refit=True).fit(X_tr, y_tr)
    giay = time.perf_counter() - t0
    print("  best_params_ %s" % do.best_params_)
    print("  best_score_ %.4f | %d ô x 5 phần + 1 lần khớp lại | %.1f giây"
          % (do.best_score_, len(SAU) * len(LA), giay))
    bang = pd.DataFrame(do.cv_results_).sort_values("rank_test_score")
    print("  năm cấu hình đầu theo xác thực chéo:")
    for _, r in bang.head(5).iterrows():
        print("    d%-3s l%-3s | trung bình %.4f | sd %.4f"
              % (r["param_cay__max_depth"], r["param_cay__min_samples_leaf"],
                 r["mean_test_score"], r["std_test_score"]))
    du = do.predict(X_te)
    print("  đo một lần trên tập kiểm tra: R2 %.4f | MAE %.3f"
          % (r2_score(y_te, du), mean_absolute_error(y_te, du)))
    return do, du


def doi_hat_xac_thuc_cheo(tien, X_tr, y_tr, X_te, y_te):
    """Đổi hạt của phép chia năm phần: cấu hình chọn được ổn định tới mức nào."""
    print("-- đổi hạt của KFold --")
    for hat in (0, 1, 7, 42, 2026):
        do = GridSearchCV(ong_cay(tien), LUOI, cv=KFold(5, shuffle=True, random_state=hat),
                          scoring="r2").fit(X_tr, y_tr)
        print("  hạt %-5d chọn d%s-l%-3s | best_score_ %.4f | R2 kiểm tra %.4f"
              % (hat, do.best_params_["cay__max_depth"],
                 do.best_params_["cay__min_samples_leaf"], do.best_score_,
                 r2_score(y_te, do.predict(X_te))))


def do_ngau_nhien_co_xac_thuc_cheo(tien, X_tr, y_tr, X_te, y_te):
    """RandomizedSearchCV trên không gian ba chiều lớn, cùng phép chia năm phần."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kg = {"cay__max_depth": randint(2, 21), "cay__min_samples_leaf": randint(1, 51),
          "cay__min_samples_split": randint(2, 41)}
    print("-- RandomizedSearchCV trên không gian 20 x 50 x 39 cấu hình --")
    for n in (20, 60):
        t0 = time.perf_counter()
        do = RandomizedSearchCV(ong_cay(tien), kg, n_iter=n, cv=kf, scoring="r2",
                                random_state=42).fit(X_tr, y_tr)
        print("  n_iter %-3d | best_score_ %.4f | %s | R2 kiểm tra %.4f | %.1f giây"
              % (n, do.best_score_,
                 {k.split("__")[1]: v for k, v in do.best_params_.items()},
                 r2_score(y_te, do.predict(X_te)), time.perf_counter() - t0))


def ba_con_so_hai_cach(y_te, du_a, du_b, ten_a, ten_b):
    """Ba con số cho phép so hai cấu hình trên cùng tập kiểm tra."""
    print("  %s so với %s:" % (ten_b, ten_a))
    print("    đổi %d/%d dự đoán | lệch lớn nhất %.4f điểm | lệch R2 %+.4f"
          " | đổi phía ngưỡng 4,0: %d sinh viên"
          % (int((du_a != du_b).sum()), len(y_te), float(np.abs(du_a - du_b).max()),
             r2_score(y_te, du_b) - r2_score(y_te, du_a),
             int(((du_a >= 4) != (du_b >= 4)).sum())))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- GridSearchCV trên lưới 42 cấu hình, KFold(5, shuffle=True, random_state=42) --")
    do, du_cv = do_bang_luoi(tien, X_tr, y_tr, X_te, y_te)

    print("\n-- ba cách chọn cấu hình, cùng đo trên 603 dòng kiểm tra --")
    du = {}
    for nhan, tham_so in (("d6-l5 (một tập kiểm định)", {"max_depth": 6, "min_samples_leaf": 5}),
                          ("d8-l20 (chốt ở Chương 5)", {"max_depth": 8, "min_samples_leaf": 20})):
        du[nhan] = ong_cay(tien, **tham_so).fit(X_tr, y_tr).predict(X_te)
        print("  %-28s R2 %.4f" % (nhan, r2_score(y_te, du[nhan])))
    print("  %-28s R2 %.4f" % ("GridSearchCV chọn", r2_score(y_te, du_cv)))
    cao_nhat = max(((s, l) for s in SAU for l in LA),
                   key=lambda k: r2_score(y_te, ong_cay(tien, max_depth=k[0],
                                                        min_samples_leaf=k[1])
                                          .fit(X_tr, y_tr).predict(X_te)))
    du_gian = ong_cay(tien, max_depth=cao_nhat[0],
                      min_samples_leaf=cao_nhat[1]).fit(X_tr, y_tr).predict(X_te)
    print("  %-28s R2 %.4f (cấu hình d%d-l%d)"
          % ("nhìn thẳng tập kiểm tra", r2_score(y_te, du_gian), cao_nhat[0], cao_nhat[1]))
    ba_con_so_hai_cach(y_te, du["d6-l5 (một tập kiểm định)"], du_cv,
                       "cấu hình do một tập kiểm định chọn", "cấu hình do xác thực chéo chọn")
    ba_con_so_hai_cach(y_te, du["d6-l5 (một tập kiểm định)"], du_gian,
                       "cấu hình do một tập kiểm định chọn", "cấu hình nhìn thẳng tập kiểm tra")

    print()
    doi_hat_xac_thuc_cheo(tien, X_tr, y_tr, X_te, y_te)
    print()
    do_ngau_nhien_co_xac_thuc_cheo(tien, X_tr, y_tr, X_te, y_te)
    joblib.dump(do.best_estimator_, THU_MUC_MO_HINH / "ch07-cay-xac-thuc-cheo.joblib")
    print("\nĐã lưu models/ch07-cay-xac-thuc-cheo.joblib")


if __name__ == "__main__":
    main()
