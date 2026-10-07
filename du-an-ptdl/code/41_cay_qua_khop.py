"""Chứng minh quá khớp bằng số: cây không giới hạn độ sâu so với hồi quy tuyến tính (mục 5.3.3)."""
import collections
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def do_hai_tap(ten: str, mo, X_tr, X_te, y_tr, y_te) -> list:
    """Đo một mô hình trên cả hai tập và tính khoảng cách tổng quát hóa."""
    dong = []
    for nhan, X, y in (("huấn luyện", X_tr, y_tr), ("kiểm tra", X_te, y_te)):
        d = mo.predict(X)
        dong.append({"mo_hinh": ten, "tap": nhan,
                     "MAE": round(mean_absolute_error(y, d), 3),
                     "RMSE": round(root_mean_squared_error(y, d), 3),
                     "R2": round(r2_score(y, d), 4)})
    return dong


def dem_la_mot_mau(cay_day, X_tr) -> None:
    """Đếm số lá chỉ chứa đúng một dòng huấn luyện: dấu hiệu học thuộc."""
    cay = cay_day.named_steps["cay"]
    la = cay.apply(cay_day.named_steps["tien"].transform(X_tr))
    dem = collections.Counter(la)
    mot = sum(1 for v in dem.values() if v == 1)
    print("Độ sâu %d | số lá %d | số lá chỉ chứa một dòng huấn luyện %d (%.1f%%)"
          % (cay.get_depth(), cay.get_n_leaves(), mot, mot / cay.get_n_leaves() * 100))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    cay_day = Pipeline([("tien", clone(tien)), ("cay", DecisionTreeRegressor(random_state=42))])
    cay_day.fit(X_tr, y_tr)
    tuyen_tinh = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())])
    tuyen_tinh.fit(X_tr, y_tr)

    dem_la_mot_mau(cay_day, X_tr)
    bang = pd.DataFrame(do_hai_tap("cây không giới hạn", cay_day, X_tr, X_te, y_tr, y_te)
                        + do_hai_tap("tuyến tính", tuyen_tinh, X_tr, X_te, y_tr, y_te))
    print(bang.to_string(index=False))
    for ten in bang["mo_hinh"].unique():
        p = bang[bang["mo_hinh"] == ten].set_index("tap")["R2"]
        print("%-20s khoảng cách tổng quát hóa theo R2: %+.4f"
              % (ten, p["huấn luyện"] - p["kiểm tra"]))
    r_cay = bang[(bang["mo_hinh"] == "cây không giới hạn") & (bang["tap"] == "kiểm tra")]
    r_tt = bang[(bang["mo_hinh"] == "tuyến tính") & (bang["tap"] == "kiểm tra")]
    print("Cây kém mô hình tuyến tính %.4f điểm R2 trên tập kiểm tra"
          % (float(r_tt["R2"].iloc[0]) - float(r_cay["R2"].iloc[0])))


if __name__ == "__main__":
    main()
