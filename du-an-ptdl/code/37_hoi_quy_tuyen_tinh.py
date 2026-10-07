"""Huấn luyện và đo mô hình hồi quy tuyến tính trên gói dữ liệu Chương 4 (mục 5.2.2)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

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


def do_theo_cap(mo_hinh, X_tr, X_te, y_tr, y_te) -> pd.DataFrame:
    """Đo cùng một mô hình trên cả hai tập: chênh lệch mới là dấu hiệu quá khớp."""
    dong = []
    for nhan, X, y in (("huấn luyện", X_tr, y_tr), ("kiểm tra", X_te, y_te)):
        d = mo_hinh.predict(X)
        dong.append({"tập": nhan,
                     "MAE": round(mean_absolute_error(y, d), 3),
                     "RMSE": round(root_mean_squared_error(y, d), 3),
                     "R2": round(r2_score(y, d), 4)})
    bang = pd.DataFrame(dong)
    return bang


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    mo = Pipeline([("tien", tien), ("mo_hinh", LinearRegression())])
    mo.fit(X_tr, y_tr)
    bang = do_theo_cap(mo, X_tr, X_te, y_tr, y_te)
    print(bang.to_string(index=False))
    print("Chênh lệch R2 giữa hai tập: %+.4f"
          % (bang.loc[0, "R2"] - bang.loc[1, "R2"]))
    print("Năm dự đoán đầu của tập kiểm tra:",
          [round(float(v), 2) for v in mo.predict(X_te)[:5]])
    print("Năm giá trị thực tương ứng    :",
          [round(float(v), 2) for v in y_te.iloc[:5]])
    joblib.dump(mo, THU_MUC_MO_HINH / "ch05-hoi-quy-tuyen-tinh.joblib")
    print("Đã lưu mô hình vào models/ch05-hoi-quy-tuyen-tinh.joblib")


if __name__ == "__main__":
    main()
