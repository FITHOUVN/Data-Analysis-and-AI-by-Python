"""Bốn độ đo hồi quy: MAE, MSE, RMSE và R², kèm cách đọc từng con số (mục 5.4.1 và 5.4.2)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             root_mean_squared_error)
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


def bon_do_do(y_te, du_goc) -> None:
    """Bốn độ đo trên cùng một vectơ dự đoán, kèm phân bố sai số tuyệt đối."""
    print("MAE %.4f | MSE %.4f | RMSE %.4f | R2 %.4f"
          % (mean_absolute_error(y_te, du_goc), mean_squared_error(y_te, du_goc),
             root_mean_squared_error(y_te, du_goc), r2_score(y_te, du_goc)))
    sai = np.abs(y_te.to_numpy() - du_goc)
    print("Sai số tuyệt đối: trung vị %.3f | phân vị 90 %.3f | lớn nhất %.3f"
          % (np.median(sai), np.percentile(sai, 90), sai.max()))
    print("Số ca sai quá 2 điểm: %d | quá 3 điểm: %d | trên %d dòng"
          % (int((sai > 2).sum()), int((sai > 3).sum()), len(sai)))
    return sai


def do_nhay_voi_sai_so_lon(y_te, du_goc, sai) -> None:
    """Bỏ mười ca sai nặng nhất rồi đo lại: RMSE tụt nhiều hơn MAE."""
    te_nhat = np.argsort(-sai)[:10]
    giu = np.ones(len(sai), dtype=bool)
    giu[te_nhat] = False
    mae_truoc = mean_absolute_error(y_te, du_goc)
    rmse_truoc = root_mean_squared_error(y_te, du_goc)
    mae_sau = mean_absolute_error(y_te[giu], du_goc[giu])
    rmse_sau = root_mean_squared_error(y_te[giu], du_goc[giu])
    print("Tỉ số RMSE/MAE = %.3f" % (rmse_truoc / mae_truoc))
    print("Bỏ 10 ca tệ nhất: MAE %.4f -> %.4f (giảm %.1f%%) | RMSE %.4f -> %.4f (giảm %.1f%%)"
          % (mae_truoc, mae_sau, (mae_truoc - mae_sau) / mae_truoc * 100,
             rmse_truoc, rmse_sau, (rmse_truoc - rmse_sau) / rmse_truoc * 100))


def r2_bang_tay(y_te, du_goc) -> None:
    """Tính R² từ hai tổng bình phương rồi đối chiếu với r2_score."""
    sse = float(((y_te - du_goc) ** 2).sum())
    sst = float(((y_te - y_te.mean()) ** 2).sum())
    print("SSE %.2f | SST %.2f | 1 - SSE/SST = %.4f | r2_score = %.4f"
          % (sse, sst, 1 - sse / sst, r2_score(y_te, du_goc)))
    print("Phương sai mục tiêu trên tập kiểm tra: %.4f | RMSE %.4f"
          % (y_te.var(ddof=0), root_mean_squared_error(y_te, du_goc)))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    mo = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())])
    mo.fit(X_tr, y_tr)
    du_goc = mo.predict(X_te)
    print("-- Bốn độ đo của mô hình tuyến tính trên tập kiểm tra --")
    sai = bon_do_do(y_te, du_goc)
    print("\n-- Độ nhạy của hai độ đo với sai số lớn --")
    do_nhay_voi_sai_so_lon(y_te, du_goc, sai)
    print("\n-- R² tính bằng tay --")
    r2_bang_tay(y_te, du_goc)


if __name__ == "__main__":
    main()
