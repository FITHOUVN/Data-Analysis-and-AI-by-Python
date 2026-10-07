"""Nạp gói dữ liệu Chương 4 bàn giao và dựng mô hình mốc (mục 5.1.2 và 5.1.3)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

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


def ba_do_do(ten: str, y_thuc, du_doan) -> dict:
    """Ba độ đo dùng chung cho mọi mô hình của chương."""
    return {"mo_hinh": ten,
            "MAE": round(mean_absolute_error(y_thuc, du_doan), 3),
            "RMSE": round(root_mean_squared_error(y_thuc, du_doan), 3),
            "R2": round(r2_score(y_thuc, du_doan), 4)}


def nghiem_thu(X_tr, X_te, y_tr, y_te, tien) -> None:
    """Kiểm gói bàn giao trước khi huấn luyện bất kỳ mô hình nào."""
    print("Tập huấn luyện", X_tr.shape, "| tập kiểm tra", X_te.shape)
    print("Mục tiêu: trung bình tập huấn luyện %.4f | tập kiểm tra %.4f"
          % (y_tr.mean(), y_te.mean()))
    print("Số đặc trưng sau biến đổi:", tien.transform(X_te).shape[1])
    print("Số ô thiếu trong đặc trưng:", int(X_tr.isna().sum().sum()),
          "| số dòng hai tập cộng lại:", len(X_tr) + len(X_te))


def dung_moc(X_tr, X_te, y_tr, y_te) -> None:
    """Hai mô hình mốc: luôn đoán trung bình, và luôn đoán trung vị."""
    dong = []
    for cach, nhan in (("mean", "mốc trung bình"), ("median", "mốc trung vị")):
        moc = DummyRegressor(strategy=cach).fit(X_tr, y_tr)
        d = moc.predict(X_te)
        ket = ba_do_do(nhan, y_te, d)
        ket["gia_tri_doan"] = round(float(d[0]), 4)
        dong.append(ket)
    print(pd.DataFrame(dong).to_string(index=False))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Nghiệm thu gói dữ liệu Chương 4 bàn giao --")
    nghiem_thu(X_tr, X_te, y_tr, y_te, tien)
    print("\n-- Hai mô hình mốc --")
    dung_moc(X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    main()
