"""Cây hồi quy ba tầng: huấn luyện, in cây, và kiểm cây có cần chuẩn hóa hay không (mục 5.3)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def tien_xu_ly_tho() -> ColumnTransformer:
    """Giữ nguyên thang đo gốc của tám cột số: ngưỡng in trong cây mới đọc được."""
    bo = ColumnTransformer([
        ("so", "passthrough", COT_SO),
        ("pl", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DAC_TRUNG_PL)])
    bo.set_output(transform="pandas")
    return bo


def cay_ba_tang(X_tr, X_te, y_tr, y_te):
    """Huấn luyện cây sâu ba tầng trên đặc trưng thô rồi in toàn bộ cây."""
    bo = tien_xu_ly_tho()
    mo = Pipeline([("tien", bo), ("cay", DecisionTreeRegressor(max_depth=3, random_state=42))])
    mo.fit(X_tr, y_tr)
    cay = mo.named_steps["cay"]
    d = mo.predict(X_te)
    print("Độ sâu %d | số lá %d | MAE %.3f | RMSE %.3f | R2 %.4f"
          % (cay.get_depth(), cay.get_n_leaves(), mean_absolute_error(y_te, d),
             root_mean_squared_error(y_te, d), r2_score(y_te, d)))
    print(export_text(cay, feature_names=list(bo.get_feature_names_out()), decimals=2))
    return mo


def co_can_chuan_hoa(X_tr, X_te, y_tr, y_te, tien) -> None:
    """Cây chia theo thứ tự giá trị, nên đổi thang đo không đổi cách chia."""
    dong = []
    for ten_bo, bo in (("thang gốc", tien_xu_ly_tho()), ("đã chuẩn hóa", clone(tien))):
        for sau in (3, None):
            mo = Pipeline([("tien", clone(bo)),
                           ("cay", DecisionTreeRegressor(max_depth=sau, random_state=42))])
            mo.fit(X_tr, y_tr)
            dong.append({"bo_tien_xu_ly": ten_bo,
                         "max_depth": "không giới hạn" if sau is None else sau,
                         "so_la": mo.named_steps["cay"].get_n_leaves(),
                         "R2": round(r2_score(y_te, mo.predict(X_te)), 4)})
    print(pd.DataFrame(dong).to_string(index=False))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Cây hồi quy ba tầng trên đặc trưng thang gốc --")
    cay_ba_tang(X_tr, X_te, y_tr, y_te)
    print("-- Cây có cần chuẩn hóa thang đo hay không --")
    co_can_chuan_hoa(X_tr, X_te, y_tr, y_te, tien)


if __name__ == "__main__":
    main()
