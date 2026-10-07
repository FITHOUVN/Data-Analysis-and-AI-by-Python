"""Hai tham số chặn quá khớp của cây hồi quy: max_depth và min_samples_leaf (mục 5.3.4)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
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


def mot_luot(tien, X_tr, X_te, y_tr, y_te, **tham_so) -> dict:
    """Khớp một cây với một bộ siêu tham số rồi trả về một dòng kết quả."""
    mo = Pipeline([("tien", clone(tien)),
                   ("cay", DecisionTreeRegressor(random_state=42, **tham_so))])
    mo.fit(X_tr, y_tr)
    r_tr = r2_score(y_tr, mo.predict(X_tr))
    d = mo.predict(X_te)
    return {"so_la": mo.named_steps["cay"].get_n_leaves(),
            "R2_huan_luyen": round(r_tr, 4),
            "R2_kiem_tra": round(r2_score(y_te, d), 4),
            "khoang_cach": round(r_tr - r2_score(y_te, d), 4),
            "MAE": round(mean_absolute_error(y_te, d), 3),
            "RMSE": round(root_mean_squared_error(y_te, d), 3)}


def quet_do_sau(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Quét max_depth từ nông tới không giới hạn."""
    dong = []
    for sau in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None):
        ket = {"max_depth": "không giới hạn" if sau is None else sau}
        ket.update(mot_luot(tien, X_tr, X_te, y_tr, y_te, max_depth=sau))
        dong.append(ket)
    print(pd.DataFrame(dong).to_string(index=False))


def quet_so_mau_la(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Quét min_samples_leaf, giữ max_depth ở giá trị tốt nhất của lần quét trước."""
    dong = []
    for it_nhat in (1, 5, 10, 20, 30, 50):
        ket = {"min_samples_leaf": it_nhat}
        ket.update(mot_luot(tien, X_tr, X_te, y_tr, y_te,
                            max_depth=8, min_samples_leaf=it_nhat))
        dong.append(ket)
    print(pd.DataFrame(dong).to_string(index=False))


def chon_bang_tap_kiem_dinh(tien, X_tr, X_te, y_tr, y_te) -> None:
    """Chọn max_depth trên tập kiểm định tách từ tập huấn luyện, rồi đo MỘT lần trên kiểm tra."""
    X_con, X_kd, y_con, y_kd = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
    print("Còn lại %d dòng | kiểm định %d dòng" % (len(X_con), len(X_kd)))
    dong = []
    for sau in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None):
        mo = Pipeline([("tien", clone(tien)),
                       ("cay", DecisionTreeRegressor(max_depth=sau, random_state=42))])
        mo.fit(X_con, y_con)
        dong.append({"max_depth": "không giới hạn" if sau is None else sau,
                     "so_la": mo.named_steps["cay"].get_n_leaves(),
                     "R2_kiem_dinh": round(r2_score(y_kd, mo.predict(X_kd)), 4)})
    bang = pd.DataFrame(dong)
    print(bang.to_string(index=False))
    tot = bang.loc[bang["R2_kiem_dinh"].idxmax(), "max_depth"]
    cuoi = Pipeline([("tien", clone(tien)),
                     ("cay", DecisionTreeRegressor(max_depth=tot, random_state=42))])
    cuoi.fit(X_tr, y_tr)
    print("Chọn max_depth=%s trên kiểm định | khớp lại trên cả %d dòng | R2 kiểm tra %.4f"
          % (tot, len(X_tr), r2_score(y_te, cuoi.predict(X_te))))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    print("-- Quét max_depth --")
    quet_do_sau(tien, X_tr, X_te, y_tr, y_te)
    print("\n-- Quét min_samples_leaf với max_depth = 8 --")
    quet_so_mau_la(tien, X_tr, X_te, y_tr, y_te)
    print("\n-- Cấu hình chốt của chương --")
    chot = mot_luot(tien, X_tr, X_te, y_tr, y_te, max_depth=8, min_samples_leaf=20)
    print(pd.DataFrame([chot]).to_string(index=False))
    print("1%% số dòng huấn luyện = %.0f dòng" % (len(X_tr) * 0.01))
    print("\n-- Chọn siêu tham số bằng tập kiểm định, không nhìn tập kiểm tra --")
    chon_bang_tap_kiem_dinh(tien, X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    main()
