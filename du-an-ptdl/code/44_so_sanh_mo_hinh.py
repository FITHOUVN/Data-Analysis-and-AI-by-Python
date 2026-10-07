"""So sáu mô hình trên cùng tập kiểm tra, kèm ba con số của mọi so sánh (mục 5.4.3)."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
MO_HINH = {
    "mốc trung bình": DummyRegressor(strategy="mean"),
    "tuyến tính": LinearRegression(),
    "cây không giới hạn": DecisionTreeRegressor(random_state=42),
    "cây sâu 5": DecisionTreeRegressor(max_depth=5, random_state=42),
    "cây chặn quá khớp": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                               random_state=42),
    "rừng ngẫu nhiên": RandomForestRegressor(n_estimators=200, random_state=42),
}


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def khop_tat_ca(tien, X_tr, X_te, y_tr, y_te):
    """Khớp từng mô hình trên cùng tập huấn luyện, đo trên cùng tập kiểm tra."""
    dong, du_doan = [], {}
    for ten, goc in MO_HINH.items():
        t0 = time.perf_counter()
        mo = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(goc))]).fit(X_tr, y_tr)
        giay = time.perf_counter() - t0
        d = mo.predict(X_te)
        du_doan[ten] = d
        dong.append({"mo_hinh": ten,
                     "MAE": round(mean_absolute_error(y_te, d), 3),
                     "RMSE": round(root_mean_squared_error(y_te, d), 3),
                     "R2": round(r2_score(y_te, d), 4),
                     "giay_khop": round(giay, 3)})
    return pd.DataFrame(dong), du_doan


def bang_ba_con_so(du_doan, y_te, chuan="tuyến tính") -> pd.DataFrame:
    """Ba con số bắt buộc khi so một mô hình với mô hình chuẩn."""
    goc = du_doan[chuan]
    dong = []
    for ten, d in du_doan.items():
        if ten == chuan:
            continue
        dong.append({"so_voi_" + chuan: ten,
                     "so_du_doan_doi": int((np.round(goc, 4) != np.round(d, 4)).sum()),
                     "tren_tong": len(goc),
                     "lech_lon_nhat": round(float(np.abs(goc - d).max()), 4),
                     "lech_R2": round(r2_score(y_te, d) - r2_score(y_te, goc), 4)})
    return pd.DataFrame(dong)


def hieu_qua_nhieu_lan_chia(X_tr, X_te, y_tr, y_te, tien, so_lan=10) -> None:
    """Đo HIỆU hai mô hình trên từng lần chia: hiệu có đổi dấu hay không mới là câu trả lời."""
    X_all = pd.concat([X_tr, X_te], ignore_index=True)
    y_all = pd.concat([y_tr, y_te], ignore_index=True)
    hieu = []
    for hat in range(so_lan):
        A, B, a, b = train_test_split(X_all, y_all, test_size=0.2, random_state=hat)
        diem = {}
        for ten in ("tuyến tính", "rừng ngẫu nhiên"):
            mo = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(MO_HINH[ten]))]).fit(A, a)
            diem[ten] = r2_score(b, mo.predict(B))
        hieu.append(diem["rừng ngẫu nhiên"] - diem["tuyến tính"])
        print("  hạt %d: tuyến tính %.4f | rừng %.4f | hiệu %+.4f"
              % (hat, diem["tuyến tính"], diem["rừng ngẫu nhiên"], hieu[-1]))
    h = np.array(hieu)
    print("Hiệu dương %d/%d lần | nhỏ nhất %+.4f | lớn nhất %+.4f | trung bình %+.4f"
          % (int((h > 0).sum()), so_lan, h.min(), h.max(), h.mean()))


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    bang, du_doan = khop_tat_ca(tien, X_tr, X_te, y_tr, y_te)
    print(bang.to_string(index=False))
    print("\n-- Ba con số của mọi so sánh 'có đổi hay không' --")
    print(bang_ba_con_so(du_doan, y_te).to_string(index=False))
    print("\n-- Hiệu rừng trừ tuyến tính trên mười lần chia tập độc lập --")
    hieu_qua_nhieu_lan_chia(X_tr, X_te, y_tr, y_te, tien)
    bang.to_csv(VAO / "ch05-so-sanh-mo-hinh.csv", index=False, encoding="utf-8")
    print("\nĐã ghi bảng so sánh vào data/processed/ch05-so-sanh-mo-hinh.csv")


if __name__ == "__main__":
    main()
