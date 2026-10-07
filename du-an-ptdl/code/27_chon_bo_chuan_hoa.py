"""Mô hình nào đổi kết quả khi chuẩn hóa, mô hình nào không (mục 4.1.3)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
MUC_TIEU = "diem_cuoi_ky"
MO_HINH = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=10)": Ridge(alpha=10.0),
    "KNeighborsRegressor(k=10)": KNeighborsRegressor(n_neighbors=10),
    "DecisionTreeRegressor(max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForestRegressor(100 cây)": RandomForestRegressor(n_estimators=100, random_state=42),
}


def r2_voi_bo(mo_hinh, bo, X_tr, X_te, y_tr, y_te) -> float:
    """Huấn luyện với một bộ chuẩn hóa rồi trả về R2 trên tập kiểm tra."""
    if bo is None:
        return r2_score(y_te, mo_hinh.fit(X_tr, y_tr).predict(X_te))
    bo = bo.fit(X_tr)                                   # chỉ khớp trên tập huấn luyện
    mo_hinh.fit(bo.transform(X_tr), y_tr)
    return r2_score(y_te, mo_hinh.predict(bo.transform(X_te)))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    X_tr, X_te, y_tr, y_te = train_test_split(d[COT_SO], d[MUC_TIEU],
                                              test_size=0.2, random_state=42)
    cac_bo = [("không chuẩn hóa", None), ("khoảng [0, 1]", MinMaxScaler()),
              ("phân phối chuẩn", StandardScaler()), ("bền với ngoại lai", RobustScaler())]
    dong = []
    for ten, mo in MO_HINH.items():
        ket = {"mo_hinh": ten}
        for nhan, bo in cac_bo:
            ket[nhan] = round(r2_voi_bo(clone(mo), clone(bo) if bo else None,
                                        X_tr, X_te, y_tr, y_te), 4)
        ket["lệch lớn nhất"] = round(max(ket[n] for n, _ in cac_bo)
                                     - min(ket[n] for n, _ in cac_bo), 4)
        dong.append(ket)
    print(pd.DataFrame(dong).to_string(index=False))

    # Cái giá của việc chuẩn hóa: thêm một đối tượng phải lưu và phải áp lại y nguyên
    bo = StandardScaler().fit(X_tr)
    quen = KNeighborsRegressor(n_neighbors=10).fit(bo.transform(X_tr), y_tr)
    print()
    print("R2 khi quên chuẩn hóa tập kiểm tra = %.4f" % r2_score(y_te, quen.predict(X_te)))


if __name__ == "__main__":
    main()
