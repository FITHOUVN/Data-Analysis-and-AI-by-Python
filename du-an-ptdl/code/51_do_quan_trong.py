"""Hai cách đo độ quan trọng của đặc trưng, và chúng khác nhau ở đâu (mục 6.3.3)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
KET_QUA = GOC_DU_AN / "output"
KET_QUA.mkdir(exist_ok=True)
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def do_quan_trong_gini(mo) -> pd.Series:
    """Độ quan trọng sẵn có của rừng, tính trên 30 cột SAU biến đổi."""
    ten = mo.named_steps["tien"].get_feature_names_out()
    return pd.Series(mo.named_steps["mo_hinh"].feature_importances_,
                     index=ten).sort_values(ascending=False)


def do_quan_trong_hoan_vi(mo, X_te, y_te) -> pd.Series:
    """Độ quan trọng hoán vị, tính trên 14 cột TRƯỚC biến đổi, theo F1."""
    kq = permutation_importance(mo, X_te, y_te, n_repeats=10,
                                random_state=42, scoring="f1")
    return pd.Series(kq.importances_mean,
                     index=X_te.columns).sort_values(ascending=False)


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    mo = Pipeline([("tien", clone(tien)),
                   ("mo_hinh", RandomForestClassifier(n_estimators=200,
                                                      random_state=42))])
    mo.fit(X_tr, y_tr)
    gini = do_quan_trong_gini(mo)
    print("-- Độ quan trọng sẵn có của rừng, tám dòng đầu trên 30 cột --")
    print(gini.head(8).round(4).to_string())
    print("Tổng 30 giá trị: %.4f | số cột nhận giá trị dương: %d/30"
          % (float(gini.sum()), int((gini > 0).sum())))
    print("Tổng của 22 cột một nóng: %.4f"
          % float(gini[[c for c in gini.index if c.startswith("pl__")]].sum()))
    hoan_vi = do_quan_trong_hoan_vi(mo, X_te, y_te)
    print("\n-- Độ quan trọng hoán vị theo F1, trên 14 cột gốc --")
    print(hoan_vi.round(4).to_string())
    print("Số cột có giá trị không dương: %d/14" % int((hoan_vi <= 0).sum()))
    hoan_vi.rename("do_quan_trong_hoan_vi").to_csv(
        KET_QUA / "ch06-do-quan-trong.csv", encoding="utf-8")
    print("\nĐã ghi bảng độ quan trọng vào output/ch06-do-quan-trong.csv")


if __name__ == "__main__":
    main()
