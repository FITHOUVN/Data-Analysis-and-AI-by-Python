"""Ba biểu đồ đọc kết quả phân loại: ma trận nhầm lẫn, đường ROC, độ quan trọng (mục 6.5)."""
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
KET_QUA = GOC_DU_AN / "output"
KET_QUA.mkdir(exist_ok=True)
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
TEN_LOP = ["Đạt", "Không đạt"]


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.1."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, tien


def ve_ma_tran(hai_mo, X_te, y_te) -> None:
    """Hai ma trận nhầm lẫn cạnh nhau, thang xám, có cả số đếm."""
    fig, truc = plt.subplots(1, 2, figsize=(9, 4))
    for o, (ten, mo) in zip(truc, hai_mo.items()):
        ConfusionMatrixDisplay.from_estimator(
            mo, X_te, y_te, display_labels=TEN_LOP, cmap="Greys",
            colorbar=False, ax=o, values_format="d")
        o.set_title(ten)
        o.set_xlabel("Nhãn mô hình đoán")
        o.set_ylabel("Nhãn thực tế")
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch06-ma-tran-nham-lan.png", dpi=150)
    plt.close(fig)


def ve_roc(hai_mo, X_te, y_te) -> None:
    """Hai đường ROC trên cùng một hệ trục, phân biệt bằng nét chứ không bằng màu."""
    fig, o = plt.subplots(figsize=(6, 6))
    for thu_tu, ((ten, mo), kieu) in enumerate(zip(hai_mo.items(), ("-", "--"))):
        # scikit-learn 1.9 nhận kiểu vẽ qua `curve_kwargs`; truyền thẳng `color=` như
        # mã của tài liệu cũ sẽ dừng với TypeError. Đường chéo chỉ vẽ một lần, nếu
        # không thì chú giải có hai dòng "Chance level" giống hệt nhau.
        RocCurveDisplay.from_estimator(
            mo, X_te, y_te, name=ten, ax=o,
            curve_kwargs={"color": "black", "linestyle": kieu, "linewidth": 1.4},
            plot_chance_level=(thu_tu == len(hai_mo) - 1),
            chance_level_kw={"color": "black", "linestyle": ":", "linewidth": 1.0,
                             "label": "Đoán ngẫu nhiên (AUC = 0,5)"})
    o.set_xlabel("Tỉ lệ dương giả (FPR)")
    o.set_ylabel("Tỉ lệ dương thật (TPR, bằng recall)")
    o.set_title("Đường ROC trên 603 dòng kiểm tra")
    o.legend(loc="lower right")
    o.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch06-roc.png", dpi=150)
    plt.close(fig)


def ve_do_quan_trong(mo, X_te, y_te) -> None:
    """Biểu đồ cột ngang độ quan trọng hoán vị, 14 cột gốc, kèm thanh độ lệch chuẩn."""
    kq = permutation_importance(mo, X_te, y_te, n_repeats=10,
                                random_state=42, scoring="f1")
    bang = pd.DataFrame({"tb": kq.importances_mean, "sd": kq.importances_std},
                        index=X_te.columns).sort_values("tb")
    fig, o = plt.subplots(figsize=(7, 5))
    o.barh(bang.index, bang["tb"], xerr=bang["sd"], color="white",
           edgecolor="black", linewidth=1.0, hatch="//",
           error_kw={"ecolor": "black", "capsize": 2, "linewidth": 0.8})
    o.axvline(0, color="black", linewidth=1.0)
    o.set_xlabel("Mức F1 mất đi khi xáo trộn cột đó")
    o.set_title("Độ quan trọng hoán vị của rừng ngẫu nhiên")
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch06-do-quan-trong.png", dpi=150)
    plt.close(fig)


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    hai_mo = {}
    for ten, goc in (("Hồi quy logistic", LogisticRegression(max_iter=1000)),
                     ("Rừng ngẫu nhiên", RandomForestClassifier(n_estimators=200,
                                                                random_state=42))):
        hai_mo[ten] = Pipeline([("tien", clone(tien)),
                                ("mo_hinh", goc)]).fit(X_tr, y_tr)
    ve_ma_tran(hai_mo, X_te, y_te)
    ve_roc(hai_mo, X_te, y_te)
    ve_do_quan_trong(hai_mo["Rừng ngẫu nhiên"], X_te, y_te)
    print("Đã ghi ba biểu đồ vào output/: ch06-ma-tran-nham-lan.png, "
          "ch06-roc.png, ch06-do-quan-trong.png")


if __name__ == "__main__":
    main()
