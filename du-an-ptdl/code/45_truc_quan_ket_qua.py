"""Ba biểu đồ đọc kết quả hồi quy: thực so với dự đoán, phần dư, và cột R² (mục 5.5)."""
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
KET_QUA = GOC_DU_AN / "output"
KET_QUA.mkdir(exist_ok=True)
THU_MUC_MO_HINH = GOC_DU_AN / "models"
# Hai cách chia nhóm. Điểm thực luôn nằm trong thang 0 tới 10, nên khoảng ngoài cùng đóng;
# dự đoán thì tràn ra ngoài thang, nên hai khoảng ngoài cùng phải để mở mới phủ hết 603 dòng.
KHOANG_THUC = [("0-2", 0, 2), ("2-4", 2, 4), ("4-6", 4, 6), ("6-8", 6, 8), ("8-10", 8, 10.01)]
KHOANG_DOAN = [("dưới 2", -99, 2), ("2-4", 2, 4), ("4-6", 4, 6), ("6-8", 6, 8),
               ("từ 8", 8, 99)]


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def ve_thuc_va_du_doan(y_te, du_goc) -> None:
    """Biểu đồ phân tán giá trị thực so với giá trị dự đoán, kèm đường chéo y = y_mũ."""
    fig, truc = plt.subplots(figsize=(6, 6))
    truc.scatter(du_goc, y_te, s=10, facecolors="none", edgecolors="black", linewidths=0.5)
    truc.plot([0, 10], [0, 10], color="black", linestyle="--", linewidth=1.2)
    truc.set_xlabel("Điểm cuối kỳ dự đoán")
    truc.set_ylabel("Điểm cuối kỳ thực tế")
    truc.set_title("Giá trị thực so với giá trị dự đoán, 603 dòng kiểm tra")
    truc.set_xlim(-2, 12)
    truc.set_ylim(-2, 12)
    truc.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch05-thuc-va-du-doan.png", dpi=150)
    plt.close(fig)


def ve_phan_du(y_te, du_goc) -> None:
    """Biểu đồ phần dư theo giá trị dự đoán, kèm đường ngang 0."""
    phan_du = y_te.to_numpy() - du_goc
    fig, truc = plt.subplots(figsize=(7, 4.5))
    truc.scatter(du_goc, phan_du, s=10, facecolors="none", edgecolors="black", linewidths=0.5)
    truc.axhline(0, color="black", linewidth=1.2)
    truc.set_xlabel("Điểm cuối kỳ dự đoán")
    truc.set_ylabel("Phần dư (thực trừ dự đoán)")
    truc.set_title("Phần dư theo giá trị dự đoán")
    truc.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch05-phan-du.png", dpi=150)
    plt.close(fig)


def phan_du_theo_nhom(y_te, du_goc, moc, khoang, ten_cot) -> pd.DataFrame:
    """Phần dư trung bình theo nhóm, chia theo đại lượng `moc` do người gọi chọn."""
    phan_du = y_te.to_numpy() - du_goc
    dong = []
    for nhan, lo, hi in khoang:
        chon = (moc >= lo) & (moc < hi)
        dong.append({ten_cot: nhan, "n": int(chon.sum()),
                     "phan_du_tb": round(float(phan_du[chon].mean()), 3),
                     "MAE": round(mean_absolute_error(y_te[chon], du_goc[chon]), 3)})
    return pd.DataFrame(dong)


def hai_truc_chan_doan(y_te, du_goc) -> None:
    """Chia phần dư theo hai trục rồi so: chỉ trục dự đoán chẩn đoán được dạng mô hình."""
    phan_du = y_te.to_numpy() - du_goc
    print(phan_du_theo_nhom(y_te, du_goc, y_te.to_numpy(), KHOANG_THUC,
                            "khoang_diem_thuc").to_string(index=False))
    print()
    print(phan_du_theo_nhom(y_te, du_goc, du_goc, KHOANG_DOAN,
                            "khoang_du_doan").to_string(index=False))
    print("Tương quan phần dư với dự đoán %+.4f | với điểm thực %+.4f | căn(1 - R2) %.4f"
          % (np.corrcoef(phan_du, du_goc)[0, 1],
             np.corrcoef(phan_du, y_te.to_numpy())[0, 1],
             np.sqrt(1 - r2_score(y_te, du_goc))))
    cao = (y_te == 10.0).to_numpy()
    print("Số dòng có điểm thực đúng 10,0: %d | mô hình đoán từ %.2f tới %.2f, trung bình %.2f"
          % (int(cao.sum()), du_goc[cao].min(), du_goc[cao].max(), du_goc[cao].mean()))


def ve_cot_r2(bang_ss: pd.DataFrame) -> None:
    """Biểu đồ cột R² sáu mô hình: trục bắt đầu từ 0 và ghi số lên đầu cột."""
    fig, truc = plt.subplots(figsize=(8, 4.5))
    cot = truc.bar(bang_ss["mo_hinh"], bang_ss["R2"].clip(lower=0), color="white",
                   edgecolor="black", linewidth=1.0, hatch="//")
    truc.bar_label(cot, labels=["%.3f" % v for v in bang_ss["R2"]], padding=2)
    truc.axhline(0, color="black", linewidth=1.0)
    truc.set_ylim(0, 1.0)
    truc.set_ylabel("$R^2$ trên tập kiểm tra")
    truc.set_title("Hệ số xác định của sáu mô hình, cùng 603 dòng kiểm tra")
    truc.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(KET_QUA / "ch05-cot-r2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    mo = Pipeline([("tien", clone(tien)), ("mo_hinh", LinearRegression())]).fit(X_tr, y_tr)
    du_goc = mo.predict(X_te)
    print("Mô hình tuyến tính: R2 %.4f | phần dư trung bình %+.4f"
          % (r2_score(y_te, du_goc), float((y_te.to_numpy() - du_goc).mean())))
    ve_thuc_va_du_doan(y_te, du_goc)
    ve_phan_du(y_te, du_goc)
    print("\n-- Phần dư trung bình theo hai cách chia nhóm --")
    hai_truc_chan_doan(y_te, du_goc)
    bang_ss = pd.read_csv(VAO / "ch05-so-sanh-mo-hinh.csv")
    ve_cot_r2(bang_ss)
    print("\nĐã ghi ba biểu đồ vào output/: ch05-thuc-va-du-doan.png, "
          "ch05-phan-du.png, ch05-cot-r2.png")


if __name__ == "__main__":
    main()
