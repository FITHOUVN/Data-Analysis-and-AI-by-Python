"""Chia tập: ba tập, vai trò của random_state và chia phân tầng (mục 4.3.1 và 4.3.2)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

MUC_TIEU = "diem_cuoi_ky"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def tien_xu_ly() -> ColumnTransformer:
    """Bộ tiền xử lý dùng chung cho mọi lần chia."""
    return ColumnTransformer([("so", StandardScaler(), COT_SO),
                              ("pl", OneHotEncoder(handle_unknown="ignore"), DAC_TRUNG_PL)])


def chia_ba_tap(X: pd.DataFrame, y: pd.Series) -> None:
    """Chia 60/20/20 bằng hai lần gọi train_test_split."""
    X_con, X_te, y_con, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_kd, y_tr, y_kd = train_test_split(X_con, y_con, test_size=0.25, random_state=42)
    for ten, phan in (("huấn luyện", X_tr), ("kiểm định", X_kd), ("kiểm tra", X_te)):
        print("%-12s %5d dòng = %5.1f%%" % (ten, len(phan), len(phan) / len(X) * 100))
    print("Tổng:", len(X_tr) + len(X_kd) + len(X_te),
          "| tập còn lại sau lần chia một:", len(X_con))


def doi_hat_ngau_nhien(X: pd.DataFrame, y: pd.Series) -> None:
    """R2 của cùng một mô hình trên mười cách chia khác nhau."""
    ket = []
    for hat in range(10):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=hat)
        mo = Pipeline([("tien", tien_xu_ly()), ("hoi_quy", LinearRegression())]).fit(X_tr, y_tr)
        ket.append(round(r2_score(y_te, mo.predict(X_te)), 4))
    s = pd.Series(ket)
    print("R2 theo random_state 0..9:", ket)
    print("nhỏ nhất %.4f | lớn nhất %.4f | khoảng rộng %.4f | độ lệch chuẩn %.4f"
          % (s.min(), s.max(), s.max() - s.min(), s.std()))


def phan_tang(d: pd.DataFrame) -> None:
    """Tỉ lệ lớp 'Không đạt' trong tập kiểm tra, chia ngẫu nhiên và chia phân tầng."""
    y_nhan = d["ket_qua"]
    print("Tỉ lệ 'Không đạt' trên toàn bộ 3.012 dòng: %.2f%%"
          % (float((y_nhan == "Không đạt").mean()) * 100))
    for ten, tang in (("ngẫu nhiên", None), ("phân tầng", y_nhan)):
        ty_le = []
        for hat in range(10):
            _, y_te_nhan = train_test_split(y_nhan, test_size=0.2, random_state=hat,
                                            stratify=tang)
            ty_le.append(round(float((y_te_nhan == "Không đạt").mean()) * 100, 2))
        print("%-12s %s | rộng %.2f điểm phần trăm"
              % (ten, ty_le, max(ty_le) - min(ty_le)))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    X, y = d[COT_SO + DAC_TRUNG_PL], d[MUC_TIEU]
    print("-- Chia ba tập 60/20/20 --")
    chia_ba_tap(X, y)
    print("\n-- Cùng mô hình, mười cách chia --")
    doi_hat_ngau_nhien(X, y)
    print("\n-- Chia phân tầng theo nhãn ket_qua --")
    phan_tang(d)


if __name__ == "__main__":
    main()
