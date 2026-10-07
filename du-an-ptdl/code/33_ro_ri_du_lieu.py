"""Rò rỉ dữ liệu: thứ tự giữa chia tập và chuẩn hóa phải đúng (mục 4.3.3)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

MUC_TIEU = "diem_cuoi_ky"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
MO_HINH = {"LinearRegression": LinearRegression(),
           "KNeighborsRegressor(k=10)": KNeighborsRegressor(n_neighbors=10)}


def sai_thu_tu(d: pd.DataFrame, bo_mau) -> pd.DataFrame:
    """So thứ tự chuẩn hóa trước khi chia với thứ tự chia trước khi chuẩn hóa."""
    X, y = d[COT_SO], d[MUC_TIEU]
    dong = []
    for ten, mo in MO_HINH.items():
        X_tat_ca = clone(bo_mau).fit_transform(X)       # SAI: khớp trên toàn bộ dữ liệu
        a_tr, a_te, y_tr, y_te = train_test_split(X_tat_ca, y, test_size=0.2, random_state=42)
        sai = r2_score(y_te, clone(mo).fit(a_tr, y_tr).predict(a_te))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        bo = clone(bo_mau).fit(X_tr)                    # ĐÚNG: khớp sau khi đã chia
        dung = r2_score(y_te, clone(mo).fit(bo.transform(X_tr), y_tr).predict(bo.transform(X_te)))
        dong.append({"mo_hinh": ten, "thu_tu_sai": round(sai, 4),
                     "thu_tu_dung": round(dung, 4), "lech": round(sai - dung, 4)})
    return pd.DataFrame(dong)


def tham_so_bi_nhiem(d: pd.DataFrame) -> None:
    """Tham số nào của bộ chuẩn hóa bị dữ liệu kiểm tra làm nhiễu, và nhiễu bao nhiêu."""
    X_tr, X_te = train_test_split(d[COT_SO], test_size=0.2, random_state=42)
    mm_all, mm_tr = MinMaxScaler().fit(d[COT_SO]), MinMaxScaler().fit(X_tr)
    st_all, st_tr = StandardScaler().fit(d[COT_SO]), StandardScaler().fit(X_tr)
    bang = pd.DataFrame({
        "lech_mean_%": (st_all.mean_ - st_tr.mean_) / st_tr.mean_ * 100,
        "lech_max_": mm_all.data_max_ - mm_tr.data_max_,
        "lech_max_%": (mm_all.data_max_ - mm_tr.data_max_) / mm_tr.data_max_ * 100,
    }, index=COT_SO).abs()
    print("Lệch lớn nhất: mean_ %.2f%% | data_max_ %.2f%%"
          % (bang["lech_mean_%"].max(), bang["lech_max_%"].max()))


def dong_trung_vuot_bien(d: pd.DataFrame) -> None:
    """Mười hai mã sinh viên còn trùng sau Chương 3: cặp nào bị chia sang hai tập."""
    tr, te = train_test_split(d, test_size=0.2, random_state=42)
    trung = d.loc[d.duplicated(subset=["ma_sv"], keep=False), "ma_sv"].unique()
    bi_tach = [m for m in trung if (tr["ma_sv"] == m).any() and (te["ma_sv"] == m).any()]
    print("Số mã còn trùng:", len(trung), "| số mã bị chia sang hai tập:", len(bi_tach))
    print("Số dòng tập kiểm tra có bản sinh đôi trong tập huấn luyện:",
          int(te["ma_sv"].isin(bi_tach).sum()), "trên", len(te))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("-- Chuẩn hóa về phân phối chuẩn, hai thứ tự chạy --")
    print(sai_thu_tu(d, StandardScaler()).to_string(index=False))
    print("\n-- Chuẩn hóa về khoảng [0, 1], hai thứ tự chạy --")
    print(sai_thu_tu(d, MinMaxScaler()).to_string(index=False))
    print("\n-- Tham số bị dữ liệu kiểm tra làm nhiễu --")
    tham_so_bi_nhiem(d)
    print("\n-- Một đường rò rỉ khác: dòng trùng vượt biên hai tập --")
    dong_trung_vuot_bien(d)


if __name__ == "__main__":
    main()
