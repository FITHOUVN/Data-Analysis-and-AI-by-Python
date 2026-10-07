"""fit, transform, fit_transform: bộ chuẩn hóa chỉ được học từ tập huấn luyện (mục 4.1.4)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]


def lech_trung_binh(d: pd.DataFrame, X_tr: pd.DataFrame) -> None:
    """So trung bình của bộ chuẩn hóa khi khớp trên toàn bộ dữ liệu và khi khớp trên tập huấn luyện."""
    toan_bo = StandardScaler().fit(d[COT_SO])
    chi_train = StandardScaler().fit(X_tr)
    bang = pd.DataFrame({"mean_ toàn bộ": toan_bo.mean_, "mean_ chỉ train": chi_train.mean_},
                        index=COT_SO)
    bang["lệch mean_"] = (bang["mean_ toàn bộ"] - bang["mean_ chỉ train"]).abs()
    bang["lệch mean_ %"] = bang["lệch mean_"] / bang["mean_ chỉ train"].abs() * 100
    print(bang.round(4).to_string())


def lech_gia_tri_bien(d: pd.DataFrame, X_tr: pd.DataFrame, X_te: pd.DataFrame) -> None:
    """Tham số do một dòng duy nhất quyết định thì lệch mạnh hơn hẳn."""
    mm_toan_bo, mm_train = MinMaxScaler().fit(d[COT_SO]), MinMaxScaler().fit(X_tr)
    bang2 = pd.DataFrame({"max_ toàn bộ": mm_toan_bo.data_max_,
                          "max_ chỉ train": mm_train.data_max_}, index=COT_SO)
    bang2["lệch"] = bang2["max_ toàn bộ"] - bang2["max_ chỉ train"]
    print(bang2[bang2["lệch"] != 0].round(3).to_string())

    bo = MinMaxScaler().set_output(transform="pandas").fit(X_tr)
    z_te = bo.transform(X_te)
    ngoai = ((z_te < 0) | (z_te > 1)).sum()
    print("Ô của tập kiểm tra nằm ngoài [0, 1]:", {c: int(v) for c, v in ngoai.items() if v})
    print("Miền thực tế sau biến đổi: từ %.4f tới %.4f" % (z_te.min().min(), z_te.max().max()))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    X_tr, X_te = train_test_split(d[COT_SO], test_size=0.2, random_state=42)
    print("Tập huấn luyện", X_tr.shape, "| tập kiểm tra", X_te.shape)
    print("\n-- Trung bình: khớp sai chỗ thì lệch bao nhiêu --")
    lech_trung_binh(d, X_tr)
    print("\n-- Giá trị biên: tham số do một dòng quyết định --")
    lech_gia_tri_bien(d, X_tr, X_te)


if __name__ == "__main__":
    main()
