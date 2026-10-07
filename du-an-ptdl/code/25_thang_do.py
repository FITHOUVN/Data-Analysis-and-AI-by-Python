"""Đo tác hại của việc tám cột số không cùng thang đo (mục 4.1.1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
MUC_TIEU = "diem_cuoi_ky"


def bang_thang_do(d: pd.DataFrame) -> None:
    """In miền giá trị và độ rộng của tám cột số, sắp theo độ rộng."""
    bang = pd.DataFrame({"nho_nhat": d[COT_SO].min(), "lon_nhat": d[COT_SO].max()})
    bang["do_rong"] = bang["lon_nhat"] - bang["nho_nhat"]
    bang["do_lech_chuan"] = d[COT_SO].std()
    bang = bang.sort_values("do_rong", ascending=False)
    bang["lan_so_cot_hep_nhat"] = (bang["do_rong"] / bang["do_rong"].min()).round(1)
    print(bang.round(3).to_string())


def gop_vao_khoang_cach(d: pd.DataFrame) -> None:
    """Mỗi cột góp bao nhiêu phần trăm vào khoảng cách Euclid giữa hai sinh viên."""
    a, b = d.loc[0, COT_SO].astype("float64"), d.loc[1, COT_SO].astype("float64")
    hieu_goc = (a - b) ** 2
    z = pd.DataFrame(StandardScaler().fit_transform(d[COT_SO]), columns=COT_SO)
    hieu_z = (z.loc[0] - z.loc[1]) ** 2
    bang = pd.DataFrame({"gop_goc_%": hieu_goc / hieu_goc.sum() * 100,
                         "gop_sau_chuan_hoa_%": hieu_z / hieu_z.sum() * 100})
    print(bang.sort_values("gop_goc_%", ascending=False).round(2).to_string())
    print("Khoảng cách Euclid: bản gốc %.2f, sau chuẩn hóa %.2f"
          % (np.sqrt(hieu_goc.sum()), np.sqrt(hieu_z.sum())))


def knn_truoc_va_sau(d: pd.DataFrame) -> None:
    """So R2 của k lân cận gần nhất khi không chuẩn hóa và khi chuẩn hóa."""
    X_tr, X_te, y_tr, y_te = train_test_split(d[COT_SO], d[MUC_TIEU],
                                              test_size=0.2, random_state=42)
    tho = KNeighborsRegressor(n_neighbors=10).fit(X_tr, y_tr)
    bo = StandardScaler().fit(X_tr)                    # chỉ khớp trên tập huấn luyện
    chuan = KNeighborsRegressor(n_neighbors=10).fit(bo.transform(X_tr), y_tr)
    print("không chuẩn hóa: R2 = %.4f" % r2_score(y_te, tho.predict(X_te)))
    print("đã chuẩn hóa   : R2 = %.4f" % r2_score(y_te, chuan.predict(bo.transform(X_te))))


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("Dữ liệu nhận từ Chương 3:", d.shape, "| số ô thiếu:", int(d.isna().sum().sum()))
    print("\n-- Miền giá trị tám cột số --")
    bang_thang_do(d)
    print("\n-- Hai sinh viên đầu bảng: cột nào quyết định khoảng cách --")
    gop_vao_khoang_cach(d)
    print("\n-- Hệ quả trên một mô hình dựa trên khoảng cách --")
    knn_truoc_va_sau(d)


if __name__ == "__main__":
    main()
