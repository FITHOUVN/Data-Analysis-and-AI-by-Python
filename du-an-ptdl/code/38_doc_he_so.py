"""Đọc bảng hệ số của mô hình tuyến tính trên hai thang đo khác nhau (mục 5.2.3)."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]


def nap_goi():
    """Nạp bốn file tập đã chia và bộ tiền xử lý đã khớp của Chương 4."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").iloc[:, 0]
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").iloc[:, 0]
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def bang_he_so(bo_tien, X_tr, y_tr) -> pd.Series:
    """Khớp mô hình với một bộ tiền xử lý rồi trả hệ số kèm tên đặc trưng."""
    mo = Pipeline([("tien", clone(bo_tien)), ("mo_hinh", LinearRegression())])
    mo.fit(X_tr, y_tr)
    ten = mo.named_steps["tien"].get_feature_names_out()
    hs = pd.Series(mo.named_steps["mo_hinh"].coef_, index=ten)
    hs.attrs["chan"] = float(mo.named_steps["mo_hinh"].intercept_)
    hs.attrs["mo_hinh"] = mo
    return hs


def tien_xu_ly_tho() -> ColumnTransformer:
    """Bộ tiền xử lý giữ nguyên thang đo gốc của tám cột số, chỉ mã hóa cột phân loại."""
    bo = ColumnTransformer([
        ("so", "passthrough", COT_SO),
        ("pl", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DAC_TRUNG_PL)])
    bo.set_output(transform="pandas")
    return bo


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    hs = bang_he_so(tien, X_tr, y_tr)
    print("Hệ số chặn: %.4f | số hệ số: %d" % (hs.attrs["chan"], len(hs)))
    print("\n-- Mười hệ số lớn nhất theo giá trị tuyệt đối, thang đã chuẩn hóa --")
    print(hs.sort_values(key=abs, ascending=False).head(10).round(4).to_string())

    hs_tho = bang_he_so(tien_xu_ly_tho(), X_tr, y_tr)
    print("\n-- Tám cột số: hai thang đo cho hai bảng hệ số khác nhau --")
    bang = pd.DataFrame({"don_vi_goc": hs_tho.iloc[:8].to_numpy(),
                         "da_chuan_hoa": hs.iloc[:8].to_numpy(),
                         "do_lech_chuan": X_tr[COT_SO].std().to_numpy()}, index=COT_SO)
    bang["kiem_lai"] = bang["don_vi_goc"] * bang["do_lech_chuan"]
    print(bang.round({"don_vi_goc": 6, "da_chuan_hoa": 4,
                      "do_lech_chuan": 4, "kiem_lai": 4}).to_string())
    print("\nR2 trên tập kiểm tra: thang gốc %.4f | thang đã chuẩn hóa %.4f"
          % (r2_score(y_te, hs_tho.attrs["mo_hinh"].predict(X_te)),
             r2_score(y_te, hs.attrs["mo_hinh"].predict(X_te))))
    cap = [t for t in hs.index if "hinh_thuc_hoc" in t]
    print("Cặp hệ số one-hot của biến hai mức:",
          {t: round(float(hs[t]), 4) for t in cap},
          "| khoảng cách:", round(float(abs(hs[cap[0]] - hs[cap[1]])), 4))


if __name__ == "__main__":
    main()
