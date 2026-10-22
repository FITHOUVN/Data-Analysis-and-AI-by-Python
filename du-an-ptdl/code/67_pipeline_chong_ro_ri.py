"""Pipeline chống rò rỉ: lạm phát đo được, và lỗi khớp lại trên lô mới (mục 8.3.4)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan", "nop_tre_tb_gio",
          "diem_giua_ky"]
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc", "hinh_thuc_hoc",
          "thiet_bi_chinh"]


def bang_co_o_thieu():
    """Khoét lại đúng các ô thiếu của bản thô, để bước điền có việc thật phải làm."""
    sach = pd.read_csv(VAO / "du-lieu-sach.csv")
    tho = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv")
    X = sach[COT_SO + COT_PL].copy()
    for cot in ("ty_le_hoan_thanh_video", "diem_giua_ky"):
        ma = set(tho.loc[tho[cot].isna(), "ma_sv"])
        X.loc[sach["ma_sv"].isin(ma), cot] = np.nan
    y = (sach["ket_qua"] == "Không đạt").astype(int)
    print("  ô thiếu đã khoét lại: %s" % X.isna().sum()[lambda s: s > 0].to_dict())
    return X, y


def bo_tien():
    return ColumnTransformer([
        ("so", Pipeline([("dien", SimpleImputer(strategy="median")),
                         ("thang", StandardScaler())]), COT_SO),
        ("pl", Pipeline([("dien", SimpleImputer(strategy="constant",
                                                fill_value="Không rõ")),
                         ("ma", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))]), COT_PL)])


def lam_phat(X, y, skf):
    """Hai cách chạy cùng một phép xác thực chéo: bước tiền xử lý trong và ngoài."""
    ong = Pipeline([("tien", bo_tien()),
                    ("mo_hinh", LogisticRegression(max_iter=1000))])
    trong = cross_val_score(ong, X, y, cv=skf, scoring="f1")
    A = bo_tien().fit_transform(X)
    ngoai = cross_val_score(LogisticRegression(max_iter=1000), A, y, cv=skf, scoring="f1")
    in_mot_dong("điền và chuẩn hóa", trong, ngoai)

    ong_chon = Pipeline([("tien", bo_tien()), ("chon", SelectKBest(f_classif, k=5)),
                         ("mo_hinh", LogisticRegression(max_iter=1000))])
    trong2 = cross_val_score(ong_chon, X, y, cv=skf, scoring="f1")
    B = SelectKBest(f_classif, k=5).fit_transform(A, y)
    ngoai2 = cross_val_score(LogisticRegression(max_iter=1000), B, y, cv=skf, scoring="f1")
    in_mot_dong("thêm chọn 5 đặc trưng", trong2, ngoai2)


def in_mot_dong(ten, trong, ngoai):
    print("  %-24s trong pipeline %.4f ± %.4f"
          % (ten, trong.mean(), trong.std(ddof=1)))
    print("  %-24s ngoài pipeline %.4f ± %.4f"
          % ("", ngoai.mean(), ngoai.std(ddof=1)))
    print("    lạm phát %+.4f | dương %d/5 phần"
          % (ngoai.mean() - trong.mean(), int((ngoai > trong).sum())))
    print("    lệch lớn nhất một phần %+.4f"
          % (ngoai - trong)[np.argmax(np.abs(ngoai - trong))])


def khop_lai_tren_lo_moi():
    """Lỗi triển khai hay gặp nhất: gọi fit thay vì transform trên lô dữ liệu mới."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ong = joblib.load(THU_MUC_MO_HINH / "ch08-pipeline-khong-nguong.joblib")
    tien, lin = ong.named_steps["tien"], ong.named_steps["mo_hinh"]
    nguong = 0.2727104755205902
    for n in (40, 150, 300, 603):
        lo, y_lo = X_te.iloc[:n], y_te.iloc[:n]
        A = tien.transform(lo)
        ps = lin.predict_proba(A)[:, 1]
        nhan = (ps >= nguong).astype(int)
        B = clone(tien).fit(lo).transform(lo)
        if list(B.columns) != list(A.columns):
            print("  lô %3d dòng | %d đặc trưng thay vì %d -> ValueError"
                  % (n, B.shape[1], A.shape[1]))
            print("    mất %s" % sorted(set(A.columns) - set(B.columns)))
            continue
        ps2 = lin.predict_proba(B)[:, 1]
        nhan2 = (ps2 >= nguong).astype(int)
        print("  lô %3d dòng | đổi %2d/%3d nhãn | lệch xác suất lớn nhất %.4f"
              % (n, int((nhan != nhan2).sum()), n,
                 float(np.abs(ps2 - ps).max())))
        print("    lệch F1 %+.4f | recall %.4f -> %.4f"
              % (f1_score(y_lo, nhan2) - f1_score(y_lo, nhan),
                 recall_score(y_lo, nhan), recall_score(y_lo, nhan2)))


def main() -> None:
    X, y = bang_co_o_thieu()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    print("\n-- 8.3.4 lạm phát khi bước tiền xử lý nằm ngoài xác thực chéo --")
    lam_phat(X, y, skf)

    print("\n-- 8.3.4 khớp lại bộ tiền xử lý trên lô dữ liệu mới --")
    khop_lai_tren_lo_moi()


if __name__ == "__main__":
    main()
