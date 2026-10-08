"""Dựng bài toán phân loại `ket_qua`, chia phân tầng, và hai mô hình mốc (mục 6.1)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
          "hinh_thuc_hoc", "thiet_bi_chinh"]
# Năm cột không được làm đặc trưng của bài toán phân loại. Bốn cột đầu đã bị loại từ
# Chương 4; `diem_cuoi_ky` là cột mới phải loại, vì `ket_qua` chính là phép so nó với 4,0.
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"


def nap_goi():
    """Nạp bảng đầy đủ đã làm sạch và bộ tiền xử lý đã khớp của Chương 4."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return bang, tien


def dung_bai_toan(bang):
    """Tách đặc trưng và nhãn nhị phân, rồi chia phân tầng theo nhãn."""
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def nghiem_thu(bang, X_tr, X_te, y_tr, y_te) -> None:
    """Bốn con số phải kiểm trước khi huấn luyện mô hình phân loại nào."""
    print("Bảng đầy đủ", bang.shape, "| đặc trưng", X_tr.shape[1], "cột")
    print("Tập huấn luyện", X_tr.shape, "| tập kiểm tra", X_te.shape)
    print("Lớp dương '%s': cả bảng %d (%.2f%%) | huấn luyện %d (%.2f%%)"
          " | kiểm tra %d (%.2f%%)"
          % (LOP_DUONG, int(y_tr.sum() + y_te.sum()),
             float((y_tr.sum() + y_te.sum()) / (len(y_tr) + len(y_te))) * 100,
             int(y_tr.sum()), float(y_tr.mean()) * 100,
             int(y_te.sum()), float(y_te.mean()) * 100))
    print("Số ô thiếu trong đặc trưng:", int(X_tr.isna().sum().sum()),
          "| số dòng hai tập cộng lại:", len(X_tr) + len(X_te))


def nam_do_do(ten: str, y_thuc, nhan, diem=None) -> dict:
    """Năm độ đo dùng chung cho mọi mô hình phân loại của chương."""
    ket = {"mo_hinh": ten,
           "accuracy": round(accuracy_score(y_thuc, nhan), 4),
           "precision": round(precision_score(y_thuc, nhan, zero_division=0), 4),
           "recall": round(recall_score(y_thuc, nhan), 4),
           "F1": round(f1_score(y_thuc, nhan), 4)}
    ket["AUC"] = round(roc_auc_score(y_thuc, diem), 4) if diem is not None else None
    return ket


def tien_xu_ly_co_ro_ri() -> ColumnTransformer:
    """Bộ tiền xử lý CỐ Ý SAI: giữ `diem_cuoi_ky` trong nhóm đặc trưng số."""
    return ColumnTransformer([
        ("so", StandardScaler(), COT_SO + ["diem_cuoi_ky"]),
        ("pl", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COT_PL)])


def do_ro_ri(bang, tien) -> None:
    """Đo hậu quả của việc giữ `diem_cuoi_ky` làm đặc trưng."""
    X = bang.drop(columns=["ma_sv", "ngay_dang_ky", "tinh_thanh", "ket_qua"])
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    A, B, a, b = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for ten, goc in (("hồi quy logistic", LogisticRegression(max_iter=1000)),
                     ("rừng ngẫu nhiên", RandomForestClassifier(n_estimators=200,
                                                                random_state=42))):
        mo = Pipeline([("tien", tien_xu_ly_co_ro_ri()), ("mo_hinh", clone(goc))]).fit(A, a)
        sach = Pipeline([("tien", clone(tien)), ("mo_hinh", clone(goc))]).fit(A, a)
        nh, nh_sach = mo.predict(B), sach.predict(B)
        print("%-18s accuracy %.4f | recall %.4f | ma trận nhầm lẫn %s"
              % (ten, accuracy_score(b, nh), recall_score(b, nh),
                 confusion_matrix(b, nh).tolist()))
        print("%-18s ba con số so với bản bỏ cột: đổi %d/%d nhãn"
              " | lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
              % ("", int((nh != nh_sach).sum()), len(b),
                 float(np.abs(mo.predict_proba(B)[:, 1]
                              - sach.predict_proba(B)[:, 1]).max()),
                 f1_score(b, nh) - f1_score(b, nh_sach)))


def dung_moc(X_tr, X_te, y_tr, y_te) -> None:
    """Bốn mô hình mốc của scikit-learn, trên cùng tập kiểm tra."""
    dong = []
    for cach, nhan in (("most_frequent", "luôn đoán lớp hay gặp nhất"),
                       ("stratified", "đoán ngẫu nhiên theo tỉ lệ lớp"),
                       ("uniform", "đoán ngẫu nhiên hai lớp đều nhau")):
        moc = DummyClassifier(strategy=cach, random_state=42).fit(X_tr, y_tr)
        nh = moc.predict(X_te)
        dong.append(nam_do_do(nhan, y_te, nh, moc.predict_proba(X_te)[:, 1]))
    print(pd.DataFrame(dong).to_string(index=False))
    moc = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    print("Ma trận nhầm lẫn của mốc luôn đoán lớp hay gặp nhất:",
          confusion_matrix(y_te, moc.predict(X_te)).tolist())


def main() -> None:
    bang, tien = nap_goi()
    X_tr, X_te, y_tr, y_te = dung_bai_toan(bang)
    print("-- Nghiệm thu bài toán phân loại --")
    nghiem_thu(bang, X_tr, X_te, y_tr, y_te)
    print("Số đặc trưng sau biến đổi:", clone(tien).fit(X_tr).transform(X_te).shape[1])
    print("\n-- Hậu quả của rò rỉ: giữ diem_cuoi_ky làm đặc trưng --")
    do_ro_ri(bang, tien)
    print("\n-- Ba mô hình mốc --")
    dung_moc(X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    main()
