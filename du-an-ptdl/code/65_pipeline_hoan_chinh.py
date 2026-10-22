"""Gom các bước thành một Pipeline với ColumnTransformer hai nhánh (mục 8.3.1, 8.3.2)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def dung_pipeline():
    """Hai nhánh, mỗi nhánh là một Pipeline con: điền trước, biến đổi sau."""
    nhanh_so = Pipeline([("dien", SimpleImputer(strategy="median")),
                         ("thang", StandardScaler())])
    nhanh_pl = Pipeline([("dien", SimpleImputer(strategy="constant",
                                                fill_value="Không rõ")),
                         ("ma", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))])
    tien = ColumnTransformer([("so", nhanh_so, COT_SO), ("pl", nhanh_pl, COT_PL)])
    tien.set_output(transform="pandas")   # giữ tên cột, nhờ đó phép kiểm tên cột còn hiệu lực
    return Pipeline([("tien", tien), ("mo_hinh", LogisticRegression(max_iter=1000))])


def doc_cau_truc(ong):
    """Số bước, tên bước và tên đặc trưng ra: ba thứ cần đọc được trước khi lưu."""
    tien = ong.named_steps["tien"]
    ra = list(tien.get_feature_names_out())
    print("  pipeline ngoài: %d bước %s" % (len(ong.steps), [t for t, _ in ong.steps]))
    print("  nhánh so: %s | nhánh pl: %s"
          % ([t for t, _ in tien.named_transformers_["so"].steps],
             [t for t, _ in tien.named_transformers_["pl"].steps]))
    print("  %d cột vào -> %d đặc trưng ra" % (len(tien.feature_names_in_), len(ra)))
    print("  8 đặc trưng đầu ra : %s" % ", ".join(ra[:8]))
    print("  3 đặc trưng cuối ra: %s" % ", ".join(ra[-3:]))
    print("  thứ tự đặc trưng ra theo THỨ TỰ KHAI BÁO NHÁNH, không theo thứ tự cột vào:"
          " cột vào đầu tiên là %s" % tien.feature_names_in_[0])


def so_voi_mo_hinh_chuong_7(ong, X_te, y_te):
    """Hai bước điền thêm vào có đổi mô hình hay không: trả lời bằng ba con số."""
    cu = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib")
    nguong = float(cu.best_threshold_)
    ps_moi = ong.predict_proba(X_te)[:, 1]
    ps_cu = cu.predict_proba(X_te)[:, 1]
    nhan_moi = (ps_moi >= nguong).astype(int)
    nhan_cu = cu.predict(X_te)
    print("  đổi %d/%d nhãn | lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
          % (int((nhan_moi != nhan_cu).sum()), len(y_te),
             float(np.abs(ps_moi - ps_cu).max()),
             f1_score(y_te, nhan_moi) - f1_score(y_te, nhan_cu)))
    print("  F1 %.4f | recall %.4f | precision %.4f | ma trận %s"
          % (f1_score(y_te, nhan_moi), recall_score(y_te, nhan_moi),
             precision_score(y_te, nhan_moi), confusion_matrix(y_te, nhan_moi).tolist()))
    return nguong


def chiu_duoc_o_thieu(ong, X_te, nguong):
    """Cái mà hai bước điền mua được: bản ghi khuyết ô vẫn dự đoán được."""
    mot = X_te.iloc[[0]]
    khuyet = mot.assign(diem_giua_ky=np.nan, thiet_bi_chinh=None)
    cu = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib")
    try:
        cu.predict_proba(khuyet)
    except ValueError as loi:
        print("  mô hình Chương 7 với bản ghi khuyết hai ô -> ValueError: %s"
              % str(loi).splitlines()[0])
    p_du = ong.predict_proba(mot)[0, 1]
    p_khuyet = ong.predict_proba(khuyet)[0, 1]
    print("  pipeline Chương 8: bản đầy đủ %.4f -> bản khuyết hai ô %.4f"
          " | lệch %.4f | nhãn %d -> %d"
          % (p_du, p_khuyet, abs(p_khuyet - p_du), int(p_du >= nguong),
             int(p_khuyet >= nguong)))
    gia_tri_dien = ong.named_steps["tien"].named_transformers_["so"] \
        .named_steps["dien"].statistics_
    print("  giá trị điền của 8 cột số (trung vị tập huấn luyện): %s"
          % np.round(gia_tri_dien, 2).tolist())


def thu_tu_trong_nhanh():
    """Đảo thứ tự hai bước trong nhánh số: không sinh lỗi, nhưng đổi kết quả."""
    sach = pd.read_csv(VAO / "du-lieu-sach.csv")
    tho = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv")
    X = sach[COT_SO].copy()
    for cot in ("ty_le_hoan_thanh_video", "diem_giua_ky"):
        X.loc[sach["ma_sv"].isin(set(tho.loc[tho[cot].isna(), "ma_sv"])), cot] = np.nan
    dung = Pipeline([("dien", SimpleImputer(strategy="median")),
                     ("thang", StandardScaler())]).fit_transform(X)
    dao = Pipeline([("thang", StandardScaler()),
                    ("dien", SimpleImputer(strategy="median"))]).fit_transform(X)
    print("  ô thiếu trên 8 cột số: %d | đảo thứ tự: chạy được, không cảnh báo"
          % int(X.isna().sum().sum()))
    print("  lệch giá trị lớn nhất %.4f | số cột lệch quá 0,01: %d trên 8"
          % (float(np.abs(dung - dao).max()),
             int((np.abs(dung - dao).max(axis=0) > 0.01).sum())))
    print("  độ lệch chuẩn sau biến đổi, thứ tự đúng: %s"
          % np.round(dung.std(axis=0), 4).tolist())
    print("  độ lệch chuẩn sau biến đổi, thứ tự đảo: %s"
          % np.round(dao.std(axis=0), 4).tolist())


def dem_cho_co_the_sai(ong):
    """Cái giá của cách gọi rời: mỗi thứ phải nêu lại là một chỗ có thể lệch."""
    tien = ong.named_steps["tien"]
    nhanh_so = tien.named_transformers_["so"]
    nhanh_pl = tien.named_transformers_["pl"]
    hang_muc = {
        "danh sách cột vào theo thứ tự": len(tien.feature_names_in_),
        "giá trị điền cột số": len(nhanh_so.named_steps["dien"].statistics_),
        "trung bình và độ lệch chuẩn": 2 * len(nhanh_so.named_steps["thang"].mean_),
        "mức của cột phân loại": sum(len(m) for m in
                                     nhanh_pl.named_steps["ma"].categories_),
        "hệ số và hệ số tự do": ong.named_steps["mo_hinh"].coef_.size + 1,
        "ngưỡng quyết định": 1,
    }
    for ten, n in hang_muc.items():
        print("    %-32s %3d giá trị" % (ten, n))
    print("  gọi rời: 6 hạng mục, %d giá trị phải khớp giữa chương trình"
          % sum(hang_muc.values()))
    print("  huấn luyện và chương trình dự đoán")
    print("  gọi bằng pipeline: 1 file, 0 giá trị phải nêu lại")


def main() -> None:
    X_tr, X_te, y_tr, y_te = nap_goi()
    ong = dung_pipeline().fit(X_tr, y_tr)

    print("-- 8.3.2 cấu trúc của pipeline vừa khớp --")
    doc_cau_truc(ong)

    print("\n-- 8.3.2 ba con số so với mô hình đã chốt ở Chương 7 --")
    nguong = so_voi_mo_hinh_chuong_7(ong, X_te, y_te)

    print("\n-- 8.3.2 bản ghi khuyết ô --")
    chiu_duoc_o_thieu(ong, X_te, nguong)

    print("\n-- 8.3.2 đảo thứ tự hai bước trong nhánh số --")
    thu_tu_trong_nhanh()

    print("\n-- 8.3.1 số giá trị phải nêu lại nếu gọi rời từng bước --")
    dem_cho_co_the_sai(ong)
    joblib.dump(ong, THU_MUC_MO_HINH / "ch08-pipeline-khong-nguong.joblib")
    print("  đã lưu models/ch08-pipeline-khong-nguong.joblib")


if __name__ == "__main__":
    main()
