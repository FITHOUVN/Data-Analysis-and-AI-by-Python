"""Mô hình đã khớp gồm những gì, lưu và tải bằng joblib, hồ sơ mô hình đi kèm (mục 8.1)."""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.base
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
TAM = GOC_DU_AN / "models" / "tam"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]


def nap_goi():
    """Dựng lại đúng phép chia của Chương 7 để mọi con số so được với bảng của chương đó."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch07-mo-hinh-chot.joblib")
    return X_tr, X_te, y_tr, y_te, mo_hinh


def ben_trong(mo_hinh):
    """Liệt kê đúng những đại lượng mà lệnh fit đã ghi vào đối tượng."""
    ong = mo_hinh.estimator_
    tien = ong.named_steps["tien"]
    lin = ong.named_steps["mo_hinh"]
    thang = tien.named_transformers_["so"]
    ma = tien.named_transformers_["pl"]
    so_muc = sum(len(m) for m in ma.categories_)
    print("  hệ số hồi quy          %2d con số" % lin.coef_.size)
    print("  hệ số tự do            %2d con số  %s"
          % (lin.intercept_.size, np.round(lin.intercept_, 4).tolist()))
    print("  trung bình 8 cột số    %2d con số" % thang.mean_.size)
    print("  độ lệch chuẩn 8 cột số %2d con số" % thang.scale_.size)
    print("  tổng tham số đã học    %2d con số"
          % (lin.coef_.size + lin.intercept_.size + thang.mean_.size + thang.scale_.size))
    print("  mức của 6 cột phân loại %d chuỗi, theo thứ tự đã ghi" % so_muc)
    print("  tên cột vào            %2d tên, thứ tự:"
          % len(tien.feature_names_in_))
    print("    %s …" % ", ".join(tien.feature_names_in_[:3]))
    print("  đặc trưng ra           %2d tên" % len(tien.get_feature_names_out()))
    print("  ngưỡng quyết định      %.4f  (không phải tham số của mô hình)"
          % mo_hinh.best_threshold_)
    return tien, lin


def hai_thu_tu_cot(tien):
    """Hai bộ tiền xử lý của cùng dự án ghi nhớ hai thứ tự cột khác nhau."""
    cu = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    a, b = list(cu.feature_names_in_), list(tien.feature_names_in_)
    print("  bo-tien-xu-ly.joblib    : %s …" % ", ".join(a[:3]))
    print("  ch07-mo-hinh-chot.joblib: %s …" % ", ".join(b[:3]))
    print("  cùng %d tên, trùng thứ tự: %s | số vị trí lệch: %d"
          % (len(a), sorted(a) == sorted(b) and a == b, sum(x != y for x, y in zip(a, b))))
    print("  thứ tự đặc trưng RA thì giống nhau: %s"
          % (list(cu.get_feature_names_out()) == list(tien.get_feature_names_out())))


def kich_thuoc_va_thoi_gian(mo_hinh, X_tr, y_tr):
    """Cái giá của mỗi lựa chọn lưu: dung lượng file, thời gian ghi, thời gian nạp."""
    TAM.mkdir(parents=True, exist_ok=True)
    ong = mo_hinh.estimator_
    rung = Pipeline([("tien", clone(ong.named_steps["tien"])),
                     ("mo_hinh", RandomForestClassifier(n_estimators=200,
                                                        random_state=42))]).fit(X_tr, y_tr)
    for ten, doi_tuong in (("chỉ LogisticRegression", ong.named_steps["mo_hinh"]),
                           ("chỉ bộ tiền xử lý", ong.named_steps["tien"]),
                           ("pipeline, không ngưỡng", ong),
                           ("cả đối tượng có ngưỡng", mo_hinh),
                           ("pipeline rừng 200 cây", rung)):
        for nen in (0, 3):
            p = TAM / ("thu-nen%d.joblib" % nen)
            t0 = time.perf_counter()
            joblib.dump(doi_tuong, p, compress=nen)
            t1 = time.perf_counter()
            t2 = time.perf_counter()
            joblib.load(p)
            t3 = time.perf_counter()
            print("  %-24s nén %d | %8.1f KB | ghi %6.1f ms | nạp %6.1f ms"
                  % (ten, nen, p.stat().st_size / 1024, (t1 - t0) * 1000, (t3 - t2) * 1000))
            p.unlink()


def ho_so_mo_hinh(mo_hinh, X_tr, X_te, y_te):
    """Những thứ phải lưu kèm: cái gì mất đi khi chỉ còn file .joblib."""
    ong = mo_hinh.estimator_
    tien = ong.named_steps["tien"]
    nhan = mo_hinh.predict(X_te)
    ho_so = {
        "ten_mo_hinh": "ch08-mo-hinh-kem-nguong",
        "nguon_du_lieu": "data/processed/ch04-bang-day-du.parquet",
        "so_dong_huan_luyen": int(len(X_tr)),
        "cach_chia": "train_test_split(test_size=0.2, random_state=42, stratify=y)",
        "cot_vao_theo_thu_tu": list(tien.feature_names_in_),
        "muc_tieu": "ket_qua == 'Không đạt' -> 1",
        "nguong_quyet_dinh": float(mo_hinh.best_threshold_),
        "do_do_chon_nguong": "f1, StratifiedKFold(5, shuffle=True, random_state=42)",
        "do_do_tren_tap_kiem_tra": {"f1": round(float(f1_score(y_te, nhan)), 4),
                                    "recall": round(float(recall_score(y_te, nhan)), 4)},
        "phien_ban": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__,
                      "joblib": joblib.__version__},
    }
    duong_dan = THU_MUC_MO_HINH / "ch08-ho-so-mo-hinh.json"
    duong_dan.write_text(json.dumps(ho_so, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  đã ghi %s, %d khóa" % (duong_dan.name, len(ho_so)))
    print("  ngưỡng %.4f | %d cột vào | F1 %.4f"
          % (ho_so["nguong_quyet_dinh"], len(ho_so["cot_vao_theo_thu_tu"]),
             ho_so["do_do_tren_tap_kiem_tra"]["f1"]))
    return ho_so


def mat_nguong(mo_hinh, X_te, y_te):
    """Ba con số của lỗi lưu thiếu ngưỡng: lưu pipeline thay vì lưu cả đối tượng."""
    ong = mo_hinh.estimator_
    nhan_dung, nhan_thieu = mo_hinh.predict(X_te), ong.predict(X_te)
    ps_dung = mo_hinh.predict_proba(X_te)[:, 1]
    ps_thieu = ong.predict_proba(X_te)[:, 1]
    print("  đổi %d/%d nhãn | lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
          % (int((nhan_dung != nhan_thieu).sum()), len(y_te),
             float(np.abs(ps_dung - ps_thieu).max()),
             f1_score(y_te, nhan_thieu) - f1_score(y_te, nhan_dung)))
    print("  recall %.4f -> %.4f | bỏ sót %d -> %d sinh viên"
          % (recall_score(y_te, nhan_dung), recall_score(y_te, nhan_thieu),
             int(((nhan_dung == 0) & (y_te == 1)).sum()),
             int(((nhan_thieu == 0) & (y_te == 1)).sum())))


def gia_lap_file_phien_ban_cu(mo_hinh, X_te):
    """Ghi một file mang số phiên bản khác rồi nạp lại, để thấy cảnh báo thật."""
    TAM.mkdir(parents=True, exist_ok=True)
    p = TAM / "gia-lap-phien-ban-cu.joblib"
    sklearn.base.__version__ = "1.4.2"          # chỉ để sinh file thử
    joblib.dump(mo_hinh, p)
    sklearn.base.__version__ = sklearn.__version__
    with warnings.catch_warnings(record=True) as ghi:
        warnings.simplefilter("always", InconsistentVersionWarning)
        lai = joblib.load(p)
        print("  số cảnh báo: %d | phiên bản ghi trong file: %s | đang dùng: %s"
              % (len(ghi), ghi[0].message.original_sklearn_version, sklearn.__version__))
        print("  %s" % str(ghi[0].message).splitlines()[0])
    cot = list(lai.estimator_.named_steps["tien"].feature_names_in_)
    print("  vẫn dự đoán được: %s" % lai.predict(X_te[cot])[:5].tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            joblib.load(p)
        except InconsistentVersionWarning:
            print("  đổi cảnh báo thành lỗi: lệnh nạp dừng lại ngay")
    p.unlink()


def kiem_tra_sau_khi_nap(mo_hinh, X_te, ho_so):
    """Phép kiểm tra kết quả cuối cùng của mục 8.1.2: nạp lại rồi so từng phần tử."""
    p = THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib"
    joblib.dump(mo_hinh, p)
    lai = joblib.load(p)
    cot = ho_so["cot_vao_theo_thu_tu"]
    a, b = mo_hinh.predict_proba(X_te[cot])[:, 1], lai.predict_proba(X_te[cot])[:, 1]
    print("  tên cột khớp từng phần tử: %s"
          % (list(lai.estimator_.named_steps["tien"].feature_names_in_) == cot))
    print("  ngưỡng khớp: %s | xác suất lệch lớn nhất %.2e"
          % (float(lai.best_threshold_) == ho_so["nguong_quyet_dinh"],
             float(np.abs(a - b).max())))
    print("  nhãn khớp từng dòng: %s"
          % bool((mo_hinh.predict(X_te[cot]) == lai.predict(X_te[cot])).all()))


def main() -> None:
    X_tr, X_te, y_tr, y_te, mo_hinh = nap_goi()

    print("-- 8.1.1 những gì lệnh fit đã ghi vào đối tượng --")
    tien, _ = ben_trong(mo_hinh)

    print("\n-- 8.1.1 thứ tự cột mà hai file mô hình của dự án ghi nhớ --")
    hai_thu_tu_cot(tien)

    print("\n-- 8.1.2 dung lượng và thời gian của từng lựa chọn lưu --")
    kich_thuoc_va_thoi_gian(mo_hinh, X_tr, y_tr)

    print("\n-- 8.1.2 hồ sơ mô hình đi kèm file .joblib --")
    ho_so = ho_so_mo_hinh(mo_hinh, X_tr, X_te, y_te)

    print("\n-- 8.1.2 ba con số của lỗi lưu thiếu ngưỡng --")
    mat_nguong(mo_hinh, X_te, y_te)

    print("\n-- 8.1.2 file .joblib mang số phiên bản khác --")
    gia_lap_file_phien_ban_cu(mo_hinh, X_te)

    print("\n-- 8.1.2 phép kiểm tra kết quả cuối cùng sau khi nạp lại --")
    kiem_tra_sau_khi_nap(mo_hinh, X_te, ho_so)
    TAM.rmdir()
    print("  đã lưu models/ch08-mo-hinh-kem-nguong.joblib"
          " và models/ch08-ho-so-mo-hinh.json")


if __name__ == "__main__":
    main()
