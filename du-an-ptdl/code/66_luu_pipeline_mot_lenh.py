"""Gắn ngưỡng vào pipeline, lưu một file và dự đoán bằng một lệnh (mục 8.3.3)."""
import json
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import FixedThresholdClassifier, train_test_split

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    ong = joblib.load(THU_MUC_MO_HINH / "ch08-pipeline-khong-nguong.joblib")
    ho_so = json.loads((THU_MUC_MO_HINH / "ch08-ho-so-mo-hinh.json")
                       .read_text(encoding="utf-8"))
    return X_tr, X_te, y_tr, y_te, ong, ho_so


def gan_nguong(ong, nguong, X_tr, y_tr):
    """FrozenEstimator giữ pipeline đã khớp; FixedThresholdClassifier gắn điểm cắt."""
    san = FixedThresholdClassifier(FrozenEstimator(ong),
                                   threshold=nguong).fit(X_tr, y_tr)
    print("  ngưỡng gắn vào đối tượng: %.4f" % san.threshold)
    print("  lớp của đối tượng cuối: %s | lớp bên trong: %s"
          % (type(san).__name__, type(san.estimator).__name__))
    return san


def kiem_ba_con_so(san, X_te, y_te):
    """Gắn ngưỡng không được đổi mô hình: xác suất phải trùng tới chữ số cuối."""
    cu = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib")
    nhan, nhan_cu = san.predict(X_te), cu.predict(X_te)
    ps, ps_cu = san.predict_proba(X_te)[:, 1], cu.predict_proba(X_te)[:, 1]
    print("  đổi %d/%d nhãn | lệch xác suất lớn nhất %.2e | lệch F1 %+.4f"
          % (int((nhan != nhan_cu).sum()), len(y_te), float(np.abs(ps - ps_cu).max()),
             f1_score(y_te, nhan) - f1_score(y_te, nhan_cu)))
    print("  F1 %.4f | recall %.4f | ma trận %s"
          % (f1_score(y_te, nhan), recall_score(y_te, nhan),
             confusion_matrix(y_te, nhan).tolist()))


def luu_va_nap_lai(san, X_te, nguong, cot):
    """Một file, một lệnh nạp, một lệnh dự đoán."""
    p = THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib"
    joblib.dump(san, p)
    lai = joblib.load(p)
    nhan = lai.predict(X_te)
    print("  %s: %.1f KB" % (p.name, p.stat().st_size / 1024))
    print("  nạp một lệnh | dự đoán một lệnh")
    print("  ngưỡng sau khi nạp lại: %.4f | khớp: %s"
          % (lai.threshold, float(lai.threshold) == nguong))
    print("  nhãn khớp từng dòng với bản trong bộ nhớ: %s"
          % bool((nhan == san.predict(X_te)).all()))
    print("  tên cột mà đối tượng đòi, %d tên, khớp hồ sơ: %s"
          % (len(cot), list(lai.estimator.estimator.feature_names_in_) == cot))
    return lai


def ho_so_trien_khai(san, X_tr, X_te, y_te, ho_so_cu):
    nhan = san.predict(X_te)
    ho_so = dict(ho_so_cu)
    ho_so.update({
        "ten_mo_hinh": "ch08-mo-hinh-trien-khai",
        "cach_gan_nguong": "FixedThresholdClassifier(FrozenEstimator(pipeline),"
                           " threshold=0.2727104755205902)",
        "buoc_tien_xu_ly": ["SimpleImputer(median) + StandardScaler cho 8 cột số",
                            "SimpleImputer(constant='Không rõ')"
                            " + OneHotEncoder(handle_unknown='ignore') cho 6 cột phân loại"],
        "so_dong_huan_luyen": int(len(X_tr)),
        "do_do_tren_tap_kiem_tra": {"f1": round(float(f1_score(y_te, nhan)), 4),
                                    "recall": round(float(recall_score(y_te, nhan)), 4)},
        "phien_ban": dict(ho_so_cu["phien_ban"], **{"scikit-learn": sklearn.__version__}),
    })
    p = THU_MUC_MO_HINH / "ch08-ho-so-trien-khai.json"
    p.write_text(json.dumps(ho_so, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  đã ghi %s, %d khóa" % (p.name, len(ho_so)))


def mot_lenh_tu_dau(cot):
    """Cách gọi mà Chương 9 tới Chương 11 dùng lại: nạp một file, dự đoán một lệnh."""
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib")
    ban_ghi = pd.DataFrame([{"nganh": "Kế toán", "gioi_tinh": "Nữ", "nhom_tuoi": "23-30",
                             "vua_lam_vua_hoc": "Có", "hinh_thuc_hoc": "Trực tuyến",
                             "thiet_bi_chinh": "Điện thoại", "so_lan_dang_nhap": 9,
                             "tong_thoi_luong_phut": 300, "ty_le_hoan_thanh_video": 0.21,
                             "so_bai_tap_nop": 3, "diem_tb_bai_tap": 3.0,
                             "so_lan_dien_dan": 0, "nop_tre_tb_gio": 22.0,
                             "diem_giua_ky": 2.5}])
    print("  nhãn %d | xác suất %.4f"
          % (mo_hinh.predict(ban_ghi)[0], mo_hinh.predict_proba(ban_ghi)[0, 1]))
    print("  cùng bản ghi, thiếu hai ô: nhãn %d | xác suất %.4f"
          % (mo_hinh.predict(ban_ghi.assign(diem_giua_ky=np.nan,
                                            thiet_bi_chinh=None))[0],
             mo_hinh.predict_proba(ban_ghi.assign(diem_giua_ky=np.nan,
                                                  thiet_bi_chinh=None))[0, 1]))
    print("  cùng bản ghi, cột xếp đảo thứ tự: nhãn %d"
          % mo_hinh.predict(ban_ghi[cot[::-1]])[0])


def main() -> None:
    X_tr, X_te, y_tr, y_te, ong, ho_so = nap_goi()
    nguong = ho_so["nguong_quyet_dinh"]
    cot = ho_so["cot_vao_theo_thu_tu"]

    print("-- 8.3.3 gắn ngưỡng %.4f vào pipeline --" % nguong)
    san = gan_nguong(ong, nguong, X_tr, y_tr)

    print("\n-- 8.3.3 ba con số: gắn ngưỡng không được đổi mô hình --")
    kiem_ba_con_so(san, X_te, y_te)

    print("\n-- 8.3.3 lưu một file, nạp lại, dự đoán --")
    luu_va_nap_lai(san, X_te, nguong, cot)

    print("\n-- 8.3.3 hồ sơ của mô hình triển khai --")
    ho_so_trien_khai(san, X_tr, X_te, y_te, ho_so)
    # Ghi sau khi chốt cả mô hình và hồ sơ; giữ bản kê cùng bản phát hành tin cậy.
    ten = ["ch08-mo-hinh-trien-khai.joblib", "ch08-ho-so-trien-khai.json"]
    bam = {t: hashlib.sha256((THU_MUC_MO_HINH / t).read_bytes()).hexdigest()
           for t in ten}
    (THU_MUC_MO_HINH / "ch08-ban-ke-sha256.json").write_text(
        json.dumps(bam, indent=2), encoding="utf-8")

    print("\n-- 8.3.3 cách gọi mà các chương sau dùng lại --")
    mot_lenh_tu_dau(cot)


if __name__ == "__main__":
    main()
