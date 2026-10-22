"""Hợp đồng dữ liệu vào, ba lỗi hay gặp, dự đoán một bản ghi và theo lô (mục 8.2)."""
import sys
import textwrap
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
RA = GOC_DU_AN / "output"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
MOT_SINH_VIEN = {"nganh": "Kế toán", "gioi_tinh": "Nữ",
                 "nhom_tuoi": "23-30", "vua_lam_vua_hoc": "Có",
                 "hinh_thuc_hoc": "Trực tuyến",
                 "thiet_bi_chinh": "Điện thoại",
                 "so_lan_dang_nhap": 9, "tong_thoi_luong_phut": 300,
                 "ty_le_hoan_thanh_video": 0.21,
                 "so_bai_tap_nop": 3, "diem_tb_bai_tap": 3.0,
                 "so_lan_dien_dan": 0, "nop_tre_tb_gio": 22.0,
                 "diem_giua_ky": 2.5}


def nap_goi():
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib")
    return X_tr, X_te, y_tr, y_te, mo_hinh


def in_theo_nhom(ten, thut, moi_dong=3):
    """In một danh sách tên thành nhiều dòng ngắn, mỗi dòng vài tên."""
    for i in range(0, len(ten), moi_dong):
        print(thut + ", ".join(ten[i:i + moi_dong]))


def in_ngat(s, thut="      ", rong=60):
    """In một chuỗi dài thành nhiều dòng ngắn, chỉ ngắt ở khoảng trắng."""
    for dong in textwrap.wrap(s, rong):
        print(thut + dong)


def hop_dong_vao(mo_hinh, X_tr):
    """Hợp đồng dữ liệu vào đọc được từ chính mô hình, không phải từ tài liệu viết tay."""
    tien = mo_hinh.estimator_.named_steps["tien"]
    cot = list(tien.feature_names_in_)
    so = [c for c in cot if pd.api.types.is_numeric_dtype(X_tr[c])]
    print("  %d cột, theo thứ tự mô hình ghi nhớ:" % len(cot))
    in_theo_nhom(cot, "    ")
    print("  miền giá trị quan sát trên %d dòng huấn luyện:" % len(X_tr))
    for c in so:
        print("    %-24s %-8s [%.2f … %.2f]"
              % (c, X_tr[c].dtype, X_tr[c].min(), X_tr[c].max()))
    for c, muc in zip(tien.transformers_[1][2],
                      tien.named_transformers_["pl"].categories_):
        print("    %-24s %d mức:" % (c, len(muc)))
        in_theo_nhom([str(m) for m in muc], "      ")
    return cot


def thu_mot_ca(mo_hinh, ten, bang):
    """Gọi predict_proba và ghi lại cả lỗi lẫn cảnh báo, thay vì để chương trình dừng."""
    with warnings.catch_warnings(record=True) as ghi:
        warnings.simplefilter("always")
        print("  %s" % ten)
        try:
            p = mo_hinh.predict_proba(bang)[:, 1]
            canh_bao = ghi[0].category.__name__ if ghi else "không"
            print("      CHẠY ĐƯỢC, xác suất %.4f | cảnh báo: %s"
                  % (p[0], canh_bao))
        except ValueError as loi:
            in_ngat("ValueError: " + str(loi).splitlines()[0])


def ba_loi_hay_gap(mo_hinh, X_te, cot):
    mot = X_te.iloc[[0]]
    print("  bản ghi đúng hợp đồng")
    print("      xác suất %.4f (mốc so sánh)"
          % mo_hinh.predict_proba(mot)[0, 1])
    thu_mot_ca(mo_hinh, "thiếu cột diem_giua_ky", mot.drop(columns=["diem_giua_ky"]))
    thu_mot_ca(mo_hinh, "đảo thứ tự cả 14 cột", mot[cot[::-1]])
    thu_mot_ca(mo_hinh, "thêm một cột không có trong hợp đồng", mot.assign(diem_chuyen_can=8))
    thu_mot_ca(mo_hinh, "nganh = 'Trí tuệ nhân tạo' (mức lạ)",
               mot.assign(nganh="Trí tuệ nhân tạo"))
    thu_mot_ca(mo_hinh, "thiết bị chính để trống", mot.assign(thiet_bi_chinh=None))
    thu_mot_ca(mo_hinh, "diem_giua_ky để trống", mot.assign(diem_giua_ky=np.nan))
    thu_mot_ca(mo_hinh, "đưa vào mảng NumPy thay vì DataFrame", mot.to_numpy())
    thu_mot_ca(mo_hinh, "diem_giua_ky = '3,6' (dấu phẩy)", mot.assign(diem_giua_ky="3,6"))


def thu_tu_cot_voi_mo_hinh_tran(mo_hinh, X_te, y_te):
    """Đảo thứ tự cột: pipeline chặn được, mô hình trần thì không, và hậu quả đo được."""
    ong = mo_hinh.estimator_
    tien, lin = ong.named_steps["tien"], ong.named_steps["mo_hinh"]
    nguong = float(mo_hinh.best_threshold_)
    A = tien.transform(X_te)
    ps = lin.predict_proba(A)[:, 1]
    nhan = (ps >= nguong).astype(int)
    try:
        lin.predict_proba(A.iloc[:, ::-1])
    except ValueError as loi:
        print("  mô hình trần, DataFrame đảo cột -> ValueError:")
        print("    %s" % str(loi).splitlines()[1])
    with warnings.catch_warnings(record=True) as ghi:
        warnings.simplefilter("always")
        lin.predict_proba(A.to_numpy()[:, ::-1])
        print("  mô hình trần, mảng NumPy đảo cột -> chỉ %s" % ghi[0].category.__name__)
        for ten, mang in (("đảo cả 30 cột", A.to_numpy()[:, ::-1]),
                          ("đổi chỗ đúng hai cột 4 và 5", doi_cho_hai_cot(A.to_numpy(), 3, 4))):
            ps2 = lin.predict_proba(mang)[:, 1]
            nhan2 = (ps2 >= nguong).astype(int)
            print("    %-28s đổi %3d/%d nhãn"
                  % (ten, int((nhan != nhan2).sum()), len(y_te)))
            print("      lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
                  % (float(np.abs(ps2 - ps).max()),
                     f1_score(y_te, nhan2) - f1_score(y_te, nhan)))
            print("      recall %.4f -> %.4f" % (recall_score(y_te, nhan),
                                                 recall_score(y_te, nhan2)))


def doi_cho_hai_cot(mang, i, j):
    ra = mang.copy()
    ra[:, [i, j]] = ra[:, [j, i]]
    return ra


def muc_la_ba_con_so(mo_hinh, X_te, y_te):
    """handle_unknown='ignore' không báo lỗi, nên hậu quả chỉ thấy được bằng số đo."""
    ps = mo_hinh.predict_proba(X_te)[:, 1]
    nhan = mo_hinh.predict(X_te)
    ca = {"cả 603 dòng mang mức lạ": X_te.assign(nganh="Trí tuệ nhân tạo"),
          "30 dòng đầu mang mức lạ": pd.concat(
              [X_te.iloc[:30].assign(nganh="Trí tuệ nhân tạo"), X_te.iloc[30:]]),
          "cột gioi_tinh viết hoa hết": X_te.assign(gioi_tinh=X_te["gioi_tinh"].str.upper()),
          "cột thiet_bi_chinh để trống": X_te.assign(thiet_bi_chinh=None)}
    for ten, bang in ca.items():
        ps2 = mo_hinh.predict_proba(bang)[:, 1]
        nhan2 = mo_hinh.predict(bang)
        print("  %-30s đổi %3d/%d nhãn"
              % (ten, int((nhan != nhan2).sum()), len(y_te)))
        print("      lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
              % (float(np.abs(ps2 - ps).max()),
                 f1_score(y_te, nhan2) - f1_score(y_te, nhan)))


def ngoai_mien_gia_tri(mo_hinh):
    """Giá trị nằm ngoài miền quan sát không sinh lỗi, và đổi hẳn kết luận."""
    goc = pd.DataFrame([MOT_SINH_VIEN])
    print("  %-38s xác suất %.4f | nhãn %d"
          % ("bản ghi gốc", mo_hinh.predict_proba(goc)[0, 1], mo_hinh.predict(goc)[0]))
    for cot, gia_tri in (("tong_thoi_luong_phut", 99999), ("so_lan_dang_nhap", 1000),
                         ("ty_le_hoan_thanh_video", 85.0), ("diem_giua_ky", -1.0)):
        e = goc.assign(**{cot: gia_tri})
        print("  %-24s = %8s xác suất %.4f | nhãn %d"
              % (cot, gia_tri, mo_hinh.predict_proba(e)[0, 1], mo_hinh.predict(e)[0]))


def mot_ban_ghi(mo_hinh, cot):
    """Một bản ghi là một DataFrame MỘT DÒNG, không phải Series và không phải mảng."""
    bang = pd.DataFrame([MOT_SINH_VIEN])
    print("  pd.DataFrame([từ_điển]) -> hình dạng %s," % (bang.shape,))
    print("  thứ tự cột theo từ điển")
    print("  xác suất %.4f | nhãn %d | ngưỡng %.4f"
          % (mo_hinh.predict_proba(bang)[0, 1], mo_hinh.predict(bang)[0],
             mo_hinh.best_threshold_))
    thu_mot_ca(mo_hinh, "pd.Series(từ_điển)", pd.Series(MOT_SINH_VIEN))
    print("  reindex về thứ tự hợp đồng rồi dự đoán: xác suất %.4f"
          % mo_hinh.predict_proba(bang[cot])[0, 1])


def theo_lo(mo_hinh, X_te, y_te, cot):
    """Một lệnh cho cả lô, so với một lệnh cho mỗi dòng: cùng kết quả, khác thời gian."""
    t0 = time.perf_counter()
    ps_lo = mo_hinh.predict_proba(X_te)[:, 1]
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    ps_le = np.array([mo_hinh.predict_proba(X_te.iloc[[i]])[0, 1] for i in range(len(X_te))])
    t3 = time.perf_counter()
    print("  theo lô   %8.1f ms cho %d dòng | %.3f ms mỗi dòng"
          % ((t1 - t0) * 1000, len(X_te), (t1 - t0) * 1000 / len(X_te)))
    print("  từng dòng %8.1f ms cho %d dòng | %.3f ms mỗi dòng"
          % ((t3 - t2) * 1000, len(X_te), (t3 - t2) * 1000 / len(X_te)))
    print("  chậm hơn %.0f lần | kết quả lệch lớn nhất %.2e"
          % ((t3 - t2) / (t1 - t0), float(np.abs(ps_le - ps_lo).max())))
    RA.mkdir(parents=True, exist_ok=True)
    ket = pd.DataFrame({"xac_suat": ps_lo.round(4), "nhan": mo_hinh.predict(X_te)},
                       index=X_te.index).sort_values("xac_suat", ascending=False)
    ket.to_csv(RA / "ch08-du-doan-theo-lo.csv", index_label="chi_muc", encoding="utf-8")
    print("  đã ghi output/ch08-du-doan-theo-lo.csv | %d dòng vượt ngưỡng"
          % int(ket["nhan"].sum()))


def main() -> None:
    X_tr, X_te, y_tr, y_te, mo_hinh = nap_goi()

    print("-- 8.2.1 hợp đồng dữ liệu vào, đọc từ chính mô hình --")
    cot = hop_dong_vao(mo_hinh, X_tr)

    print("\n-- 8.2.2 tám cách làm sai, cùng một bản ghi --")
    ba_loi_hay_gap(mo_hinh, X_te, cot)

    print("\n-- 8.2.2 thứ tự cột: chỗ pipeline chặn được và chỗ không --")
    thu_tu_cot_voi_mo_hinh_tran(mo_hinh, X_te, y_te)

    print("\n-- 8.2.2 mức phân loại lạ: ba con số cho từng ca --")
    muc_la_ba_con_so(mo_hinh, X_te, y_te)

    print("\n-- 8.2.2 giá trị ngoài miền quan sát --")
    ngoai_mien_gia_tri(mo_hinh)

    print("\n-- 8.2.3 dự đoán một bản ghi --")
    mot_ban_ghi(mo_hinh, cot)

    print("\n-- 8.2.3 dự đoán theo lô --")
    theo_lo(mo_hinh, X_te, y_te, cot)


if __name__ == "__main__":
    main()
