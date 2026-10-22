"""Sinh các file lô dữ liệu mới dùng cho Chương 9 (mục 9.1.1).

Lô mới không có nhãn thật: nó là bảng mà phòng đào tạo gửi tới sau khi một đợt đăng
ký đóng. Ở đây lô được dựng lại từ 40 dòng đầu của tập kiểm tra Chương 7 để mọi con
số của Chương 9 tái lập được, kèm cột ma_sv mà mô hình không dùng.

Bốn lỗi của file ch09-lo-loi.csv được đặt vào những dòng CHỌN TRƯỚC theo xác suất,
không đặt vào dòng bất kỳ: hai ô chữ nằm đúng ở hai dòng vượt ngưỡng, nhờ đó mục
9.2.2 đo được cái mất khi bỏ dòng sai thay vì chỉ nói rằng có mất.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
RA = GOC_DU_AN / "data" / "raw"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
SO_DONG = 40


def lo_sach(so_dong=SO_DONG):
    """40 dòng đầu của tập kiểm tra, thêm lại cột ma_sv để người nhận biết là ai."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang["ket_qua"] == "Không đạt").astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2,
                                        random_state=42, stratify=y)
    lo = X_te.iloc[:so_dong].copy()
    lo.insert(0, "ma_sv", bang.loc[lo.index, "ma_sv"])
    return lo.reset_index(drop=True), y_te.iloc[:so_dong].reset_index(drop=True)


def chon_dong_dat_loi(lo):
    """Xếp các dòng theo xác suất giảm dần để đặt lỗi vào đúng chỗ muốn đo."""
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib")
    ho_so = json.loads((THU_MUC_MO_HINH / "ch08-ho-so-trien-khai.json")
                       .read_text(encoding="utf-8"))
    p = mo_hinh.predict_proba(lo[ho_so["cot_vao_theo_thu_tu"]])[:, 1]
    xep = list(np.argsort(-p))
    vuot = [int(i) for i in xep if p[i] >= ho_so["nguong_quyet_dinh"]]
    duoi = [int(i) for i in xep if p[i] < ho_so["nguong_quyet_dinh"]]
    print("  %d/%d dòng vượt ngưỡng %.4f" % (len(vuot), len(p), ho_so["nguong_quyet_dinh"]))
    print("  xác suất lớn nhất %.4f, nhỏ nhất %.4f" % (p[xep[0]], p[xep[-1]]))
    return p, vuot, duoi


def lo_co_loi(lo, vuot, duoi):
    """Bốn loại lỗi, mỗi loại ở những dòng biết trước nên đếm lại được."""
    xau = lo.copy()
    xau["so_lan_dang_nhap"] = xau["so_lan_dang_nhap"].astype(object)
    o_chu = [vuot[3], vuot[5]]                      # 2 ô chữ, đặt ở dòng vượt ngưỡng
    o_trong = duoi[:3]                              # 3 ô trống ở cột số
    muc_la, ngoai_mien = vuot[1], vuot[4]           # 1 mức lạ, 1 giá trị ngoài miền
    xau.loc[o_chu, "so_lan_dang_nhap"] = "mười"
    xau.loc[o_trong, "diem_giua_ky"] = None
    xau.loc[[muc_la], "nganh"] = "Trí tuệ nhân tạo"
    xau.loc[[ngoai_mien], "ty_le_hoan_thanh_video"] = 3.5
    for ten, ds in [("ô chữ", o_chu), ("ô trống", o_trong),
                    ("mức lạ", [muc_la]), ("ngoài miền", [ngoai_mien])]:
        print("  %-12s đặt ở dòng %s của file" % (ten, [i + 2 for i in ds]))
    return xau, o_chu


def ghi(lo, xau):
    RA.mkdir(parents=True, exist_ok=True)
    p1 = RA / "ch09-lo-moi.csv"
    lo.to_csv(p1, index=False, encoding="utf-8")
    p2 = RA / "ch09-lo-loi.csv"
    xau.to_csv(p2, index=False, encoding="utf-8")
    p3 = RA / "ch09-lo-thieu-cot.csv"
    lo.drop(columns=["diem_giua_ky"]).to_csv(p3, index=False, encoding="utf-8")
    p4 = RA / "ch09-lo-excel-viet.csv"
    lo.to_csv(p4, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    for p in (p1, p2, p3, p4):
        print("  %-28s %5d byte" % (p.name, p.stat().st_size))


def main() -> None:
    RA.mkdir(parents=True, exist_ok=True)
    lo, y_lo = lo_sach()
    print("-- 9.1.1 lô mới: %d dòng, %d cột --" % lo.shape)
    print("  cột đầu %s, cột cuối %s" % (lo.columns[0], lo.columns[-1]))
    print("  nhãn thật của lô, chỉ dùng để đối chiếu: %d/%d không đạt"
          % (int(y_lo.sum()), len(y_lo)))
    y_lo.to_frame("ket_qua_that").to_csv(RA / "ch09-lo-moi-nhan-that.csv", index=False)
    print("\n-- 9.2.2 chọn chỗ đặt lỗi theo xác suất --")
    _, vuot, duoi = chon_dong_dat_loi(lo)
    xau, _ = lo_co_loi(lo, vuot, duoi)
    print("\n-- bốn file lô của Chương 9 --")
    ghi(lo, xau)


if __name__ == "__main__":
    main()
