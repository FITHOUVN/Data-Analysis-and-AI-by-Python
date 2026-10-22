"""Xuất kết quả dự đoán: mã hóa file, mức ưu tiên, và bản đọc được cho phòng đào tạo (mục 9.3)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
RAW = GOC_DU_AN / "data" / "raw"
RA = GOC_DU_AN / "output"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
NGUONG_MOI_MO_HINH_DONG_Y = 0.50


def nap_goi():
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib")
    ho_so = json.loads((THU_MUC_MO_HINH / "ch08-ho-so-trien-khai.json")
                       .read_text(encoding="utf-8"))
    bang = pd.read_csv(RAW / "ch09-lo-moi.csv")
    p = mo_hinh.predict_proba(bang[ho_so["cot_vao_theo_thu_tu"]])[:, 1]
    return bang, p, ho_so["nguong_quyet_dinh"]


def ba_cach_ma_hoa(bang, p):
    """Cùng một bảng kết quả, ba cách mã hóa, ba kết quả khi mở bằng Excel."""
    ra = pd.DataFrame({"ma_sv": bang["ma_sv"], "muc_uu_tien": "Liên hệ ngay"})
    for ten, ma in [("utf-8", "utf-8"), ("utf-8-sig", "utf-8-sig"), ("cp1258", "cp1258")]:
        p_file = RA / ("ch09-ma-hoa-%s.csv" % ten)
        try:
            ra.to_csv(p_file, index=False, encoding=ma)
        except UnicodeEncodeError as loi:
            print("  %-10s không ghi được: %s" % (ten, str(loi)[:48]))
            p_file.unlink(missing_ok=True)   # to_csv đã mở file trước khi dừng
            continue
        byte = p_file.read_bytes()[:3]
        doc_cp = p_file.read_bytes().decode("cp1252", errors="replace").splitlines()[1]
        print("  %-10s 3 byte đầu %-14s | đọc bằng cp1252 ra %r"
              % (ten, " ".join("%02X" % b for b in byte), doc_cp[:28]))
        p_file.unlink()


def ba_muc_uu_tien(bang, p, nguong):
    """Hai điểm cắt chia lô thành ba nhóm; nhóm giữa là nhóm điểm cắt 0,5 bỏ sót."""
    muc = pd.cut(p, [-0.01, nguong, NGUONG_MOI_MO_HINH_DONG_Y, 1.01],
                 labels=["Chưa cần", "Cần xem lại", "Liên hệ ngay"])
    dem = muc.value_counts()
    for ten in ["Liên hệ ngay", "Cần xem lại", "Chưa cần"]:
        print("  %-14s %2d dòng" % (ten, dem[ten]))
    that = pd.read_csv(RAW / "ch09-lo-moi-nhan-that.csv")["ket_qua_that"]
    nhan = (p >= nguong).astype(int)
    print("  lô có %d nhãn không đạt thật; chương trình gọi %d dòng cần liên hệ"
          % (int(that.sum()), int(nhan.sum())))
    print("  precision %.4f | recall %.4f trên lô 40 dòng"
          % (precision_score(that, nhan), recall_score(that, nhan)))
    nhan_05 = (p >= NGUONG_MOI_MO_HINH_DONG_Y).astype(int)
    print("  nếu dùng điểm cắt 0,50: %d dòng cần liên hệ, recall %.4f"
          % (int(nhan_05.sum()), recall_score(that, nhan_05)))


def ban_doc_duoc(bang, p, nguong):
    """Bản cho người không làm kỹ thuật: xếp theo nguy cơ, bỏ cột xác suất thô."""
    ra = pd.DataFrame({
        "ma_sv": bang["ma_sv"],
        "nguy_co_phan_tram": (p * 100).round(1),
        "muc_uu_tien": pd.cut(p, [-0.01, nguong, NGUONG_MOI_MO_HINH_DONG_Y, 1.01],
                              labels=["Chưa cần", "Cần xem lại", "Liên hệ ngay"]),
    }).sort_values("nguy_co_phan_tram", ascending=False)
    can = ra[ra["muc_uu_tien"] != "Chưa cần"]
    p_file = RA / "ch09-danh-sach-lien-he.csv"
    can.to_csv(p_file, index=False, encoding="utf-8-sig")
    print("  đã ghi %s, %d dòng" % (p_file.name, len(can)))
    print(can.to_string(index=False))
    cuoi = can["nguy_co_phan_tram"].iloc[-1]
    ngoai = ra[ra["muc_uu_tien"] == "Chưa cần"]["nguy_co_phan_tram"].iloc[0]
    print("  dòng cuối trong danh sách %.1f%%, dòng đầu ngoài danh sách %.1f%%,"
          " cách nhau %.1f điểm phần trăm" % (cuoi, ngoai, cuoi - ngoai))


def main() -> None:
    bang, p, nguong = nap_goi()
    print("-- 9.3.1 ba cách mã hóa cùng một file kết quả --")
    ba_cach_ma_hoa(bang, p)
    print("\n-- 9.3.1 ba mức ưu tiên theo hai điểm cắt --")
    ba_muc_uu_tien(bang, p, nguong)
    print("\n-- 9.3.2 bản kết quả cho phòng đào tạo --")
    ban_doc_duoc(bang, p, nguong)


if __name__ == "__main__":
    main()
