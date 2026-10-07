"""Lưu tập đã chia, bộ tiền xử lý đã khớp và bản ghi các bước đã làm (mục 4.4.1)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"
RA = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
THU_MUC_MO_HINH.mkdir(exist_ok=True)

MUC_TIEU = "diem_cuoi_ky"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
DAC_TRUNG_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
                "hinh_thuc_hoc", "thiet_bi_chinh"]
COT_BO = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "ket_qua"]
BUOC_DA_LAM = {
    "nguon": "data/processed/du-lieu-sach.csv (3012 dòng, 19 cột, 0 ô thiếu)",
    "dac_trung_so": COT_SO,
    "dac_trung_phan_loai": DAC_TRUNG_PL,
    "cot_bo_khoi_dac_trung": {
        "ma_sv": "định danh, 3000 giá trị khác nhau, 595/601 mã của tập kiểm tra là mã lạ",
        "ngay_dang_ky": "chỉ 2 tháng khác nhau, không mang thông tin phân biệt",
        "tinh_thanh": "16 mức, thêm 16 cột mà R2 giảm 0,0019",
        "ket_qua": "suy ra trực tiếp từ diem_cuoi_ky bằng ngưỡng 4,0; là mục tiêu của Chương 6",
    },
    "muc_tieu_hoi_quy": MUC_TIEU,
    "muc_tieu_phan_loai": "ket_qua",
    "chia_tap": "train_test_split(test_size=0.2, random_state=42)",
    "chuan_hoa": "StandardScaler cho 8 cột số, khớp chỉ trên tập huấn luyện",
    "ma_hoa": "OneHotEncoder(handle_unknown='ignore', sparse_output=False) cho 6 cột phân loại",
    "dong_trung_vuot_bien": "6 dòng tập kiểm tra có bản sinh đôi trong tập huấn luyện",
}


def chuan_bi(d: pd.DataFrame):
    """Chia tập rồi khớp bộ tiền xử lý chỉ trên tập huấn luyện."""
    X, y = d[COT_SO + DAC_TRUNG_PL], d[MUC_TIEU]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    tien = ColumnTransformer([("so", StandardScaler(), COT_SO),
                              ("pl", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), DAC_TRUNG_PL)])
    tien.set_output(transform="pandas")    # giữ tên cột; chỉ chạy khi không dùng mảng thưa
    tien.fit(X_tr)
    return X_tr, X_te, y_tr, y_te, tien


def ghi_ra_file(X_tr, X_te, y_tr, y_te, tien) -> None:
    """Ghi bốn phần dữ liệu đã chia, bộ tiền xử lý và bản ghi các bước."""
    for ten, bang in (("X-train", X_tr), ("X-test", X_te),
                      ("y-train", y_tr.to_frame()), ("y-test", y_te.to_frame())):
        bang.to_parquet(RA / f"ch04-{ten}.parquet", index=False)
    joblib.dump(tien, THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    (RA / "ch04-cac-buoc-da-lam.json").write_text(
        json.dumps(BUOC_DA_LAM, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã ghi 4 file tập đã chia, bộ tiền xử lý và bản ghi các bước.")


def so_csv_voi_parquet(d: pd.DataFrame) -> None:
    """So dung lượng và khả năng giữ kiểu của hai định dạng, trên bảng 3.012 dòng × 19 cột."""
    csv, pq = RA / "ch04-bang-day-du.csv", RA / "ch04-bang-day-du.parquet"
    d.to_csv(csv, index=False, encoding="utf-8")
    d.to_parquet(pq, index=False)
    print("CSV %7.1f KB | Parquet %7.1f KB | Parquet nhỏ hơn %.1f lần"
          % (csv.stat().st_size / 1024, pq.stat().st_size / 1024,
             csv.stat().st_size / pq.stat().st_size))
    for ten, lai in (("CSV", pd.read_csv(csv)), ("Parquet", pd.read_parquet(pq))):
        lech = {c: str(lai[c].dtype) for c in d.columns if lai[c].dtype != d[c].dtype}
        print("%-8s số cột lệch kiểu sau khi đọc lại: %d %s" % (ten, len(lech), lech))


def kiem_nghiem_thu(tien) -> None:
    """Nạp lại bộ tiền xử lý từ file và kiểm nó cho đúng kết quả cũ."""
    # THU_MUC_MO_HINH là thư mục models đã dựng ở Mã 4.28
    nap = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X_te_lai = pd.read_parquet(RA / "ch04-X-test.parquet")
    a, b = nap.transform(X_te_lai), tien.transform(X_te_lai)
    print("Bộ nạp lại cho kết quả giống bộ trong bộ nhớ:", bool(a.equals(b)))
    print("Số đặc trưng sau biến đổi:", a.shape[1], "| ba cột cuối:", list(a.columns[-3:]))
    print("Trung bình tám cột số của tập kiểm tra sau chuẩn hóa:",
          [round(float(v), 4) for v in a.iloc[:, :8].mean()])


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    X_tr, X_te, y_tr, y_te, tien = chuan_bi(d)
    print("Bỏ khỏi đặc trưng:", COT_BO)
    print("Đặc trưng:", len(COT_SO), "cột số +", len(DAC_TRUNG_PL), "cột phân loại")
    print("Tập huấn luyện", X_tr.shape, "| tập kiểm tra", X_te.shape)
    ghi_ra_file(X_tr, X_te, y_tr, y_te, tien)
    print("\n-- CSV so với Parquet trên bảng 3.012 dòng × 19 cột --")
    so_csv_voi_parquet(d)
    print("\n-- Kiểm nghiệm thu: nạp lại và chạy lại --")
    kiem_nghiem_thu(tien)


if __name__ == "__main__":
    main()
