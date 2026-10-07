"""Sinh bộ dữ liệu mô phỏng dùng xuyên suốt sách.

Bộ dữ liệu mô tả kết quả học tập một học phần trực tuyến của 3.000 sinh viên tại một
trung tâm đào tạo trực tuyến giả định. Toàn bộ số liệu là MÔ PHỎNG, không phải dữ liệu
thật của bất kỳ người học nào.

Chạy:
    python sinh_du_lieu.py

Sinh ra trong ../du-an-ptdl/data/raw/:
    sinhvien-truc-tuyen.csv       bản thô, còn lỗi (dùng từ chương 2)
    sinhvien-truc-tuyen.xlsx      cùng nội dung, định dạng Excel (chương 2, mục đọc file Excel)
    tinh-thanh.csv                bảng tra vùng miền theo tỉnh thành

Cùng một giá trị SEED luôn cho ra đúng một bộ dữ liệu, nên mọi con số in trong sách
đều tái lập được.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Console Windows mặc định không dùng UTF-8, in tiếng Việt sẽ báo UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 20260101
N = 3000
THU_MUC_DU_LIEU = Path(__file__).resolve().parent.parent / "du-an-ptdl" / "data" / "raw"

rng = np.random.default_rng(SEED)

# --- Danh mục ---------------------------------------------------------------
NGANH = ["Công nghệ thông tin", "Kinh tế", "Quản trị kinh doanh", "Kế toán",
         "Luật", "Ngôn ngữ Anh", "Du lịch"]
TY_LE_NGANH = [0.34, 0.14, 0.14, 0.11, 0.10, 0.10, 0.07]

TINH_THANH = ["Hà Nội", "TP Hồ Chí Minh", "Hải Phòng", "Đà Nẵng", "Cần Thơ",
              "Nghệ An", "Thanh Hóa", "Bắc Ninh", "Hưng Yên", "Thái Nguyên",
              "Quảng Ninh", "Lâm Đồng", "Khánh Hòa", "Bình Dương", "Đồng Nai"]
TY_LE_TINH = np.array([28, 14, 5, 5, 3, 7, 7, 4, 4, 3, 4, 3, 3, 5, 5], dtype=float)
TY_LE_TINH /= TY_LE_TINH.sum()

VUNG = {"Hà Nội": "Đồng bằng sông Hồng", "Hải Phòng": "Đồng bằng sông Hồng",
        "Bắc Ninh": "Đồng bằng sông Hồng", "Hưng Yên": "Đồng bằng sông Hồng",
        "Thái Nguyên": "Trung du và miền núi phía Bắc",
        "Quảng Ninh": "Trung du và miền núi phía Bắc",
        "Nghệ An": "Bắc Trung Bộ", "Thanh Hóa": "Bắc Trung Bộ",
        "Đà Nẵng": "Duyên hải miền Trung", "Khánh Hòa": "Duyên hải miền Trung",
        "Lâm Đồng": "Tây Nguyên",
        "TP Hồ Chí Minh": "Đông Nam Bộ", "Bình Dương": "Đông Nam Bộ",
        "Đồng Nai": "Đông Nam Bộ", "Cần Thơ": "Đồng bằng sông Cửu Long"}

THIET_BI = ["Máy tính", "Điện thoại", "Máy tính bảng"]
HINH_THUC = ["Trực tuyến", "Kết hợp"]


def sinh_du_lieu_sach() -> pd.DataFrame:
    """Sinh bảng dữ liệu đúng, chưa gài lỗi."""
    ma_sv = [f"SV{26:02d}{i:05d}" for i in range(1, N + 1)]
    nganh = rng.choice(NGANH, size=N, p=TY_LE_NGANH)
    gioi_tinh = rng.choice(["Nam", "Nữ", "Khác"], size=N, p=[0.52, 0.46, 0.02])
    nhom_tuoi = rng.choice(["18-22", "23-30", "31-40", "Trên 40"],
                           size=N, p=[0.46, 0.31, 0.17, 0.06])

    # Sinh viên lớn tuổi phần lớn vừa làm vừa học. Quan hệ này có thật trong hệ đào tạo
    # mở và là gốc của phần bàn về thiên lệch dữ liệu ở chương 11.
    p_vua_lam = pd.Series(nhom_tuoi).map(
        {"18-22": 0.22, "23-30": 0.68, "31-40": 0.88, "Trên 40": 0.93}).to_numpy()
    vua_lam_vua_hoc = np.where(rng.random(N) < p_vua_lam, "Có", "Không")

    hinh_thuc = rng.choice(HINH_THUC, size=N, p=[0.72, 0.28])
    tinh_thanh = rng.choice(TINH_THANH, size=N, p=TY_LE_TINH)
    thiet_bi = rng.choice(THIET_BI, size=N, p=[0.55, 0.39, 0.06])

    # Năng lực tiềm ẩn của từng sinh viên: không có trong dữ liệu, chỉ dùng để sinh
    # các biến quan sát được một cách nhất quán với nhau.
    nang_luc = rng.normal(0, 1, N)
    # Người vừa làm vừa học có ít thời gian hơn, không phải kém hơn.
    quy_thoi_gian = np.where(vua_lam_vua_hoc == "Có",
                             rng.normal(-0.55, 0.8, N), rng.normal(0.25, 0.8, N))

    so_lan_dang_nhap = np.clip(
        rng.normal(38 + 9 * nang_luc + 7 * quy_thoi_gian, 11), 1, 160).round().astype(int)
    tong_thoi_luong_phut = np.clip(
        rng.normal(900 + 220 * nang_luc + 190 * quy_thoi_gian, 260), 20, 4000).round().astype(int)
    ty_le_hoan_thanh_video = np.clip(
        rng.normal(0.62 + 0.14 * nang_luc + 0.10 * quy_thoi_gian, 0.16), 0.0, 1.0).round(3)
    so_bai_tap_nop = np.clip(
        rng.normal(7.2 + 1.5 * nang_luc + 1.1 * quy_thoi_gian, 1.9), 0, 10).round().astype(int)
    diem_tb_bai_tap = np.where(
        so_bai_tap_nop == 0, 0.0,
        np.clip(rng.normal(6.6 + 1.25 * nang_luc, 1.25), 0, 10).round(1))
    so_lan_dien_dan = rng.poisson(np.clip(3.2 + 1.5 * nang_luc + 0.8 * quy_thoi_gian, 0.2, None))
    nop_tre_tb_gio = np.clip(
        rng.gamma(shape=1.6, scale=np.clip(9 - 2.3 * nang_luc - 1.6 * quy_thoi_gian, 1.2, None)),
        0, None).round(1)
    diem_giua_ky = np.clip(rng.normal(6.0 + 1.45 * nang_luc + 0.25 * quy_thoi_gian, 1.2),
                           0, 10).round(1)

    ngay_dang_ky = pd.to_datetime("2026-01-20") + pd.to_timedelta(
        rng.integers(0, 28, N), unit="D")

    # Điểm cuối kỳ phụ thuộc hành vi học tập quan sát được, cộng nhiễu.
    # Không phụ thuộc giới tính, ngành hay tỉnh thành.
    #
    # Quan hệ CỐ Ý không hoàn toàn tuyến tính, vì ba lẽ sư phạm:
    #  (a) thời lượng học có hiệu suất giảm dần (dùng căn bậc hai), nên mô hình cây
    #      và rừng ngẫu nhiên có chỗ tỏ ra hơn hồi quy tuyến tính (chương 5, 7);
    #  (b) nộp thiếu bài tập tạo một bậc nhảy (ngưỡng 5 bài), là dạng quan hệ mà
    #      mô hình tuyến tính không mô tả được;
    #  (c) tỉ lệ xem video có tác dụng khác nhau giữa lớp trực tuyến và lớp kết hợp,
    #      và nộp muộn gây hậu quả nhẹ hơn với sinh viên vừa làm vừa học. Hai tương tác
    #      này là nguồn của phần chênh lệch hiệu năng theo nhóm ở chương 11.
    he_so_video = np.where(hinh_thuc == "Trực tuyến", 1.65, 0.45)
    he_so_tre = np.where(vua_lam_vua_hoc == "Có", 0.35, 1.25)
    diem_cuoi_ky = (
        0.40 * diem_giua_ky
        + 0.18 * diem_tb_bai_tap
        + 1.70 * ty_le_hoan_thanh_video * he_so_video
        + 0.030 * np.sqrt(tong_thoi_luong_phut)
        + 0.010 * so_lan_dang_nhap
        - 0.050 * np.minimum(nop_tre_tb_gio, 60) * he_so_tre
        - 0.85 * (so_bai_tap_nop < 5)
        - 0.70 * (ty_le_hoan_thanh_video < 0.40)
        + 0.75
        + rng.normal(0, 0.95, N)
    )
    diem_cuoi_ky = np.clip(diem_cuoi_ky, 0, 10).round(1)

    df = pd.DataFrame({
        "ma_sv": ma_sv,
        "nganh": nganh,
        "gioi_tinh": gioi_tinh,
        "nhom_tuoi": nhom_tuoi,
        "vua_lam_vua_hoc": vua_lam_vua_hoc,
        "hinh_thuc_hoc": hinh_thuc,
        "tinh_thanh": tinh_thanh,
        "thiet_bi_chinh": thiet_bi,
        "ngay_dang_ky": ngay_dang_ky.strftime("%Y-%m-%d"),
        "so_lan_dang_nhap": so_lan_dang_nhap,
        "tong_thoi_luong_phut": tong_thoi_luong_phut,
        "ty_le_hoan_thanh_video": ty_le_hoan_thanh_video,
        "so_bai_tap_nop": so_bai_tap_nop,
        "diem_tb_bai_tap": diem_tb_bai_tap,
        "so_lan_dien_dan": so_lan_dien_dan,
        "nop_tre_tb_gio": nop_tre_tb_gio,
        "diem_giua_ky": diem_giua_ky,
        "diem_cuoi_ky": diem_cuoi_ky,
    })
    df["ket_qua"] = np.where(df["diem_cuoi_ky"] >= 4.0, "Đạt", "Không đạt")
    return df


def gai_loi(df: pd.DataFrame) -> pd.DataFrame:
    """Gài đúng những lỗi mà chương 2 phát hiện và chương 3 xử lý.

    Mỗi loại lỗi dưới đây tương ứng một mục của chương 3, nên không được bỏ loại nào.
    """
    df = df.copy()

    # 1. Giá trị thiếu (chương 3, mục xử lý dữ liệu thiếu)
    for cot, ty_le in (("diem_giua_ky", 0.062), ("ty_le_hoan_thanh_video", 0.041),
                       ("tinh_thanh", 0.019), ("thiet_bi_chinh", 0.012)):
        chi_so = rng.choice(df.index, size=int(len(df) * ty_le), replace=False)
        df.loc[chi_so, cot] = np.nan

    # 2. Giá trị ngoại lai và giá trị canh (sentinel) vô nghĩa
    chi_so = rng.choice(df.index, size=14, replace=False)
    df.loc[chi_so, "tong_thoi_luong_phut"] = 99999
    chi_so = rng.choice(df.index, size=9, replace=False)
    df.loc[chi_so, "so_lan_dang_nhap"] = -1
    chi_so = rng.choice(df.index, size=7, replace=False)
    df.loc[chi_so, "nop_tre_tb_gio"] = df.loc[chi_so, "nop_tre_tb_gio"] * 40

    # 3. Định dạng không nhất quán
    #    3a. Nhãn phân loại viết lệch hoa thường và thừa khoảng trắng
    chi_so = rng.choice(df.index, size=120, replace=False)
    df.loc[chi_so, "gioi_tinh"] = df.loc[chi_so, "gioi_tinh"].str.lower()
    chi_so = rng.choice(df.index, size=90, replace=False)
    df.loc[chi_so, "nganh"] = " " + df.loc[chi_so, "nganh"].astype(str) + " "
    chi_so = rng.choice(df.index, size=70, replace=False)
    df.loc[chi_so, "vua_lam_vua_hoc"] = df.loc[chi_so, "vua_lam_vua_hoc"].replace(
        {"Có": "CO", "Không": "KHONG"})

    #    3b. Ngày tháng ba kiểu khác nhau
    ngay = pd.to_datetime(df["ngay_dang_ky"])
    kieu = rng.choice([0, 1, 2], size=len(df), p=[0.6, 0.25, 0.15])
    df["ngay_dang_ky"] = np.where(
        kieu == 0, ngay.dt.strftime("%Y-%m-%d"),
        np.where(kieu == 1, ngay.dt.strftime("%d/%m/%Y"), ngay.dt.strftime("%d-%m-%Y")))

    #    3c. Số thập phân dùng dấu phẩy, nên cột bị đọc thành chuỗi
    diem = df["diem_tb_bai_tap"].astype("object")
    chi_so = rng.choice(df.index, size=150, replace=False)
    diem.loc[chi_so] = diem.loc[chi_so].map(
        lambda v: str(v).replace(".", ",") if pd.notna(v) else v)
    df["diem_tb_bai_tap"] = diem

    # 4. Dòng trùng lặp: trùng hoàn toàn, và trùng mã sinh viên nhưng khác vài giá trị
    trung_hoan_toan = df.sample(n=38, random_state=7)
    trung_ma = df.sample(n=12, random_state=11).copy()
    trung_ma["so_lan_dien_dan"] = trung_ma["so_lan_dien_dan"] + 1
    df = pd.concat([df, trung_hoan_toan, trung_ma], ignore_index=True)

    # Trộn thứ tự để các dòng trùng không nằm cạnh nhau
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main() -> None:
    THU_MUC_DU_LIEU.mkdir(parents=True, exist_ok=True)

    sach = sinh_du_lieu_sach()
    tho = gai_loi(sach)

    duong_dan_csv = THU_MUC_DU_LIEU / "sinhvien-truc-tuyen.csv"
    tho.to_csv(duong_dan_csv, index=False, encoding="utf-8")
    try:
        tho.to_excel(THU_MUC_DU_LIEU / "sinhvien-truc-tuyen.xlsx", index=False)
    except Exception as loi:  # openpyxl chưa cài
        print(f"  (bỏ qua bản Excel: {loi})")

    pd.DataFrame({"tinh_thanh": list(VUNG), "vung": list(VUNG.values())}).to_csv(
        THU_MUC_DU_LIEU / "tinh-thanh.csv", index=False, encoding="utf-8")

    print(f"Đã ghi {duong_dan_csv} – {len(tho)} dòng, {tho.shape[1]} cột")
    print(f"  dòng trùng hoàn toàn: {tho.duplicated().sum()}")
    print(f"  ô thiếu: {int(tho.isna().sum().sum())}")
    print(f"  tỉ lệ 'Không đạt': {(tho['ket_qua'] == 'Không đạt').mean():.1%}")


if __name__ == "__main__":
    main()
