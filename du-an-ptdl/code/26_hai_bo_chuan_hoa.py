"""So hai bộ chuẩn hóa MinMaxScaler và StandardScaler trên cùng các cột (mục 4.1.2)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]


def tham_so_da_hoc(d: pd.DataFrame) -> None:
    """In các tham số mỗi bộ chuẩn hóa học được từ dữ liệu."""
    mm = MinMaxScaler().fit(d[COT_SO])
    st = StandardScaler().fit(d[COT_SO])
    bang = pd.DataFrame({"mm_min_": mm.data_min_, "mm_max_": mm.data_max_,
                         "st_mean_": st.mean_, "st_scale_": st.scale_},
                        index=COT_SO)
    print(bang.round(3).to_string())


def hinh_dang_sau_bien_doi(d: pd.DataFrame) -> None:
    """Đo sáu thống kê của một cột trước và sau mỗi phép chuẩn hóa."""
    mm_ra = MinMaxScaler().set_output(transform="pandas").fit_transform(d[COT_SO])
    st_ra = StandardScaler().set_output(transform="pandas").fit_transform(d[COT_SO])
    do = {}
    for ten, bang in (("bản gốc", d[COT_SO]), ("khoảng [0, 1]", mm_ra),
                      ("chuẩn hóa z", st_ra)):
        s = bang["nop_tre_tb_gio"]
        do[ten] = {"nhỏ nhất": round(s.min(), 4), "lớn nhất": round(s.max(), 4),
                   "trung bình": round(s.mean(), 4), "độ lệch": round(s.std(), 4),
                   "phân vị 75%": round(s.quantile(0.75), 4),
                   "độ lệch bất đối xứng": round(s.skew(), 3)}
    bang_do = pd.DataFrame(do).T
    print(bang_do[["nhỏ nhất", "lớn nhất", "trung bình"]].to_string())
    print(bang_do[["độ lệch", "phân vị 75%", "độ lệch bất đối xứng"]].to_string())
    print("Hai cột sau khi chuẩn hóa z, trung bình và độ lệch chuẩn:")
    print(st_ra[["tong_thoi_luong_phut", "ty_le_hoan_thanh_video"]]
          .agg(["mean", "std"]).round(6).to_string())


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("-- Tham số hai bộ chuẩn hóa học được trên 3.012 dòng --")
    tham_so_da_hoc(d)
    print("\n-- Cột nop_tre_tb_gio trước và sau mỗi phép chuẩn hóa --")
    hinh_dang_sau_bien_doi(d)


if __name__ == "__main__":
    main()
