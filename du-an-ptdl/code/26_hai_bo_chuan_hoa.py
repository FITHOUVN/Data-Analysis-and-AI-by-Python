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
    bang = pd.DataFrame({"min (MinMax)": mm.data_min_, "max (MinMax)": mm.data_max_,
                         "mean_ (Standard)": st.mean_, "scale_ (Standard)": st.scale_},
                        index=COT_SO)
    print(bang.round(3).to_string())


def hinh_dang_sau_bien_doi(d: pd.DataFrame) -> None:
    """Đo sáu thống kê của một cột trước và sau mỗi phép chuẩn hóa."""
    mm_ra = MinMaxScaler().set_output(transform="pandas").fit_transform(d[COT_SO])
    st_ra = StandardScaler().set_output(transform="pandas").fit_transform(d[COT_SO])
    for ten, bang in (("bản gốc", d[COT_SO]), ("khoảng [0, 1]", mm_ra),
                      ("chuẩn hóa z", st_ra)):
        s = bang["nop_tre_tb_gio"]
        print(f"{ten:16s} nhỏ nhất {s.min():8.4f}  lớn nhất {s.max():8.4f}"
              f"  trung bình {s.mean():8.4f}  độ lệch {s.std():7.4f}"
              f"  phân vị 75% {s.quantile(0.75):8.4f}"
              f"  độ lệch bất đối xứng {s.skew():6.3f}")
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
