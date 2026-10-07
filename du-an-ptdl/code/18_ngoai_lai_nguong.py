"""So hai cách đặt ngưỡng ngoại lai: khoảng tứ phân vị và ba độ lệch chuẩn."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "so_lan_dien_dan", "nop_tre_tb_gio",
          "diem_giua_ky", "diem_cuoi_ky"]


def hai_nguong(s: pd.Series) -> dict:
    """Trả về hai cặp ngưỡng và số dòng vượt ngưỡng theo từng cách."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    d_iqr, t_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mu, sd = s.mean(), s.std()
    d_sd, t_sd = mu - 3 * sd, mu + 3 * sd
    return {"tren_iqr": round(t_iqr, 2), "vuot_iqr": int(((s < d_iqr) | (s > t_iqr)).sum()),
            "tren_3sd": round(t_sd, 2), "vuot_3sd": int(((s < d_sd) | (s > t_sd)).sum()),
            "sd_tren_iqr": round(sd / iqr, 2)}


print(pd.DataFrame([{"cot": c, **hai_nguong(df[c])} for c in COT_SO]).to_string(index=False))

# Phần vượt ngưỡng còn lại sau khi trừ các dòng đã ghi riêng là giá trị canh và lỗi ghi nhận
da_ghi_rieng = ((df["tong_thoi_luong_phut"] == 99999) | (df["so_lan_dang_nhap"] < 0)
                | (df["nop_tre_tb_gio"] > 100))
mat_na_dong = pd.Series(False, index=df.index)
tong_o, so_cot = 0, 0
for cot in COT_SO:
    q1, q3 = df[cot].quantile(0.25), df[cot].quantile(0.75)
    iqr = q3 - q1
    con_lai = ((df[cot] < q1 - 1.5 * iqr) | (df[cot] > q3 + 1.5 * iqr)) & ~da_ghi_rieng
    tong_o += int(con_lai.sum())
    so_cot += int(con_lai.any())
    mat_na_dong |= con_lai                         # một dòng có thể vượt ngưỡng ở nhiều cột
print("\nVượt ngưỡng IQR sau khi trừ", int(da_ghi_rieng.sum()), "dòng đã ghi riêng:",
      tong_o, "ô trên", int(mat_na_dong.sum()), "dòng,", so_cot, "cột")

# Hiệu ứng che: chính 14 ô giá trị canh đã kéo ngưỡng 3 độ lệch chuẩn ra xa
s = df["tong_thoi_luong_phut"]
sach = s[s != 99999]
print("\ntong_thoi_luong_phut, trước khi bỏ 14 ô giá trị canh:")
print("  trung bình %.2f, độ lệch chuẩn %.2f, ngưỡng trên 3sd %.2f"
      % (s.mean(), s.std(), s.mean() + 3 * s.std()))
print("tong_thoi_luong_phut, sau khi bỏ 14 ô giá trị canh:")
print("  trung bình %.2f, độ lệch chuẩn %.2f, ngưỡng trên 3sd %.2f, vượt %d dòng"
      % (sach.mean(), sach.std(), sach.mean() + 3 * sach.std(),
         (sach > sach.mean() + 3 * sach.std()).sum()))
q1, q3 = sach.quantile(0.25), sach.quantile(0.75)
print("  ngưỡng trên 1,5 IQR %.2f, vượt %d dòng"
      % (q3 + 1.5 * (q3 - q1), (sach > q3 + 1.5 * (q3 - q1)).sum()))
