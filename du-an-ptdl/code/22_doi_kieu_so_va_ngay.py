"""Đưa cột bị đọc thành chuỗi về đúng kiểu số và kiểu ngày, rồi kiểm kiểu của toàn bảng."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

# --- cột số bị đọc thành chuỗi ---
goc = df["diem_tb_bai_tap"]
so_o_phay = int(goc.str.contains(",").sum())
print("Số ô dùng dấu phẩy thập phân:", so_o_phay)
try:
    goc.astype("float64")
except ValueError as loi:
    print("astype('float64')           :")
    print("   ", loi)
thang = pd.to_numeric(goc, errors="coerce")   # cách sai
dung = pd.to_numeric(
    goc.str.replace(",", ".", regex=False), errors="coerce")
print("Đổi thẳng                   : thiếu",
      int(thang.isna().sum()),
      "ô, trung bình", round(thang.mean(), 4))
print("Thay dấu rồi đổi            : thiếu",
      int(dung.isna().sum()),
      "ô, trung bình", round(dung.mean(), 4))
print("Miền sau khi đổi            :",
      dung.min(), "tới", dung.max())

# --- cột ngày có ba định dạng: đọc lần lượt, mỗi lần một định dạng tường minh ---
DINH_DANG = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
ngay = pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")
# mỗi lượt một định dạng tường minh
for dinh_dang in DINH_DANG:
    chua_doc = ngay.isna()
    doc_duoc = pd.to_datetime(df.loc[chua_doc, "ngay_dang_ky"],
                              format=dinh_dang, errors="coerce")
    ngay[chua_doc] = doc_duoc
    print(f"sau {dinh_dang}: còn "
          f"{int(ngay.isna().sum()):4d} ô chưa đọc được")
print("Kiểu:", ngay.dtype, "| miền:",
      ngay.min().date(), "tới", ngay.max().date())
print("Số tháng khác nhau:", ngay.dt.month.nunique())

# Hai cách sai, cả hai đều không báo lỗi
# cách sai 1
mac_dinh = pd.to_datetime(df["ngay_dang_ky"], errors="coerce")
# cách sai 2
tron = pd.to_datetime(df["ngay_dang_ky"], format="mixed",
                      dayfirst=True, errors="coerce")
print("\nĐể pandas tự suy định dạng : NaT", int(mac_dinh.isna().sum()),
      "| số tháng khác nhau", mac_dinh.dt.month.nunique())
print("format='mixed', dayfirst   : NaT", int(tron.isna().sum()),
      "| số tháng khác nhau", tron.dt.month.nunique())
print("                           | lệch", int((tron != ngay).sum()),
      "dòng")
print("Ô đầu dạng ISO 2026-02-08 đọc thành:", tron.iloc[2].date(),
      "thay vì", ngay.iloc[2].date())

# --- kiểm kiểu của toàn bảng: khai trước rồi so ---
d = df.assign(diem_tb_bai_tap=dung, ngay_dang_ky=ngay)
KIEU_MONG_DOI = {"so_lan_dang_nhap": "int64",
                 "tong_thoi_luong_phut": "int64",
                 "so_bai_tap_nop": "int64",
                 "so_lan_dien_dan": "int64",
                 "ty_le_hoan_thanh_video": "float64",
                 "diem_tb_bai_tap": "float64",
                 "nop_tre_tb_gio": "float64",
                 "diem_giua_ky": "float64",
                 "diem_cuoi_ky": "float64",
                 "ngay_dang_ky": "datetime64[us]"}
lech = {c: str(d[c].dtype) for c, k in KIEU_MONG_DOI.items()
        if str(d[c].dtype) != k}
print("Cột lệch kiểu so với bảng khai trước:",
      lech if lech else "không có")
print("Số cột chuỗi còn lại:", int((d.dtypes == "str").sum()))
