"""Đo bốn cách xử lý một nhóm ngoại lai trên cột nop_tre_tb_gio, và vẽ biểu đồ hộp trước và sau."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
plt.style.use("grayscale")
GOC_DU_AN = Path(__file__).resolve().parent.parent
KET_QUA = GOC_DU_AN / "output"
KET_QUA.mkdir(exist_ok=True)
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")

s, y = df["nop_tre_tb_gio"], df["diem_cuoi_ky"]
q1, q3 = s.quantile(0.25), s.quantile(0.75)
tren, p99 = q3 + 1.5 * (q3 - q1), s.quantile(0.99)
nghi = s > 100                                     # tám dòng đã khoanh ở mục 3.2.1

thanh_thieu = s.mask(nghi)
cach = {"giữ nguyên": (s, y),
        f"chặn biên tại ngưỡng IQR {tren:.2f}": (s.clip(upper=tren), y),
        f"chặn biên tại phân vị 99% {p99:.2f}": (s.clip(upper=p99), y),
        "thành giá trị thiếu rồi điền trung vị": (thanh_thieu.fillna(thanh_thieu.median()), y),
        "loại bỏ 8 dòng": (s[~nghi], y[~nghi])}
dong = [{"cach": ten, "n": len(x), "trung_binh": round(x.mean(), 3),
         "do_lech": round(x.std(), 3), "lon_nhat": round(x.max(), 2),
         "tuong_quan": round(x.corr(yy), 4)} for ten, (x, yy) in cach.items()]
print(pd.DataFrame(dong).to_string(index=False))

truc = plt.subplots(figsize=(7, 4))[1]
truc.boxplot([s, s.clip(upper=tren)], tick_labels=["trước khi xử lý", "sau khi chặn biên"])
# ... hai dòng đặt tiêu đề và nhãn trục tung, rồi lưu ra output/ch03-box-nop-tre.png
truc.set_ylabel("Số giờ nộp muộn trung bình")
truc.set_title("Nộp muộn trung bình trước và sau khi chặn biên")
plt.tight_layout()
plt.savefig(KET_QUA / "ch03-box-nop-tre.png", dpi=150)
plt.close()
print("\nĐã ghi output/ch03-box-nop-tre.png")
