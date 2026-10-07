"""Khung chuẩn để vẽ và lưu biểu đồ, rồi vẽ bốn hình của chương 2."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")              # vẽ ra file, không cần cửa sổ hiển thị
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
plt.style.use("grayscale")         # mọi hình của sách in đen trắng
GOC_DU_AN = Path(__file__).resolve().parent.parent
KET_QUA = GOC_DU_AN / "output"
KET_QUA.mkdir(exist_ok=True)
df = pd.read_csv(GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv", encoding="utf-8")


def luu(ten: str) -> None:
    """Ba việc làm sau mỗi hình, gom lại một chỗ để không lặp."""
    plt.tight_layout()
    plt.savefig(KET_QUA / ten, dpi=150)
    plt.close()


# Hình 2.3: histogram điểm cuối kỳ
truc = df["diem_cuoi_ky"].plot.hist(bins=20, edgecolor="black")
truc.set_title("Phân bố điểm cuối kỳ của 3.050 sinh viên")
truc.set_xlabel("Điểm cuối kỳ")
truc.set_ylabel("Số sinh viên")
luu("ch02-hist-diem-cuoi-ky.png")

# Chiều cao từng cột của Hình 2.3, để đọc hình bằng số thay vì bằng mắt
cao, bien = np.histogram(df["diem_cuoi_ky"], bins=20)
for i in (0, 12, 13, 18, 19):
    print(f"Cột [{bien[i]:4.1f}; {bien[i + 1]:4.1f}) cao {cao[i]:3d} sinh viên")
print("Số ô mang đúng 10,0:", (df["diem_cuoi_ky"] == 10.0).sum(),
      "| đúng 0,0:", (df["diem_cuoi_ky"] == 0.0).sum())

# Hình 2.4: biểu đồ hộp theo hình thức học
truc = df.boxplot(column="diem_cuoi_ky", by="hinh_thuc_hoc", grid=False)
plt.suptitle("")                                   # bỏ tiêu đề tự sinh
truc.set_title("Điểm cuối kỳ theo hình thức học")
truc.set_xlabel("Hình thức học")
truc.set_ylabel("Điểm cuối kỳ")
luu("ch02-box-hinh-thuc.png")

# Hai biểu đồ cột: tần số thiết bị, điểm trung bình theo hình thức học
truc = df["thiet_bi_chinh"].value_counts().plot.bar(rot=0, edgecolor="black")
truc.set_title("Số sinh viên theo thiết bị học chính")
truc.set_xlabel("Thiết bị học chính")
truc.set_ylabel("Số sinh viên")
luu("ch02-bar-thiet-bi.png")

truc = df.groupby("hinh_thuc_hoc")["diem_cuoi_ky"].mean().plot.bar(rot=0, edgecolor="black")
truc.set_ylim(0, 10)                               # giữ trục tung bắt đầu từ 0
truc.set_title("Điểm cuối kỳ trung bình theo hình thức học")
truc.set_ylabel("Điểm cuối kỳ trung bình")
luu("ch02-bar-hinh-thuc.png")

# Hình 2.5: biểu đồ phân tán hai cột điểm
truc = df.plot.scatter(x="diem_giua_ky", y="diem_cuoi_ky", s=8, alpha=0.3)
truc.set_title("Quan hệ giữa điểm giữa kỳ và điểm cuối kỳ")
truc.set_xlabel("Điểm giữa kỳ")
truc.set_ylabel("Điểm cuối kỳ")
luu("ch02-scatter-diem.png")
print("Đã ghi 5 file hình vào", KET_QUA.name)
