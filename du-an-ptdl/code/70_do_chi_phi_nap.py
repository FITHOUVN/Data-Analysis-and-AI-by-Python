"""Đo chi phí nạp mô hình và chi phí khởi động tiến trình (mục 9.1.3).

Mọi con số dưới đây phụ thuộc máy: ổ đĩa, phần mềm quét tệp và số lõi đều đổi kết
quả, và trên máy đo thử phần nhập thư viện lệch tới ba lần giữa lần chạy nguội và
lần chạy nóng. Lấy giá trị NHỎ NHẤT của nhiều lần đo, và đo lại trên máy của mình
trước khi dẫn lại con số.
"""
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
THU_MUC_MA = Path(__file__).resolve().parent
FILE_MO_HINH = GOC_DU_AN / "models" / "ch08-mo-hinh-trien-khai.joblib"
LO = GOC_DU_AN / "data" / "raw" / "ch09-lo-moi.csv"
SO_LAN, SO_DONG_GOI_LE = 5, 10

PHAN_RA = """
import time, sys
t = [time.perf_counter()]
import pandas as pd, joblib
from sklearn.model_selection import FixedThresholdClassifier
t.append(time.perf_counter())
m = joblib.load(sys.argv[1]); t.append(time.perf_counter())
b = pd.read_csv(sys.argv[2]); t.append(time.perf_counter())
cot = list(m.estimator.estimator.feature_names_in_)
p = m.predict_proba(b[cot]); t.append(time.perf_counter())
ten = ["nhap thu vien", "joblib.load", "read_csv", "predict_proba"]
for i, n in enumerate(ten):
    print("  %-16s %8.1f ms" % (n, (t[i + 1] - t[i]) * 1000))
print("  %-16s %8.1f ms" % ("tong", (t[-1] - t[0]) * 1000))
"""


def nho_nhat(ham, so_lan=SO_LAN):
    return min(_do_mot_lan(ham) for _ in range(so_lan))


def _do_mot_lan(ham):
    t0 = time.perf_counter()
    ham()
    return (time.perf_counter() - t0) * 1000


def khoi_dong_tien_trinh():
    """Bốn mức: Python trống, thêm pandas, thêm joblib, thêm sklearn."""
    muc = [("Python trống", "pass"),
           ("thêm pandas", "import pandas"),
           ("thêm joblib", "import pandas, joblib"),
           ("thêm sklearn", "import pandas, joblib;"
                            " from sklearn.model_selection import FixedThresholdClassifier")]
    for ten, ma in muc:
        t = nho_nhat(lambda: subprocess.run([sys.executable, "-c", ma], check=True))
        print("  %-16s %8.0f ms" % (ten, t))


def phan_ra_mot_lan_chay():
    """Bốn phần của một lần chạy, đo trong một tiến trình mới."""
    subprocess.run([sys.executable, "-c", PHAN_RA, str(FILE_MO_HINH), str(LO)], check=True)


def nap_va_du_doan():
    bang = pd.read_csv(LO)
    mo_hinh = joblib.load(FILE_MO_HINH)
    cot = list(mo_hinh.estimator.estimator.feature_names_in_)
    X = bang[cot]
    t_nap_lai = nho_nhat(lambda: joblib.load(FILE_MO_HINH))
    t_lo = nho_nhat(lambda: mo_hinh.predict_proba(X))
    t_le = nho_nhat(lambda: [mo_hinh.predict_proba(X.iloc[[i]]) for i in range(len(X))])
    print("  joblib.load lần thứ hai trở đi      %8.1f ms" % t_nap_lai)
    print("  predict_proba cả lô %d dòng         %8.1f ms" % (len(X), t_lo))
    print("  predict_proba gọi lẻ %d lần         %8.1f ms  (%.0f lần chậm hơn)"
          % (len(X), t_le, t_le / t_lo))


def chay_chuong_trinh():
    """Gọi chương trình một lần cho cả lô, rồi một lần cho mỗi dòng."""
    bang = pd.read_csv(LO).iloc[:SO_DONG_GOI_LE]
    tam_lo = GOC_DU_AN / "data" / "processed" / "ch09-lo-ngan.csv"
    tam_mot = GOC_DU_AN / "data" / "processed" / "ch09-mot-dong.csv"
    bang.to_csv(tam_lo, index=False, encoding="utf-8")
    goc = [sys.executable, str(THU_MUC_MA / "69_du_doan.py")]

    def goi(p):
        subprocess.run(goc + ["--vao", str(p), "--log-muc", "ERROR"],
                       check=True, stdout=subprocess.DEVNULL)

    t_mot = nho_nhat(lambda: goi(tam_lo), 3)
    t0 = time.perf_counter()
    for i in range(len(bang)):
        bang.iloc[[i]].to_csv(tam_mot, index=False, encoding="utf-8")
        goi(tam_mot)
    t_moi_dong = (time.perf_counter() - t0) * 1000
    print("  một lần gọi cho %d dòng             %8.0f ms" % (len(bang), t_mot))
    print("  %d lần gọi, mỗi lần một dòng        %8.0f ms  (%.1f lần chậm hơn)"
          % (len(bang), t_moi_dong, t_moi_dong / t_mot))
    for p in (tam_lo, tam_mot):
        p.unlink(missing_ok=True)


def main() -> None:
    print("-- 9.1.3 khởi động tiến trình: phần không bỏ được (nhỏ nhất của %d lần) --" % SO_LAN)
    khoi_dong_tien_trinh()
    print("\n-- 9.1.3 bốn phần của một lần chạy, trong một tiến trình mới --")
    phan_ra_mot_lan_chay()
    print("\n-- 9.1.3 trong cùng một tiến trình (nhỏ nhất của %d lần) --" % SO_LAN)
    nap_va_du_doan()
    print("\n-- 9.1.3 gọi chương trình một lần cho cả lô so với mỗi dòng một lần --")
    chay_chuong_trinh()


if __name__ == "__main__":
    main()
