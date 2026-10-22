"""An toàn khi nạp file mô hình: pickle chạy mã, bộ nạp có hạn chế, mã băm kiểm tra (mục 8.1.3)."""
import hashlib
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib.numpy_pickle import NumpyUnpickler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
TAM = THU_MUC_MO_HINH / "tam"
GOI_CHO_PHEP = ("sklearn.", "numpy", "scipy.", "pandas.", "joblib.")
BUILTINS_CHO_PHEP = {"slice"}
# Chỉ một tên, và là kết quả đo: bỏ lần lượt từng tên của danh sách tám tên quen dùng
# ("slice", "range", "frozenset", "set", "dict", "list", "tuple", "object") rồi nạp lại
# cả chín file .joblib trong models/ thì chỉ thiếu "slice" mới làm phép nạp hỏng.


class MoHinhLa:
    """File mô hình giả: hàm __reduce__ nói cho pickle biết phải GỌI GÌ khi nạp."""

    def __reduce__(self):
        return (print, ("    [mã lạ đã chạy] đọc được biến môi trường, ghi được file,"
                        " gọi được mạng",))


class NapCoHanChe(NumpyUnpickler):
    """Chặn mọi lớp ngoài danh sách cho phép; với builtins chỉ cho tên dựng dữ liệu."""

    def find_class(self, module, name):
        if module == "builtins" and name not in BUILTINS_CHO_PHEP:
            raise pickle.UnpicklingError("chặn builtins.%s" % name)
        if module != "builtins" and not module.startswith(GOI_CHO_PHEP):
            raise pickle.UnpicklingError("chặn %s.%s" % (module, name))
        return super().find_class(module, name)


def nap_co_han_che(duong_dan):
    with open(duong_dan, "rb") as f:
        return NapCoHanChe(str(duong_dan), f, ensure_native_byte_order=True).load()


def ma_bam(duong_dan):
    """SHA-256 của nguyên văn file, tính theo từng khối để không nạp cả file vào bộ nhớ."""
    bam = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        for khoi in iter(lambda: f.read(65536), b""):
            bam.update(khoi)
    return bam.hexdigest()


def pickle_chay_ma():
    """Nạp một file 112 byte không chứa mô hình nào, nhưng vẫn chạy được mã."""
    TAM.mkdir(parents=True, exist_ok=True)
    p = TAM / "mo-hinh-la.joblib"
    joblib.dump(MoHinhLa(), p)
    print("  file %s, %d byte. Lệnh joblib.load:" % (p.name, p.stat().st_size))
    ket_qua = joblib.load(p)
    print("  giá trị nhận được: %s (không có mô hình nào trong file)" % ket_qua)
    return p


def bo_nap_co_han_che(p_la):
    """Cùng một lệnh nạp, hai file: một file thật và một file lạ."""
    for p in (THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib", p_la):
        try:
            doi_tuong = nap_co_han_che(p)
            print("%-30s nạp được: %s" % (p.name, type(doi_tuong).__name__))
        except pickle.UnpicklingError as loi:
            print("%-30s TỪ CHỐI: %s" % (p.name, loi))


def kiem_bo_nap_khong_lam_lech_ket_qua():
    """Bộ nạp có hạn chế phải cho đúng mô hình cũ, không chỉ cho đúng kiểu đối tượng."""
    p = THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib"
    a, b = nap_co_han_che(p), joblib.load(p)
    cot = list(b.estimator_.named_steps["tien"].feature_names_in_)
    X = pd.read_parquet(VAO / "ch04-X-test.parquet")[cot]
    print("  ngưỡng khớp: %s | xác suất lệch lớn nhất %.2e | nhãn khớp từng dòng: %s"
          % (float(a.best_threshold_) == float(b.best_threshold_),
             float(np.abs(a.predict_proba(X)[:, 1] - b.predict_proba(X)[:, 1]).max()),
             bool((a.predict(X) == b.predict(X)).all())))


def ghi_va_doi_chieu_ma_bam(p_la):
    """Mã băm ghi cạnh file lúc tạo ra nó; mọi lần nạp sau đều đối chiếu lại."""
    p = THU_MUC_MO_HINH / "ch08-mo-hinh-kem-nguong.joblib"
    p_bam = p.with_suffix(".joblib.sha256")
    p_bam.write_text(ma_bam(p) + "  " + p.name + "\n", encoding="utf-8")
    cho_doi = p_bam.read_text(encoding="utf-8").split()[0]
    print("  đã ghi %s" % p_bam.name)
    print("  %-32s %s | khớp: %s" % (p.name, ma_bam(p)[:32], ma_bam(p) == cho_doi))
    print("  %-32s %s | khớp: %s" % (p_la.name, ma_bam(p_la)[:32], ma_bam(p_la) == cho_doi))


def main() -> None:
    print("-- 8.1.3 lệnh nạp một file pickle thực thi mã chứa trong file --")
    p_la = pickle_chay_ma()

    print("\n-- 8.1.3 bộ nạp có hạn chế: chặn lớp ngoài danh sách cho phép --")
    bo_nap_co_han_che(p_la)

    print("\n-- 8.1.3 bộ nạp có hạn chế cho đúng mô hình cũ --")
    kiem_bo_nap_khong_lam_lech_ket_qua()

    print("\n-- 8.1.3 mã băm SHA-256 đi kèm file mô hình --")
    ghi_va_doi_chieu_ma_bam(p_la)
    p_la.unlink()
    TAM.rmdir()


if __name__ == "__main__":
    main()
