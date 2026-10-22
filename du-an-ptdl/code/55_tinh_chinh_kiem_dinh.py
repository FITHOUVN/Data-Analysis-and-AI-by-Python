"""Tinh chỉnh siêu tham số trên MỘT tập kiểm định tách riêng, và cái giá của cách đó (mục 7.2)."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
SAU = (4, 5, 6, 7, 8, 10, 12)
LA = (1, 5, 10, 20, 30, 50)


def nap_goi():
    """Nạp gói dữ liệu hồi quy của Chương 4 và bộ tiền xử lý đã khớp."""
    X_tr = pd.read_parquet(VAO / "ch04-X-train.parquet")
    X_te = pd.read_parquet(VAO / "ch04-X-test.parquet")
    y_tr = pd.read_parquet(VAO / "ch04-y-train.parquet").squeeze("columns")
    y_te = pd.read_parquet(VAO / "ch04-y-test.parquet").squeeze("columns")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    return X_tr, X_te, y_tr, y_te, tien


def in_theo_dong(nhan, o, moi_dong=3):
    """In một dãy chuỗi ngắn theo nhiều dòng cho vừa bề rộng trang in."""
    print("  %s:" % nhan)
    for i in range(0, len(o), moi_dong):
        print("    " + " | ".join(o[i:i + moi_dong]))


def ong_cay(tien, **tham_so):
    """Một pipeline gồm bộ tiền xử lý khớp lại và một cây hồi quy."""
    return Pipeline([("tien", clone(tien)),
                     ("cay", DecisionTreeRegressor(random_state=42, **tham_so))])


def khop_do(tien, A, a, B, b, **tham_so):
    """Khớp một bộ siêu tham số trên (A, a) rồi đo R² trên (B, b)."""
    mo = ong_cay(tien, **tham_so).fit(A, a)
    return r2_score(b, mo.predict(B)), mo


def duong_cong_kiem_dinh(tien, A, a, V, v):
    """Hai cột R² theo max_depth: đây là dữ liệu của đường cong kiểm định."""
    print("-- đường cong kiểm định theo max_depth (khớp trên %d, đo trên %d) --"
          % (len(a), len(v)))
    for sau in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None):
        r_v, mo = khop_do(tien, A, a, V, v, max_depth=sau)
        r_a = r2_score(a, mo.predict(A))
        print("  max_depth %-4s | R2 huấn luyện %.4f | R2 kiểm định %.4f"
              % (sau, r_a, r_v))
        print("    khoảng cách %.4f | số lá %d"
              % (r_a - r_v, mo.named_steps["cay"].get_n_leaves()))


def luoi_hai_chieu(tien, A, a, V, v):
    """Quét lưới max_depth x min_samples_leaf trên tập kiểm định."""
    t0 = time.perf_counter()
    diem = {}
    print("-- lưới hai chiều trên tập kiểm định (cột: min_samples_leaf = %s) --"
          % ", ".join(str(l) for l in LA))
    for s in SAU:
        for l in LA:
            diem[(s, l)], _ = khop_do(tien, A, a, V, v, max_depth=s,
                                      min_samples_leaf=l)
        print("  max_depth %-3d |" % s,
              " ".join("%.4f" % diem[(s, l)] for l in LA))
    print("  %d lần khớp trong %.1f giây | dải R2 kiểm định %.4f … %.4f"
          % (len(diem), time.perf_counter() - t0, min(diem.values()),
             max(diem.values())))
    print("    độ lệch chuẩn giữa các ô %.4f"
          % float(np.std(list(diem.values()), ddof=1)))
    xep = sorted(diem.items(), key=lambda kv: -kv[1])
    in_theo_dong("năm cấu hình đầu",
                 ["d%d-l%d %.4f" % (k[0], k[1], val) for k, val in xep[:5]])
    return diem


def doi_hat_kiem_dinh(tien, X_tr, y_tr, X_te, y_te):
    """Đổi hạt của phép chia kiểm định: lưới chọn cấu hình nào, và kết cục ra sao."""
    print("-- đổi hạt của phép chia kiểm định --")
    for hat in (0, 1, 7, 42, 2026):
        A, V, a, v = train_test_split(X_tr, y_tr, test_size=0.2, random_state=hat)
        diem = {}
        for s in SAU:
            for l in LA:
                diem[(s, l)], _ = khop_do(tien, A, a, V, v, max_depth=s,
                                          min_samples_leaf=l)
        k = max(diem, key=diem.get)
        r_t, _ = khop_do(tien, X_tr, y_tr, X_te, y_te, max_depth=k[0],
                         min_samples_leaf=k[1])
        print("  hạt %-5d chọn d%d-l%-3d | R2 kiểm định %.4f | R2 kiểm tra %.4f"
              % (hat, k[0], k[1], diem[k], r_t))


def do_ngau_nhien(tien, A, a, V, v, so_lan_toi_da=200):
    """Dò ngẫu nhiên trên một không gian ba chiều lớn, vẫn đo trên tập kiểm định."""
    kg_sau = list(range(2, 21)) + [None]
    kg_la = list(range(1, 51))
    kg_chia = list(range(2, 41))
    print("-- dò ngẫu nhiên trên không gian %d x %d x %d = %d cấu hình --"
          % (len(kg_sau), len(kg_la), len(kg_chia),
             len(kg_sau) * len(kg_la) * len(kg_chia)))
    rng = np.random.default_rng(42)
    tot, moc = -9.0, []
    for i in range(1, so_lan_toi_da + 1):
        tham_so = {"max_depth": kg_sau[rng.integers(len(kg_sau))],
                   "min_samples_leaf": int(kg_la[rng.integers(len(kg_la))]),
                   "min_samples_split": int(kg_chia[rng.integers(len(kg_chia))])}
        r, _ = khop_do(tien, A, a, V, v, **tham_so)
        if r > tot:
            tot, moc = r, moc + [(i, r)]
        if i in (10, 30, 60, 200):
            print("  sau %-3d lần lấy mẫu: R2 kiểm định tốt nhất %.4f" % (i, tot))
    in_theo_dong("các mốc tốt dần",
                 ["lần %d %.4f" % (i, r) for i, r in moc])


def tach_hai_thanh_phan(tien, A, a, V, v, X_te, y_te):
    """Lạm phát của phép chọn: đo trên cùng một lần khớp, không lẫn hiệu ứng khớp lại."""
    kd, kt = [], []
    for s in SAU:
        for l in LA:
            mo = ong_cay(tien, max_depth=s, min_samples_leaf=l).fit(A, a)
            kd.append(r2_score(v, mo.predict(V)))
            kt.append(r2_score(y_te, mo.predict(X_te)))
    kd, kt = np.array(kd), np.array(kt)
    print("-- 42 cấu hình, cùng khớp trên %d dòng --" % len(a))
    print("  trung bình (R2 kiểm định − R2 kiểm tra) trên cả lưới %+.4f | độ lệch chuẩn %.4f"
          % ((kd - kt).mean(), (kd - kt).std(ddof=1)))
    j = int(np.argmax(kd))
    print("  cấu hình thắng: kiểm định %.4f | kiểm tra %.4f | hiệu %+.4f"
          % (kd[j], kt[j], kd[j] - kt[j]))
    print("  cấu hình có R2 KIỂM TRA cao nhất: kiểm tra %.4f | kiểm định %.4f"
          % (kt.max(), kd[int(np.argmax(kt))]))
    return kd, kt


def lam_phat_theo_so_ung_vien(kd, kt, ten_bo, cac_k, so_lan_boc=2000):
    """Chọn bản tốt nhất trong k ứng viên rồi xem con số kiểm định vượt kiểm tra bao nhiêu."""
    rng = np.random.default_rng(7)
    print("-- lạm phát theo số ứng viên, %s --" % ten_bo)
    for k in cac_k:
        m1, m2 = [], []
        for _ in range(so_lan_boc):
            idx = rng.choice(len(kd), size=k, replace=False)
            j = idx[int(np.argmax(kd[idx]))]
            m1.append(kd[j])
            m2.append(kt[j])
        print("  k=%-4d | R2 kiểm định của bản thắng %.4f | R2 kiểm tra %.4f | lạm phát %+.4f"
              % (k, float(np.mean(m1)), float(np.mean(m2)),
                 float(np.mean(m1)) - float(np.mean(m2))))


def ung_vien_khac_nhau_that(tien, A, a, V, v, X_te, y_te, so_ung_vien=100):
    """100 ứng viên thực sự khác nhau: mỗi bản dùng 6 đặc trưng bốc ngẫu nhiên trong 30."""
    bo = clone(tien).fit(A)
    ten = list(bo.get_feature_names_out())
    A2 = pd.DataFrame(bo.transform(A), columns=ten, index=A.index)
    V2 = pd.DataFrame(bo.transform(V), columns=ten, index=V.index)
    T2 = pd.DataFrame(bo.transform(X_te), columns=ten, index=X_te.index)
    rng = np.random.default_rng(42)
    kd, kt = [], []
    for _ in range(so_ung_vien):
        chon = list(rng.choice(ten, size=6, replace=False))
        mo = DecisionTreeRegressor(random_state=42, max_depth=6,
                                   min_samples_leaf=10).fit(A2[chon], a)
        kd.append(r2_score(v, mo.predict(V2[chon])))
        kt.append(r2_score(y_te, mo.predict(T2[chon])))
    kd, kt = np.array(kd), np.array(kt)
    print("-- %d ứng viên khác nhau thật --" % so_ung_vien)
    print("  R2 kiểm định %.4f … %.4f | độ lệch chuẩn giữa các ứng viên %.4f"
          % (kd.min(), kd.max(), kd.std(ddof=1)))
    return kd, kt


def main() -> None:
    X_tr, X_te, y_tr, y_te, tien = nap_goi()
    A, V, a, v = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
    print("tập còn lại %d dòng | tập kiểm định %d dòng\n" % (len(a), len(v)))

    duong_cong_kiem_dinh(tien, A, a, V, v)
    print()
    diem = luoi_hai_chieu(tien, A, a, V, v)
    tot = max(diem, key=diem.get)
    r_te, mo_tot = khop_do(tien, X_tr, y_tr, X_te, y_te, max_depth=tot[0],
                           min_samples_leaf=tot[1])
    print("  khớp lại cấu hình d%d-l%d trên 2.409 dòng, đo một lần trên tập kiểm tra: R2 %.4f"
          % (tot[0], tot[1], r_te))
    for nhan, tham_so in (("d8-l20, cấu hình chốt của Chương 5", {"max_depth": 8,
                                                                 "min_samples_leaf": 20}),
                          ("d6, cấu hình của Quy trình 5.5", {"max_depth": 6})):
        r_kd, _ = khop_do(tien, A, a, V, v, **tham_so)
        r_kt, _ = khop_do(tien, X_tr, y_tr, X_te, y_te, **tham_so)
        print("  %-36s R2 kiểm định %.4f | R2 kiểm tra %.4f" % (nhan, r_kd, r_kt))
    joblib.dump(mo_tot, THU_MUC_MO_HINH / "ch07-cay-tap-kiem-dinh.joblib")
    print("  đã lưu models/ch07-cay-tap-kiem-dinh.joblib")

    print()
    doi_hat_kiem_dinh(tien, X_tr, y_tr, X_te, y_te)
    print()
    do_ngau_nhien(tien, A, a, V, v)
    print()
    kd, kt = tach_hai_thanh_phan(tien, A, a, V, v, X_te, y_te)
    lam_phat_theo_so_ung_vien(kd, kt, "lưới 42 cấu hình", (1, 5, 10, 20, 42))
    print()
    kd2, kt2 = ung_vien_khac_nhau_that(tien, A, a, V, v, X_te, y_te)
    lam_phat_theo_so_ung_vien(kd2, kt2, "100 ứng viên khác nhau thật",
                              (1, 5, 10, 20, 50, 100))


if __name__ == "__main__":
    main()
