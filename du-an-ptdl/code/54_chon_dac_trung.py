"""Lựa chọn đặc trưng cho bài toán phân loại: bỏ cột gì, giữ cột gì, và đo bằng gì (mục 7.1)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"
BO_KHOI_DAC_TRUNG = ["ma_sv", "ngay_dang_ky", "tinh_thanh", "diem_cuoi_ky", "ket_qua"]
COT_MUC_TIEU = "ket_qua"
LOP_DUONG = "Không đạt"
BON_COT_YEU = ["nganh", "nhom_tuoi", "so_lan_dien_dan", "gioi_tinh"]


def nap_goi():
    """Nạp bảng đầy đủ, bộ tiền xử lý, rồi chia phân tầng đúng như Mã 6.2."""
    bang = pd.read_parquet(VAO / "ch04-bang-day-du.parquet")
    tien = joblib.load(THU_MUC_MO_HINH / "bo-tien-xu-ly.joblib")
    X = bang.drop(columns=BO_KHOI_DAC_TRUNG)
    y = (bang[COT_MUC_TIEU] == LOP_DUONG).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, tien


def bo_tien_theo_cot(bang_goc, cot):
    """Dựng lại bộ tiền xử lý của Chương 4 cho một tập cột đã chọn."""
    so = [c for c in cot if pd.api.types.is_numeric_dtype(bang_goc[c])]
    pl = [c for c in cot if c not in so]
    return ColumnTransformer([("so", StandardScaler(), so),
                              ("pl", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), pl)])


def bang_dac_trung(tien, X_tr, X_te):
    """Biến đổi hai tập một lần, trả về hai bảng 30 cột kèm tên cột."""
    bo = clone(tien).fit(X_tr)
    ten = list(bo.get_feature_names_out())
    A_tr = pd.DataFrame(bo.transform(X_tr), columns=ten, index=X_tr.index)
    A_te = pd.DataFrame(bo.transform(X_te), columns=ten, index=X_te.index)
    return bo, ten, A_tr, A_te


def do_bien_thien(A_tr, ten):
    """Cột nào gần như không biến thiên: phương sai và tỉ lệ ô bằng 1."""
    pv = A_tr.var(ddof=0).sort_values()
    print("-- năm đặc trưng ít biến thiên nhất trên 2.409 dòng huấn luyện --")
    for t, giatri in pv.head(5).items():
        print("  %-28s phương sai %.5f | tỉ lệ 1 %.4f"
              % (t[4:], giatri, A_tr[t].mean()))
    hiem = [t for t in ten if t.startswith("pl__") and A_tr[t].mean() < 0.02]
    for t in hiem:
        print("  mức hiếm: %s có %d dòng trên 2.409" % (t, int(A_tr[t].sum())))
    return hiem


def do_tuong_quan(A_tr, ten):
    """Sáu cặp cột số có |r| lớn nhất: quy tắc bỏ cột trùng thông tin có kích hoạt không."""
    so = [t for t in ten if t.startswith("so__")]
    C = A_tr[so].corr().abs()
    cap = [(C.iloc[i, j], so[i], so[j])
           for i in range(len(so)) for j in range(i + 1, len(so))]
    print("-- sáu cặp cột số tương quan mạnh nhất --")
    for r, a, b in sorted(cap, reverse=True)[:6]:
        print("  |r| %.4f  %s ~ %s" % (r, a[4:], b[4:]))
    return max(r for r, _, _ in cap)


def khop_tren_tap_cot(A_tr, A_te, y_tr, giu, mo_hinh=None):
    """Khớp một mô hình trên đúng tập cột đã chọn, trả về nhãn và xác suất."""
    mo = clone(mo_hinh or LogisticRegression(max_iter=1000)).fit(A_tr[giu], y_tr)
    return mo, mo.predict(A_te[giu]), mo.predict_proba(A_te[giu])[:, 1]


def mot_dong(ten_cach, y_te, nhan, xac_suat, so_cot):
    print("  %-34s k=%2d | rec %.4f | pre %.4f | F1 %.4f | AUC %.4f"
          % (ten_cach, so_cot, recall_score(y_te, nhan),
             precision_score(y_te, nhan), f1_score(y_te, nhan),
             roc_auc_score(y_te, xac_suat)))


def ba_con_so(ten_cach, y_te, n0, p0, n1, p1):
    """Ba con số bắt buộc của mọi phép so: số nhãn đổi, lệch xác suất, lệch F1."""
    print("  %s: đổi %d/%d nhãn" % (ten_cach, int((n0 != n1).sum()), len(y_te)))
    print("    lệch xác suất lớn nhất %.4f | lệch F1 %+.4f"
          % (float(np.abs(p0 - p1).max()),
             f1_score(y_te, n1) - f1_score(y_te, n0)))


def ba_cach_xep_hang(A_tr, y_tr, ten):
    """Ba cách xếp hạng đặc trưng cho ba thứ tự khác nhau."""
    F = pd.Series(f_classif(A_tr, y_tr)[0], index=ten).sort_values(ascending=False)
    rung = RandomForestClassifier(n_estimators=200, random_state=42).fit(A_tr, y_tr)
    G = pd.Series(rung.feature_importances_, index=ten).sort_values(ascending=False)
    lg = LogisticRegression(max_iter=1000).fit(A_tr, y_tr)
    W = pd.Series(np.abs(lg.coef_[0]), index=ten).sort_values(ascending=False)
    print("-- ba cách xếp hạng, bốn đặc trưng đứng đầu --")
    for nhan, xh, dang in (("f_classif", F, "%.0f"), ("gini rừng", G, "%.4f"),
                           ("|w| logistic", W, "%.4f")):
        print("  %s:" % nhan)
        o = [("%s " + dang) % (t[4:], giatri) for t, giatri in xh.head(4).items()]
        for i in range(0, len(o), 2):
            print("    " + " · ".join(o[i:i + 2]))
    print("  giao của mười cột đầu")
    print("    f_classif với gini %d/10 | f_classif với |w| %d/10"
          % (len(set(F.head(10).index) & set(G.head(10).index)),
             len(set(F.head(10).index) & set(W.head(10).index))))
    return F


def quet_k(A_tr, A_te, y_tr, y_te, F):
    """SelectKBest theo f_classif: F1 trên tập kiểm tra không đơn điệu theo k."""
    print("-- giữ k đặc trưng đứng đầu theo f_classif --")
    for k in (5, 10, 15, 20, 25, 30):
        _, nh, ps = khop_tren_tap_cot(A_tr, A_te, y_tr, list(F.head(k).index))
        print("  k=%2d | rec %.4f | pre %.4f | F1 %.4f | AUC %.4f"
              % (k, recall_score(y_te, nh), precision_score(y_te, nh),
                 f1_score(y_te, nh), roc_auc_score(y_te, ps)))


def in_danh_sach_cot(nhan, cot, be_rong=62):
    """In một danh sách tên cột theo chiều dọc, mỗi dòng không quá be_rong ký tự."""
    dong = "    %s:" % nhan
    for i, c in enumerate(cot):
        them = c + ("," if i < len(cot) - 1 else "")
        if len(dong) + 1 + len(them) > be_rong:
            print(dong)
            dong = "      " + them
        else:
            dong += " " + them
    print(dong)


def quet_rfe(A_tr, A_te, y_tr, y_te, ten):
    """RFE bỏ dần đặc trưng theo vòng; in cả tập cột nó giữ lại."""
    print("-- RFE với hồi quy logistic --")
    for k in (5, 8, 10, 20):
        chon = RFE(LogisticRegression(max_iter=1000),
                   n_features_to_select=k, step=1).fit(A_tr, y_tr)
        giu = [t for t, g in zip(ten, chon.support_) if g]
        _, nh, ps = khop_tren_tap_cot(A_tr, A_te, y_tr, giu)
        print("  k=%2d | F1 %.4f | AUC %.4f"
              % (k, f1_score(y_te, nh), roc_auc_score(y_te, ps)))
        in_danh_sach_cot("bốn cột đầu", [t[4:] for t in giu[:4]])
        if k == 10:
            in_danh_sach_cot("mười cột RFE giữ",
                             [t.split("__", 1)[1] for t in giu])


def hieu_qua_nhieu_lan_chia(bang_a, cot_a, bang_b, cot_b, y, tao_mo_hinh,
                            ten_cach, so_lan=10):
    """Đo chính cái HIỆU giữa hai tập đặc trưng qua nhiều lần chia phân tầng."""
    gom = {k: [] for k in ("F1", "recall", "AUC")}
    for hat in range(so_lan):
        chi_so = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=hat, stratify=y)
        diem = {}
        for nhan, bang, cot in (("a", bang_a, cot_a), ("b", bang_b, cot_b)):
            A, B = bang.iloc[chi_so[0]][cot], bang.iloc[chi_so[1]][cot]
            a, b = y.iloc[chi_so[0]], y.iloc[chi_so[1]]
            mo = Pipeline([("tien", bo_tien_theo_cot(bang, cot)),
                           ("mo_hinh", tao_mo_hinh())]).fit(A, a)
            nh, ps = mo.predict(B), mo.predict_proba(B)[:, 1]
            diem[nhan] = {"F1": f1_score(b, nh), "recall": recall_score(b, nh),
                          "AUC": roc_auc_score(b, ps)}
        for k in gom:
            gom[k].append(diem["b"][k] - diem["a"][k])
    print("  %s" % ten_cach)
    for k, v in gom.items():
        v = np.array(v)
        print("    hiệu %-7s dương %2d/%d | %+.4f … %+.4f | trung bình %+.4f"
              % (k, int((v > 0).sum()), so_lan, v.min(), v.max(), v.mean()))
    return gom


def cot_nhieu_ngau_nhien(X, so_cot=10):
    """Thêm các cột nhiễu thuần để biết một cột vô dụng thật thì gây hại bao nhiêu."""
    rng = np.random.default_rng(7)
    X_nhieu = X.copy()
    for i in range(so_cot):
        X_nhieu["nhieu_%d" % i] = rng.normal(size=len(X_nhieu))
    return X_nhieu


def main() -> None:
    X, y, X_tr, X_te, y_tr, y_te, tien = nap_goi()
    bo, ten, A_tr, A_te = bang_dac_trung(tien, X_tr, X_te)
    print("14 cột gốc thành %d đặc trưng sau biến đổi\n" % len(ten))

    hiem = do_bien_thien(A_tr, ten)
    print()
    do_tuong_quan(A_tr, ten)

    print("\n-- mốc và ba phép bỏ cột, hồi quy logistic, lần chia random_state=42 --")
    mo0, n0, p0 = khop_tren_tap_cot(A_tr, A_te, y_tr, ten)
    mot_dong("đủ 30 đặc trưng", y_te, n0, p0, len(ten))
    giu1 = [t for t in ten if t not in hiem]
    _, n1, p1 = khop_tren_tap_cot(A_tr, A_te, y_tr, giu1)
    mot_dong("bỏ cột mức hiếm", y_te, n1, p1, len(giu1))
    ba_con_so("bỏ cột mức hiếm", y_te, n0, p0, n1, p1)
    lech = np.abs(p0 - p1)
    la_hiem = (X_te["thiet_bi_chinh"] == "Không rõ").to_numpy()
    print("  lệch xác suất: lớn nhất %.4f trong %d dòng 'Không rõ', lớn nhất %.4f ở %d dòng còn lại"
          % (lech[la_hiem].max(), int(la_hiem.sum()),
             lech[~la_hiem].max(), int((~la_hiem).sum())))

    giu2 = [t for t in ten
            if not any(t.split("__", 1)[1].startswith(c) for c in BON_COT_YEU)]
    _, n2, p2 = khop_tren_tap_cot(A_tr, A_te, y_tr, giu2)
    mot_dong("bỏ 4 cột yếu nhất Bảng 6.8", y_te, n2, p2, len(giu2))
    ba_con_so("bỏ 4 cột yếu nhất", y_te, n0, p0, n2, p2)

    print()
    F = ba_cach_xep_hang(A_tr, y_tr, ten)
    print()
    quet_k(A_tr, A_te, y_tr, y_te, F)
    print()
    quet_rfe(A_tr, A_te, y_tr, y_te, ten)

    print("\n-- hiệu qua mười lần chia: bản 10 cột trừ bản 14 cột --")
    cot_10 = [c for c in X.columns if c not in BON_COT_YEU]
    hieu_qua_nhieu_lan_chia(X, list(X.columns), X, cot_10, y,
                            lambda: LogisticRegression(max_iter=1000),
                            "hồi quy logistic")
    hieu_qua_nhieu_lan_chia(X, list(X.columns), X, cot_10, y,
                            lambda: RandomForestClassifier(n_estimators=200,
                                                           random_state=42),
                            'rừng 200 cây, max_features="sqrt" (mặc định)')
    hieu_qua_nhieu_lan_chia(X, list(X.columns), X, cot_10, y,
                            lambda: RandomForestClassifier(n_estimators=200,
                                                           random_state=42,
                                                           max_features=None),
                            "rừng 200 cây, max_features=None")

    print("\n-- hiệu qua mười lần chia: thêm mười cột nhiễu trừ bản 14 cột --")
    X_nhieu = cot_nhieu_ngau_nhien(X)
    hieu_qua_nhieu_lan_chia(X_nhieu, list(X.columns), X_nhieu, list(X_nhieu.columns), y,
                            lambda: RandomForestClassifier(n_estimators=200,
                                                           random_state=42),
                            'rừng 200 cây, max_features="sqrt"')
    hieu_qua_nhieu_lan_chia(X_nhieu, list(X.columns), X_nhieu, list(X_nhieu.columns), y,
                            lambda: LogisticRegression(max_iter=1000),
                            "hồi quy logistic")

    print("\n-- cột trùng thông tin hoàn toàn: diem_tb_bai_tap theo thang 100 --")
    X_doi = X.copy()
    X_doi["diem_bt_thang_100"] = X_doi["diem_tb_bai_tap"] * 10
    bo2 = bo_tien_theo_cot(X_doi, list(X_doi.columns)).fit(X_doi.loc[X_tr.index])
    ten31 = list(bo2.get_feature_names_out())
    B_tr = pd.DataFrame(bo2.transform(X_doi.loc[X_tr.index]), columns=ten31, index=X_tr.index)
    B_te = pd.DataFrame(bo2.transform(X_doi.loc[X_te.index]), columns=ten31, index=X_te.index)
    mo_doi, n3, p3 = khop_tren_tap_cot(B_tr, B_te, y_tr, ten31)
    w1 = mo0.coef_[0][ten.index("so__diem_tb_bai_tap")]
    w2a = mo_doi.coef_[0][ten31.index("so__diem_tb_bai_tap")]
    w2b = mo_doi.coef_[0][ten31.index("so__diem_bt_thang_100")]
    print("  hệ số: một cột %+.4f | hai cột %+.4f và %+.4f, tổng %+.4f"
          % (w1, w2a, w2b, w2a + w2b))
    ba_con_so("thêm cột sinh đôi, logistic", y_te, n0, p0, n3, p3)
    rung1 = RandomForestClassifier(n_estimators=200, random_state=42).fit(A_tr, y_tr)
    rung2 = RandomForestClassifier(n_estimators=200, random_state=42).fit(B_tr, y_tr)
    g1 = pd.Series(rung1.feature_importances_, index=ten)
    g2 = pd.Series(rung2.feature_importances_, index=ten31)
    print("  rừng: gini của diem_tb_bai_tap %.4f -> %.4f + %.4f = %.4f"
          % (g1["so__diem_tb_bai_tap"], g2["so__diem_tb_bai_tap"],
             g2["so__diem_bt_thang_100"],
             g2["so__diem_tb_bai_tap"] + g2["so__diem_bt_thang_100"]))
    print("    xếp hạng %d -> %d"
          % (list(g1.sort_values(ascending=False).index).index("so__diem_tb_bai_tap") + 1,
             list(g2.sort_values(ascending=False).index).index("so__diem_tb_bai_tap") + 1))
    print("  rừng F1 trên tập kiểm tra: %.4f -> %.4f"
          % (f1_score(y_te, rung1.predict(A_te)), f1_score(y_te, rung2.predict(B_te))))
    hieu_qua_nhieu_lan_chia(X_doi, list(X.columns), X_doi, list(X_doi.columns), y,
                            lambda: RandomForestClassifier(n_estimators=200,
                                                           random_state=42),
                            "rừng 200 cây, bản có cột sinh đôi trừ bản không có")


if __name__ == "__main__":
    main()
