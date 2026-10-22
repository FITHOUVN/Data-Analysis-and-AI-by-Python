"""Đo chênh lệch hiệu năng giữa các nhóm và phản ví dụ về việc bỏ thuộc tính nhóm (mục 11.5).

Chạy từ gốc dự án:
    python code/80_do_thien_lech.py

Mô hình dùng ở đây là hồi quy tuyến tính dự đoán `diem_cuoi_ky`, khớp trên TOÀN BỘ
3.012 dòng đã làm sạch. Không chia tập, vì câu hỏi của mục 11.5 không phải "mô hình dự
đoán tốt tới đâu trên dữ liệu chưa gặp" mà "mô hình đối xử với hai nhóm khác nhau thế nào",
và nhóm nhỏ nhất chỉ còn 12 dòng nếu lấy 20% làm tập kiểm tra.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed"
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut", "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky"]
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
          "hinh_thuc_hoc", "thiet_bi_chinh"]
THUOC_TINH_NHOM = ["gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc"]
NGUONG_DAT = 4.0
SO_LAN_LAY_LAI = 2000


def khop_va_du_doan(bang, cot_pl):
    tien = ColumnTransformer([("so", StandardScaler(), COT_SO),
                              ("pl", OneHotEncoder(handle_unknown="ignore"), cot_pl)])
    ong = Pipeline([("tien", tien), ("mo_hinh", LinearRegression())])
    cot = COT_SO + cot_pl
    return ong.fit(bang[cot], bang["diem_cuoi_ky"]).predict(bang[cot])


def khoang_tin_cay(sai_so, bo_sinh):
    """Khoảng tin cậy 95% của MAE một nhóm, bằng cách lấy lại mẫu có hoàn lại."""
    mau = np.array([bo_sinh.choice(sai_so, size=len(sai_so), replace=True).mean()
                    for _ in range(SO_LAN_LAY_LAI)])
    return np.percentile(mau, 2.5), np.percentile(mau, 97.5)


def khoang_hieu_mae_ghep_cap(that, du_doan_a, du_doan_b):
    """Khoảng tin cậy 95% của hiệu MAE giữa hai mô hình đo trên CÙNG một tập dòng.

    Hai mô hình dùng chung từng dòng, nên đại lượng phải dựng khoảng là hiệu **ghép cặp**
    của sai số tuyệt đối từng dòng. Dựng khoảng cho hai mẫu như thể chúng độc lập làm
    khoảng rộng hẳn ra và có thể đảo kết luận; hàm trả cả hai để thấy chênh lệch ấy.
    """
    sai_so_a = np.abs(du_doan_a - that)
    sai_so_b = np.abs(du_doan_b - that)
    hieu = np.asarray(sai_so_b - sai_so_a, dtype=float)
    n = len(hieu)
    sai_so_chuan = hieu.std(ddof=1) / math.sqrt(n)
    ghep_cap = (hieu.mean() - 1.96 * sai_so_chuan, hieu.mean() + 1.96 * sai_so_chuan)
    doc_lap = 1.96 * math.sqrt(sai_so_a.var(ddof=1) / n + sai_so_b.var(ddof=1) / n)
    return hieu.mean(), ghep_cap, doc_lap


def bang_theo_nhom(bang, du_doan, cot_nhom, bo_sinh):
    bang = bang.assign(du_doan=du_doan)
    dong = []
    for muc, phan in bang.groupby(cot_nhom):
        sai_so = (phan["du_doan"] - phan["diem_cuoi_ky"]).abs().to_numpy()
        duoi, tren = khoang_tin_cay(sai_so, bo_sinh)
        dong.append({"muc": muc, "n": len(phan), "MAE": sai_so.mean(),
                     "lech": (phan["du_doan"] - phan["diem_cuoi_ky"]).mean(),
                     "KTC_duoi": duoi, "KTC_tren": tren, "rong": tren - duoi})
    return pd.DataFrame(dong)


def in_bang(ten, bang_nhom):
    print("  -- %s" % ten)
    for _, d in bang_nhom.iterrows():
        print("     %-10s n=%5d  MAE %.4f  lệch %+.4f"
              % (d["muc"], d["n"], d["MAE"], d["lech"]))
        print("                KTC95 [%.4f; %.4f] rộng %.4f"
              % (d["KTC_duoi"], d["KTC_tren"], d["rong"]))


def co_doan_duoc_nhom_khong(bang, cot_nhom):
    """Bỏ thuộc tính nhóm khỏi đặc trưng thì thông tin nhóm có mất khỏi dữ liệu không.

    Trả về AUC của một mô hình đoán chính thuộc tính nhóm từ các đặc trưng CÒN LẠI.
    AUC xấp xỉ 0,5 nghĩa là thông tin đã mất thật; cao hơn 0,5 nghĩa là còn mã hóa gián tiếp.
    """
    con_lai = [c for c in COT_PL if c not in THUOC_TINH_NHOM]
    tien = ColumnTransformer([("so", StandardScaler(), COT_SO),
                              ("pl", OneHotEncoder(handle_unknown="ignore"), con_lai)])
    ong = Pipeline([("tien", tien), ("mo_hinh", LogisticRegression(max_iter=1000))])
    X = bang[COT_SO + con_lai]
    y = (bang[cot_nhom] == bang[cot_nhom].value_counts().index[0]).astype(int)
    xac_suat = cross_val_predict(ong, X, y, cv=5, method="predict_proba")[:, 1]
    return (roc_auc_score(y, xac_suat), accuracy_score(y, (xac_suat >= 0.5).astype(int)),
            max(y.mean(), 1 - y.mean()))


def khoang_tin_cay_hieu_ti_le(k1, n1, k2, n2):
    """Khoảng tin cậy 95% của hiệu hai tỉ lệ, xấp xỉ chuẩn, chỉ dùng thư viện đã ghim."""
    p1, p2 = k1 / n1, k2 / n2
    sai_so = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p1 - p2 - 1.96 * sai_so, p1 - p2 + 1.96 * sai_so


def ti_le_ket_qua_theo_nhom(bang, cot_nhom, lop_duong="Không đạt"):
    """Tỉ lệ lớp dương của từng mức, kèm khoảng tin cậy 95% của HIỆU và một phán quyết.

    Dùng cho mục 11.5.4: mệnh đề loại trừ giữa hai tiêu chí công bằng chỉ ràng buộc KHI
    thuộc tính nhóm và biến mục tiêu không độc lập. Phải đo điều kiện ấy, không được giả định.

    Thuộc tính có HƠN hai mức vẫn phải có phán quyết, nếu không người đọc còn lại đúng hai
    tỉ lệ thô để so bằng mắt, tức đúng việc mục 11.5.2 cấm. Ở đó hàm so **mức cao nhất với
    mức thấp nhất**: khoảng của cặp xa nhau nhất chứa 0 thì không cặp nào kết luận được.
    """
    muc = sorted(bang[cot_nhom].unique())
    ti_le = {}
    for m in muc:
        phan = bang[bang[cot_nhom] == m]
        ti_le[m] = (int((phan["ket_qua"] == lop_duong).sum()), len(phan))
    dong = ["%s %d/%d = %.2f%%" % (m, ti_le[m][0], ti_le[m][1],
                                   ti_le[m][0] / ti_le[m][1] * 100) for m in muc]
    ket = "  %s: %s" % (cot_nhom, " | ".join(dong))
    cao = max(muc, key=lambda m: ti_le[m][0] / ti_le[m][1])
    thap = min(muc, key=lambda m: ti_le[m][0] / ti_le[m][1])
    hieu, duoi, tren = khoang_tin_cay_hieu_ti_le(*ti_le[cao], *ti_le[thap])
    nhan = "hiệu" if len(muc) == 2 else "hiệu cặp xa nhau nhất"
    ket += ("\n     %s (%s trừ %s) %+.4f | KTC95 [%+.4f; %+.4f] | chứa 0: %s"
            % (nhan, cao, thap, hieu, duoi, tren, "có" if duoi * tren < 0 else "không"))
    return ket


def main():
    print("Lưu ý: đo trong mẫu huấn luyện; không suy rộng thành hiệu năng hoặc công bằng ngoài mẫu.")
    bang = pd.read_csv(VAO / "du-lieu-sach.csv")
    bo_sinh = np.random.default_rng(20260101)

    print("-- 11.5.4 điều kiện của mệnh đề loại trừ: tỉ lệ lớp dương có khác nhau theo nhóm?")
    for cot in THUOC_TINH_NHOM:
        print(ti_le_ket_qua_theo_nhom(bang, cot))

    co_nhom = khop_va_du_doan(bang, COT_PL)
    bo_nhom = khop_va_du_doan(bang, [c for c in COT_PL
                                     if c not in THUOC_TINH_NHOM])

    print("-- 11.5.2 chênh lệch giữa các nhóm, mô hình CÓ dùng ba thuộc tính nhóm")
    for cot in ["vua_lam_vua_hoc", "gioi_tinh"]:
        in_bang(cot, bang_theo_nhom(bang, co_nhom, cot, bo_sinh))
    sai_so = np.abs(co_nhom - bang["diem_cuoi_ky"])
    print("     độ lệch chuẩn của sai số tuyệt đối (ddof=1): %.4f" % sai_so.std(ddof=1))
    for delta in (0.0750, 0.1500):
        print("     cỡ nhóm cần để khoảng tin cậy hẹp hơn %.4f điểm: n >= %d"
              % (delta, int(np.ceil((2 * 1.96 * sai_so.std(ddof=1) / delta) ** 2))))

    print("-- 11.5.3 cùng mô hình, cùng dữ liệu, BỎ ba thuộc tính nhóm")
    in_bang("vua_lam_vua_hoc", bang_theo_nhom(bang, bo_nhom, "vua_lam_vua_hoc", bo_sinh))

    doi = (co_nhom >= NGUONG_DAT) != (bo_nhom >= NGUONG_DAT)
    print("  ba con số của phép so:"
          " đổi %d/%d kết luận quanh ngưỡng %.1f"
          % (doi.sum(), len(bang), NGUONG_DAT))
    print("     lệch dự đoán lớn nhất %.4f | lệch MAE %+.4f"
          % (np.abs(co_nhom - bo_nhom).max(),
             mean_absolute_error(bang["diem_cuoi_ky"], bo_nhom)
             - mean_absolute_error(bang["diem_cuoi_ky"], co_nhom)))
    lech, (duoi, tren), nua_doc_lap = khoang_hieu_mae_ghep_cap(
        bang["diem_cuoi_ky"].to_numpy(), co_nhom, bo_nhom)
    print("  lệch MAE %+.4f = %.1f%% của MAE toàn tập %.4f"
          % (lech, lech / mean_absolute_error(bang["diem_cuoi_ky"], bo_nhom) * 100,
             mean_absolute_error(bang["diem_cuoi_ky"], bo_nhom)))
    print("     KTC95 ghép cặp [%+.4f; %+.4f] | chứa 0: %s"
          % (duoi, tren, "có" if duoi * tren < 0 else "không"))
    print("     nếu dựng khoảng như hai mẫu độc lập: nửa độ rộng %.4f, lớn hơn chính lệch"
          " nên kết luận đảo thành \"không phân biệt được\"" % nua_doc_lap)
    huong = bang.assign(a=co_nhom, b=bo_nhom)[doi]
    for muc, phan in huong.groupby("vua_lam_vua_hoc"):
        xuong = ((phan["a"] >= NGUONG_DAT) & (phan["b"] < NGUONG_DAT)).sum()
        tren = ((phan["a"] < NGUONG_DAT) & (phan["b"] >= NGUONG_DAT)).sum()
        print("     %-10s %2d kết luận đổi" % (muc, len(phan)))
        print("                %d xuống dưới ngưỡng, %d lên trên ngưỡng"
              % (xuong, tren))

    auc, acc, da_so = co_doan_duoc_nhom_khong(bang, "vua_lam_vua_hoc")
    print("  đoán lại `vua_lam_vua_hoc` từ các đặc trưng còn lại, xác thực chéo 5 phần:"
          " AUC %.4f | accuracy %.4f | tỉ lệ lớp đa số %.4f" % (auc, acc, da_so))


if __name__ == "__main__":
    main()
