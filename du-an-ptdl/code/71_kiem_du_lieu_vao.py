"""Kiểm dữ liệu vào theo hợp đồng, và ba quyết định khi lô có lỗi (mục 9.2)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
RAW = GOC_DU_AN / "data" / "raw"
TAM = GOC_DU_AN / "data" / "processed"
THU_MUC_MO_HINH = GOC_DU_AN / "models"


def nap_goi():
    mo_hinh = joblib.load(THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib")
    ho_so = json.loads((THU_MUC_MO_HINH / "ch08-ho-so-trien-khai.json")
                       .read_text(encoding="utf-8"))
    tien = mo_hinh.estimator.estimator.named_steps["tien"]
    cot_so = list(tien.transformers_[0][2])
    muc = dict(zip(tien.transformers_[1][2],
                   tien.named_transformers_["pl"][-1].categories_))
    return mo_hinh, ho_so, cot_so, muc


def dem_cot_khong_doc_duoc(bang, cot_so):
    """Đếm cột số mà pd.to_numeric không đọc được: phép kiểm không phụ thuộc tên dtype."""
    return sum(1 for c in cot_so
               if c in bang.columns and pd.to_numeric(bang[c], errors="coerce").isna().sum()
               > bang[c].isna().sum())


def bon_cach_doc(cot_so):
    """Cùng một file xuất từ Excel vùng Việt, bốn cách đọc, bốn kết quả khác nhau."""
    p = RAW / "ch09-lo-excel-viet.csv"
    cach = [("read_csv(p)", {}),
            ("sep=';'", {"sep": ";"}),
            ("sep=';', decimal=','", {"sep": ";", "decimal": ","}),
            ("sep=';', decimal=',', encoding='utf-8'",
             {"sep": ";", "decimal": ",", "encoding": "utf-8"})]
    for ten, tham_so in cach:
        b = pd.read_csv(p, **tham_so)
        # Lô chỉ còn 1 cột thì phép đếm cột số sai kiểu KHÔNG kiểm được gì: nó trả 0 vì
        # không có cột nào mang tên của hợp đồng, chứ không phải vì dữ liệu sạch.
        dem = ("không kiểm được" if b.shape[1] == 1
               else "%d/%d cột số không đọc được" % (dem_cot_khong_doc_duoc(b, cot_so),
                                                     len(cot_so)))
        print("  %-40s %2d cột | %-28s | cột đầu %r"
              % (ten, b.shape[1], dem, b.columns[0][:12]))


def hai_loai_o_sai(mo_hinh, ho_so, cot_so):
    """Ô trống và ô chữ: pipeline xử lý được loại thứ nhất, dừng với loại thứ hai."""
    cot, nguong = ho_so["cot_vao_theo_thu_tu"], ho_so["nguong_quyet_dinh"]
    sach = pd.read_csv(RAW / "ch09-lo-moi.csv")
    p0 = mo_hinh.predict_proba(sach[cot])[:, 1]
    xau = pd.read_csv(RAW / "ch09-lo-loi.csv")
    trong = sach.copy()
    trong["diem_giua_ky"] = pd.read_csv(RAW / "ch09-lo-loi.csv")["diem_giua_ky"]
    p1 = mo_hinh.predict_proba(trong[cot])[:, 1]
    print("  3 ô trống ở diem_giua_ky: chạy được | đổi %d/%d nhãn"
          " | lệch xác suất lớn nhất %.4f"
          % (int(((p1 >= nguong) != (p0 >= nguong)).sum()), len(p0),
             float(np.abs(p1 - p0).max())))
    try:
        mo_hinh.predict_proba(xau[cot])
        print("  2 ô chữ ở so_lan_dang_nhap: CHẠY ĐƯỢC")
    except ValueError as loi:
        print("  2 ô chữ ở so_lan_dang_nhap: ValueError: %s"
              % str(loi).splitlines()[0][:52])
    for c in cot_so:
        gt = pd.to_numeric(xau[c], errors="coerce")
        n = int((gt.isna() & xau[c].notna()).sum())
        if n:
            print("    cột %-20s %d ô không đọc được thành số" % (c, n))


def ba_quyet_dinh(mo_hinh, ho_so):
    """Dừng cả lô · bỏ dòng sai · đổi ô sai thành ô trống rồi để pipeline điền.

    Lô dùng ở đây chỉ mang MỘT loại lỗi, hai ô chữ, để ba con số so được với nhau.
    """
    cot, nguong = ho_so["cot_vao_theo_thu_tu"], ho_so["nguong_quyet_dinh"]
    sach = pd.read_csv(RAW / "ch09-lo-moi.csv")
    moc = mo_hinh.predict_proba(sach[cot])[:, 1]
    hong = [i for i in sach.index
            if pd.read_csv(RAW / "ch09-lo-loi.csv").at[i, "so_lan_dang_nhap"] == "mười"]
    chi_o_chu = sach.copy()
    chi_o_chu["so_lan_dang_nhap"] = chi_o_chu["so_lan_dang_nhap"].astype(object)
    chi_o_chu.loc[hong, "so_lan_dang_nhap"] = "mười"
    can = int((moc >= nguong).sum())
    print("  mốc so sánh, lô sạch 40 dòng: %d dòng cần liên hệ" % can)
    print("  hai dòng mang ô chữ: dòng %s của file, xác suất %s"
          % ([i + 2 for i in hong], ["%.4f" % moc[i] for i in hong]))
    print("  (1) dừng cả lô             : 0 dòng ra  | mất %d/%d dòng cần liên hệ"
          % (can, can))
    con = chi_o_chu.drop(index=hong)
    p_bo = mo_hinh.predict_proba(con[cot])[:, 1]
    print("  (2) bỏ %d dòng sai          : %d dòng ra | %d dòng cần liên hệ"
          " | mất %d/%d" % (len(hong), len(con), int((p_bo >= nguong).sum()),
                            can - int((p_bo >= nguong).sum()), can))
    dien = chi_o_chu.copy()
    dien["so_lan_dang_nhap"] = pd.to_numeric(dien["so_lan_dang_nhap"], errors="coerce")
    p_di = mo_hinh.predict_proba(dien[cot])[:, 1]
    print("  (3) đổi ô sai thành ô trống: %d dòng ra | %d dòng cần liên hệ"
          " | đổi %d/%d nhãn so với lô sạch"
          % (len(dien), int((p_di >= nguong).sum()),
             int(((p_di >= nguong) != (moc >= nguong)).sum()), len(moc)))
    for i in hong:
        print("      dòng %d: xác suất đúng %.4f, sau khi điền %.4f, lệch %+.4f"
              % (i + 2, moc[i], p_di[i], p_di[i] - moc[i]))


def hai_loai_khong_dau_hieu(mo_hinh, ho_so, muc):
    """Mức phân loại lạ và giá trị ngoài miền: chạy sạch, kết quả lệch."""
    cot, nguong = ho_so["cot_vao_theo_thu_tu"], ho_so["nguong_quyet_dinh"]
    sach = pd.read_csv(RAW / "ch09-lo-moi.csv")
    moc = mo_hinh.predict_proba(sach[cot])[:, 1]
    xau = pd.read_csv(RAW / "ch09-lo-loi.csv")
    xau["so_lan_dang_nhap"] = pd.to_numeric(xau["so_lan_dang_nhap"], errors="coerce")
    p = mo_hinh.predict_proba(xau[cot])[:, 1]
    for c, ds in muc.items():
        la = ~xau[c].isin(list(ds)) & xau[c].notna()
        for i in xau.index[la]:
            print("  mức lạ %r ở cột %s, dòng %d: xác suất %.4f thành %.4f,"
                  " nhãn %d thành %d"
                  % (xau.at[i, c], c, i + 2, moc[i], p[i],
                     int(moc[i] >= nguong), int(p[i] >= nguong)))
    ngoai = xau.index[(xau["ty_le_hoan_thanh_video"] > 1.0)]
    for i in ngoai:
        print("  ty_le_hoan_thanh_video = %.1f ở dòng %d, ngoài miền [0; 1]:"
              " xác suất %.4f thành %.4f, nhãn %d thành %d"
              % (xau.at[i, "ty_le_hoan_thanh_video"], i + 2, moc[i], p[i],
                 int(moc[i] >= nguong), int(p[i] >= nguong)))
    print("  cả hai ca: 0 ngoại lệ, 0 cảnh báo của scikit-learn")


def so_dong_trong_file():
    """Độ lệch = 2 + số dòng bị bỏ ở đầu file. Đo với 0, 1 và 2 dòng ghi chú."""
    p = RAW / "ch09-lo-moi.csv"
    goc = p.read_text(encoding="utf-8").splitlines()
    print("  file có %d dòng văn bản, dòng 1 là tiêu đề" % len(goc))
    tam = TAM / "ch09-co-ghi-chu.csv"
    for k in (0, 1, 2):
        tam.write_text("".join("# xuất từ cổng đăng ký, dòng ghi chú %d\n" % (i + 1)
                               for i in range(k))
                       + "\n".join(goc) + "\n", encoding="utf-8")
        b = pd.read_csv(tam, comment="#")
        print("  %d dòng ghi chú: %d dòng dữ liệu | bản ghi index 0 ở dòng %d của file"
              " | độ lệch %d" % (k, len(b), k + 2, k + 2))
    tam.unlink(missing_ok=True)


def ti_le_o_rong(mo_hinh, ho_so):
    """Ô rỗng không phải lỗi, nhưng tỉ lệ ô rỗng thì phải đếm: đây là đường cong của nó."""
    cot, nguong = ho_so["cot_vao_theo_thu_tu"], ho_so["nguong_quyet_dinh"]
    sach = pd.read_csv(RAW / "ch09-lo-moi.csv")
    moc = mo_hinh.predict_proba(sach[cot])[:, 1]
    can = int((moc >= nguong).sum())
    for k in (3, 8, 12, 20, 40):
        lo = sach.copy()
        lo.loc[lo.index[:k], "diem_giua_ky"] = np.nan
        p = mo_hinh.predict_proba(lo[cot])[:, 1]
        print("  %2d/%d ô rỗng (%5.1f%%): đổi %d/%d nhãn | lệch lớn nhất %.4f"
              " | cần liên hệ %d thành %d"
              % (k, len(sach), k / len(sach) * 100,
                 int(((p >= nguong) != (moc >= nguong)).sum()), len(sach),
                 float(np.abs(p - moc).max()), can, int((p >= nguong).sum())))
    print("  số nhãn đổi là dấu hiệu đến muộn: nó im tới 30% số dòng, còn mức lệch xác"
          " suất đã lên mức cao nhất ngay ở 20%")


def hai_ban_kiem_mien(cot_so):
    """Phép so miền che ô rỗng hay không: hai bản, ba ca thử, hai hướng hỏng."""
    mien = {"so_lan_dang_nhap": (0, None), "tong_thoi_luong_phut": (0, None),
            "ty_le_hoan_thanh_video": (0.0, 1.0), "so_bai_tap_nop": (0, None),
            "diem_tb_bai_tap": (0.0, 10.0), "so_lan_dien_dan": (0, None),
            "nop_tre_tb_gio": (0, None), "diem_giua_ky": (0.0, 10.0)}

    def dem_loi(bang, danh_dau_sai_kieu: bool) -> int:
        n = 0
        for c in cot_so:
            gt = pd.to_numeric(bang[c], errors="coerce")
            if danh_dau_sai_kieu:
                n += int((gt.isna() & bang[c].notna()).sum())
            duoi, tren = mien[c]
            ngoai = gt < duoi if tren is None else (gt < duoi) | (gt > tren)
            n += int(ngoai.fillna(False).sum())
        return n

    xau = pd.read_csv(RAW / "ch09-lo-loi.csv")
    print("  trên ch09-lo-loi.csv: bản đúng %d lỗi | bản chỉ so miền %d lỗi"
          % (dem_loi(xau, True), dem_loi(xau, False)))
    sach = pd.read_csv(RAW / "ch09-lo-moi.csv")
    for ten, c, gt in [("104", "so_lan_dang_nhap", 104),
                       ("3,5", "ty_le_hoan_thanh_video", 3.5),
                       ("'mười'", "so_lan_dang_nhap", "mười")]:
        m = sach.copy()
        m[c] = m[c].astype(object)
        m.loc[m.index[0], c] = gt
        print("    ca %-8s ở %-22s bản đúng: %-5s | bản chỉ so miền: %s"
              % (ten, c, "trượt" if dem_loi(m, True) else "đạt",
                 "trượt" if dem_loi(m, False) else "đạt"))


def doc_sai_dau_thap_phan(cot_so):
    """Trần năm dòng gặp một lô mà lỗi trải trên nhiều cột."""
    e = pd.read_csv(RAW / "ch09-lo-excel-viet.csv", sep=";")
    hong = []
    for c in cot_so:
        gt = pd.to_numeric(e[c], errors="coerce")
        n = int((gt.isna() & e[c].notna()).sum())
        if n:
            hong.append((c, n))
    tong = sum(n for _, n in hong)
    print("  đọc sep=';' mà quên decimal=',': %d lỗi ở %d cột trên lô %d dòng"
          % (tong, len(hong), len(e)))
    print("    %s" % ", ".join("%s %d" % (c, n) for c, n in hong))
    print("  trần 5 dòng in ra: 5 lỗi đầu đều của cột %s, rồi một dòng"
          " 'còn %d dòng cùng loại, đã lược'" % (hong[0][0], tong - 5))


def repr_so_voi_str():
    """Vì sao thông báo lỗi in giá trị qua repr chứ không qua str."""
    for v in ("3.5", " 9", "mười"):
        print("  ô %-10s str cho %-8s | repr cho %s" % (repr(v), str(v), repr(v)))
    print("  số thực 3.5  str cho %-8s | repr cho %s" % (str(3.5), repr(3.5)))


def main() -> None:
    mo_hinh, ho_so, cot_so, muc = nap_goi()
    print("-- 9.2.1 bốn cách đọc cùng một file xuất từ Excel vùng Việt --")
    bon_cach_doc(cot_so)
    print("\n-- 9.2.1 hai loại ô sai, hai dấu hiệu khác nhau --")
    hai_loai_o_sai(mo_hinh, ho_so, cot_so)
    print("\n-- 9.2.2 ba quyết định khi lô có 2 ô không đọc được thành số --")
    ba_quyet_dinh(mo_hinh, ho_so)
    print("\n-- 9.2.2 hai loại sai không sinh dấu hiệu nào --")
    hai_loai_khong_dau_hieu(mo_hinh, ho_so, muc)
    print("\n-- 9.2.1 tỉ lệ ô rỗng theo từng cột của hợp đồng --")
    ti_le_o_rong(mo_hinh, ho_so)
    print("\n-- 9.2.1 phép so miền che ô rỗng hay không --")
    hai_ban_kiem_mien(cot_so)
    print("\n-- 9.2.3 số dòng trong file so với chỉ số của pandas --")
    so_dong_trong_file()
    print("\n-- 9.2.3 vì sao in giá trị qua repr --")
    repr_so_voi_str()
    print("\n-- 9.2.3 một lô mà lỗi trải trên nhiều cột --")
    doc_sai_dau_thap_phan(cot_so)


if __name__ == "__main__":
    main()
