"""Log so với print, mã trả về, và rò rỉ dữ liệu cá nhân trong file log (mục 9.4)."""
import ast
import json
import logging
import re
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
THU_MUC_MA = Path(__file__).resolve().parent
RAW = GOC_DU_AN / "data" / "raw"
TAM = GOC_DU_AN / "data" / "processed"
CHUONG_TRINH = [sys.executable, str(THU_MUC_MA / "69_du_doan.py")]
# Đường dẫn chắc chắn không mở được, dựng bằng cách lấy CHA là một tệp thường. Cách này
# đúng trên mọi máy, khác với một ổ đĩa tưởng tượng như Z: vốn có thể tồn tại thật.
CHA_LA_TEP = RAW / "ch09-lo-moi.csv"


def goi(tham_so, gop_stderr=False):
    """Chạy chương trình dự đoán, trả về (mã trả về, stdout, stderr)."""
    kq = subprocess.run(CHUONG_TRINH + tham_so, stdout=subprocess.PIPE, text=True,
                        encoding="utf-8", errors="replace",
                        stderr=subprocess.STDOUT if gop_stderr else subprocess.PIPE)
    return kq.returncode, kq.stdout or "", kq.stderr or ""


def log_ra_dau(lo):
    """Log ra stderr thì đường ống sạch; gộp log vào stdout thì pandas không đọc được."""
    for ten, gop in [("log ra stderr, stdout chỉ có kết quả", False),
                     ("log gộp vào stdout", True)]:
        ma, ra, _ = goi(["--vao", str(lo), "--log-muc", "INFO"], gop_stderr=gop)
        try:
            b = pd.read_csv(StringIO(ra))
            print("  %-38s mã %d | pandas đọc được %d dòng, %d cột"
                  % (ten, ma, len(b), b.shape[1]))
        except (pd.errors.ParserError, ValueError) as loi:
            print("  %-38s mã %d | pandas dừng: %s"
                  % (ten, ma, type(loi).__name__ + ": " + str(loi).splitlines()[0][:40]))
    ma, ra, _ = goi(["--vao", str(lo), "--log-muc", "ERROR"])
    print("  %-38s %d dòng văn bản ở stdout"
          % ("log mức ERROR, stdout chỉ có kết quả", len(ra.strip().splitlines())))


def ma_bam_khong_khop(lo):
    """Mã băm không khớp: chương trình TỰ ném RuntimeError và main không bắt, nên mã 1."""
    lai = ("import importlib.util, sys\n"
           "spec = importlib.util.spec_from_file_location('m69', r'%s')\n"
           "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
           "m.BAM_MONG_DOI = '0' * 64\n"
           "sys.exit(m.main(['--vao', r'%s']))\n" % (CHUONG_TRINH[1], lo))
    kq = subprocess.run([sys.executable, "-c", lai], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, encoding="utf-8",
                        errors="replace")
    return kq.returncode, kq.stdout or "", kq.stderr or ""


def bon_thu_duoc_noi_suy():
    """Tỉ số f-string so với tham số đổi theo THỨ ĐƯỢC NỘI SUY, nên không có tỉ số chung."""
    rieng = logging.getLogger("do_noi_suy")
    rieng.handlers.clear()
    rieng.addHandler(logging.NullHandler())
    rieng.setLevel(logging.ERROR)
    bang = pd.read_csv(RAW / "ch09-lo-moi.csv")
    ba_khoa = {k: bang.iloc[0][k] for k in ("ma_sv", "nganh", "diem_giua_ky")}
    muoi_bon = {k: v for k, v in list(bang.iloc[0].to_dict().items())[1:15]}
    for nhan, v in (("một chuỗi ngắn", "ch09-lo-moi.csv"), ("một số nguyên", 40),
                    ("một từ điển ba khóa", ba_khoa), ("một từ điển 14 khóa", muoi_bon)):
        ds = {}
        for cach in ("tham số", "f-string"):
            do = []
            for _ in range(3):
                t0 = time.perf_counter()
                if cach == "tham số":
                    for _ in range(SO_LAN_DO_LOG):
                        rieng.debug("giá trị: %s", v)
                else:
                    for _ in range(SO_LAN_DO_LOG):
                        rieng.debug(f"giá trị: {v}")
                do.append((time.perf_counter() - t0) * 1000)
            ds[cach] = do
        print("  %-22s tham số %s | f-string %s | tỉ số %.1f tới %.1f"
              % (nhan, " · ".join("%.1f" % x for x in ds["tham số"]),
                 " · ".join("%.1f" % x for x in ds["f-string"]),
                 min(ds["f-string"]) / max(ds["tham số"]),
                 max(ds["f-string"]) / min(ds["tham số"])))


def chi_phi_theo_dang_doi_so():
    """Chi phí nằm ở ĐỐI SỐ NÀO: đối số thứ nhất là hằng chuỗi vẫn không đủ."""
    rieng = logging.getLogger("do_doi_so")
    rieng.handlers.clear()
    rieng.addHandler(logging.NullHandler())
    rieng.setLevel(logging.ERROR)
    bang = pd.read_csv(RAW / "ch09-lo-moi.csv")
    ban_ghi = bang.iloc[0]
    da_co = ban_ghi.to_dict()
    n = 20_000

    def ten_da_co():
        for _ in range(n):
            rieng.debug("bản ghi: %s", da_co)

    def goi_to_dict():
        for _ in range(n):
            rieng.debug("bản ghi: %s", ban_ghi.to_dict())

    moc = None
    for nhan, f in (("tên đã có", ten_da_co), ("gọi .to_dict()", goi_to_dict)):
        ds = []
        for _ in range(3):
            t0 = time.perf_counter()
            f()
            ds.append((time.perf_counter() - t0) * 1000)
        print("  %-16s %d lời gọi ở mức ERROR: %.1f tới %.1f ms"
              % (nhan, n, min(ds), max(ds)))
        if moc is None:
            moc = ds
        else:
            print("  gọi hàm ở đối số thứ hai đắt hơn %.0f tới %.0f lần, dù đối số thứ"
                  " nhất của cả hai đều là hằng chuỗi"
                  % (min(ds) / max(moc), max(ds) / min(moc)))


def vong_doi_file_log(lo):
    """File log mở ở chế độ nối, nên nó dài ra mãi; mức DEBUG làm nó dài rất nhanh."""
    nhat_ky = TAM / "ch09-do-vong-doi.log"
    for muc in ("INFO", "DEBUG"):
        nhat_ky.unlink(missing_ok=True)
        goi(["--vao", str(lo), "--log-muc", muc, "--ra",
             str(GOC_DU_AN / "output" / "ch09-tam.csv"), "--file-log", str(nhat_ky)])
        print("  mức %-5s một lần chạy: %d dòng, %d byte"
              % (muc, len(nhat_ky.read_text(encoding="utf-8").splitlines()),
                 len(nhat_ky.read_bytes())))
    print("  chế độ mở mặc định của logging.FileHandler: %r"
          % logging.FileHandler.__init__.__defaults__[0])
    nhat_ky.unlink(missing_ok=True)


def bang_ma_tra_ve(lo):
    """Sáu tình huống, sáu mã trả về. Hệ thống tự động chỉ đọc được con số này."""
    ca = [("lô đúng hợp đồng", ["--vao", str(lo), "--ra", str(GOC_DU_AN / "output"
                                                             / "ch09-tam.csv")]),
          ("thiếu tham số --vao", []),
          ("giá trị --log-muc không có trong choices",
           ["--vao", str(lo), "--log-muc", "TRACE"]),
          ("file vào không tồn tại", ["--vao", str(RAW / "khong-co-file.csv")]),
          ("file vào chỉ có 1 cột", ["--vao", str(RAW / "ch09-lo-excel-viet.csv")]),
          ("lô sai hợp đồng", ["--vao", str(RAW / "ch09-lo-loi.csv")]),
          ("thư mục ra không ghi được",
           ["--vao", str(lo), "--ra", str(CHA_LA_TEP / "ket-qua.csv")]),
          ("lỗi không lường trước: thư mục file log không tồn tại",
           ["--vao", str(lo), "--file-log", str(CHA_LA_TEP / "nhat-ky.log")]),
          ("mã băm file mô hình không khớp", None)]
    for ten, tham_so in ca:
        ma, _, err = (ma_bam_khong_khop(lo) if tham_so is None else goi(tham_so))
        dong_cuoi = [d for d in err.strip().splitlines() if d][-1] if err.strip() else ""
        print("  %-42s mã %d | %s" % (ten, ma, dong_cuoi[-58:]))


def bat_nham_ngoai_le(lo):
    """Một file đổi tên một cột, ba cách viết, ba mã trả về khác nhau."""
    import joblib
    mo_hinh = joblib.load(GOC_DU_AN / "models" / "ch08-mo-hinh-trien-khai.joblib")
    ho_so = json.loads((GOC_DU_AN / "models" / "ch08-ho-so-trien-khai.json")
                       .read_text(encoding="utf-8"))
    cot = ho_so["cot_vao_theo_thu_tu"]
    sai_ten = TAM / "ch09-lo-sai-ten-cot.csv"
    (pd.read_csv(RAW / "ch09-lo-moi.csv")
     .rename(columns={"diem_giua_ky": "diem_gk"})
     .to_csv(sai_ten, index=False, encoding="utf-8"))
    bang = pd.read_csv(sai_ten)
    try:
        mo_hinh.predict_proba(bang[cot])
        print("  (a) không bắt gì: chạy được")
    except Exception as loi:
        print("  (a) không bắt gì            : %s, mã trả về 1, có thông báo truy vết"
              % type(loi).__name__)
    ket_qua = []
    for i in bang.index:
        try:
            ket_qua.append(mo_hinh.predict_proba(bang.loc[[i], cot])[0, 1])
        except Exception:
            continue
    print("  (b) bắt Exception, chạy tiếp: %d/%d dòng kết quả, mã trả về 0,"
          " không dấu hiệu nào" % (len(ket_qua), len(bang)))
    ma, ra, err = goi(["--vao", str(sai_ten)])
    print("  (c) kiểm hợp đồng trước : mã trả về %d, %d dòng kết quả, log chỉ rõ cột nào"
          % (ma, max(0, len(ra.strip().splitlines()) - 1)))
    print("      %s" % err.strip().splitlines()[-1][-56:])
    sai_ten.unlink(missing_ok=True)

    # Cùng cách viết (b) nhưng trên lô chỉ có 2 ô sai kiểu: mất đúng 2 dòng, mã vẫn 0.
    xau = pd.read_csv(RAW / "ch09-lo-loi.csv")
    giu = 0
    for i in xau.index:
        try:
            mo_hinh.predict_proba(xau.loc[[i], cot])
            giu += 1
        except Exception:
            continue
    print("  (b) trên lô chỉ có 2 ô sai kiểu: giữ %d/%d dòng, mã trả về vẫn 0"
          % (giu, len(xau)))


SO_LAN_DO_LOG = 200_000
NGUON_LENH_LOG = {
    "hằng chuỗi": 'log.debug("đọc %s: %d dòng", TEN, SO_DONG)',
    "f-string": 'log.debug(f"đọc {TEN}: {SO_DONG} dòng")',
    "phép nối chuỗi": 'log.debug("đọc " + TEN)',
    "toán tử %": 'log.debug("đọc %s" % TEN)',
}


def tham_so_so_voi_fstring():
    """Hai cách viết cùng một dòng log, rồi hai cách phát biểu phép tự kiểm."""
    rieng = logging.getLogger("do_log")
    rieng.handlers.clear()
    rieng.addHandler(logging.NullHandler())
    rieng.setLevel(logging.ERROR)
    ten, so_dong = "ch09-lo-moi.csv", 40

    def ban_tham_so():
        for _ in range(SO_LAN_DO_LOG):
            rieng.debug("đọc %s: %d dòng", ten, so_dong)

    def ban_fstring():
        for _ in range(SO_LAN_DO_LOG):
            rieng.debug(f"đọc {ten}: {so_dong} dòng")

    ket = {}
    for nhan, f in (("tham số", ban_tham_so), ("f-string", ban_fstring)):
        ds = []
        for _ in range(5):
            t0 = time.perf_counter()
            f()
            ds.append((time.perf_counter() - t0) * 1000)
        ket[nhan] = (min(ds), max(ds))
        print("  %-9s %d lời gọi ở mức ERROR: %.1f tới %.1f ms"
              % (nhan, SO_LAN_DO_LOG, min(ds), max(ds)))
    print("  f-string chậm %.1f tới %.1f lần"
          % (ket["f-string"][0] / ket["tham số"][1],
             ket["f-string"][1] / ket["tham số"][0]))

    # Phép kiểm theo DANH SÁCH TOÁN TỬ thì hở; theo DẠNG của đối số thứ nhất thì kín.
    mau = re.compile(r"log\.\w+\([^)]*[%+][^)]*\)")
    print("  phép kiểm theo danh sách toán tử, tìm '%' hoặc '+' trong lời gọi:")
    for nhan, src in NGUON_LENH_LOG.items():
        print("    %-16s %s" % (nhan, "đạt" if not mau.search(src) else "trượt"))
    print("  phép kiểm theo dạng của đối số thứ nhất, phải là hằng chuỗi:")
    for nhan, src in NGUON_LENH_LOG.items():
        dau = ast.parse(src).body[0].value.args[0]
        kin = isinstance(dau, ast.Constant) and isinstance(dau.value, str)
        print("    %-16s %-10s %s" % (nhan, type(dau).__name__, "đạt" if kin else "trượt"))


def ro_ri_du_lieu_ca_nhan(lo):
    """Một dòng log mức DEBUG in cả bản ghi là bao nhiêu ô dữ liệu cá nhân."""
    bang = pd.read_csv(lo)
    ban_ghi = bang.iloc[0].to_dict()
    nhat_ky = TAM / "ch09-log-thu.log"
    rieng = logging.getLogger("thu_ro_ri")
    rieng.handlers.clear()
    rieng.setLevel(logging.DEBUG)
    xu_ly = logging.FileHandler(nhat_ky, mode="w", encoding="utf-8")
    rieng.addHandler(xu_ly)
    rieng.debug("bản ghi: %s", json.dumps(ban_ghi, ensure_ascii=False, default=str))
    rieng.debug("mã sinh viên %s, điểm giữa kỳ %s",
                ban_ghi["ma_sv"], ban_ghi["diem_giua_ky"])
    xu_ly.close()
    noi_dung = nhat_ky.read_text(encoding="utf-8")
    print("  một dòng log in cả bản ghi: %d ô dữ liệu, %d byte"
          % (len(ban_ghi), len(noi_dung.splitlines()[0].encode("utf-8"))))
    print("  cả lô %d dòng ở mức DEBUG: %d ô dữ liệu cá nhân nằm trong file log"
          % (len(bang), len(bang) * len(ban_ghi)))
    print("  mã sinh viên có trong file log: %s" % (ban_ghi["ma_sv"] in noi_dung))

    # Cùng cách viết ấy nhưng đã loại cột định danh: phép tìm mã sinh viên ĐẠT, mà dữ
    # liệu cá nhân vẫn nằm trong log. Phép kiểm theo một loại định danh thì luôn hở.
    bo_ma = bang.drop(columns=["ma_sv"])
    rieng.handlers.clear()
    xu_ly2 = logging.FileHandler(nhat_ky, mode="w", encoding="utf-8")
    rieng.addHandler(xu_ly2)
    for i in bo_ma.index:
        rieng.debug("bản ghi: %s", json.dumps(bo_ma.loc[i].to_dict(),
                                              ensure_ascii=False, default=str))
    xu_ly2.close()
    noi_dung = nhat_ky.read_text(encoding="utf-8")
    tim_thay = sum(1 for m in bang["ma_sv"] if m in noi_dung)
    print("  cùng cách viết, đã loại cột ma_sv: %d mã sinh viên tìm thấy nên phép tìm %s,"
          " trong khi log mang %d x %d = %d ô dữ liệu, %d byte"
          % (tim_thay, "đạt" if tim_thay == 0 else "trượt", len(bo_ma), bo_ma.shape[1],
             len(bo_ma) * bo_ma.shape[1], len(noi_dung.encode("utf-8"))))
    nhat_ky.unlink(missing_ok=True)


def main() -> None:
    lo = RAW / "ch09-lo-moi.csv"
    print("-- 9.4.2 log ra đâu: stderr hay stdout --")
    log_ra_dau(lo)
    print("\n-- 9.4.1 bắt ngoại lệ rồi chạy tiếp --")
    bat_nham_ngoai_le(lo)
    print("\n-- 9.4.3 chín tình huống, sáu mã trả về --")
    bang_ma_tra_ve(lo)
    print("\n-- 9.4.2 truyền tham số so với f-string trong lệnh log --")
    tham_so_so_voi_fstring()
    print("\n-- 9.4.2 chi phí nằm ở đối số nào --")
    print("\n-- 9.4.2 bốn thứ được nội suy, bốn tỉ số (bảng tham chiếu của Bài 9.9c) --")
    bon_thu_duoc_noi_suy()
    chi_phi_theo_dang_doi_so()
    print("\n-- 9.4.2 vòng đời file log --")
    vong_doi_file_log(lo)
    print("\n-- 9.4.2 rò rỉ dữ liệu cá nhân qua file log --")
    ro_ri_du_lieu_ca_nhan(lo)


if __name__ == "__main__":
    main()
