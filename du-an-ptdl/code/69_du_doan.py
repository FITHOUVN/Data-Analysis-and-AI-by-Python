"""Chương trình dự đoán từ dòng lệnh: nạp một file mô hình, trả một file kết quả (Chương 9).

Dùng:
    python code/69_du_doan.py --vao data/raw/ch09-lo-moi.csv --ra output/ch09-ket-qua.csv

Mã trả về (mục 9.4.3): 0 xong · 1 lỗi không lường trước · 2 sai tham số dòng lệnh
(do argparse sinh) · 3 không đọc được file vào · 4 dữ liệu vào không đúng hợp đồng
· 5 không ghi được file ra.
"""
import argparse
import hashlib
import json
import logging
import numpy as np
import sys
from pathlib import Path

import joblib
import pandas as pd

GOC_DU_AN = Path(__file__).resolve().parent.parent
THU_MUC_MO_HINH = GOC_DU_AN / "models"
FILE_MO_HINH = THU_MUC_MO_HINH / "ch08-mo-hinh-trien-khai.joblib"
FILE_HO_SO = THU_MUC_MO_HINH / "ch08-ho-so-trien-khai.json"
FILE_BAM = THU_MUC_MO_HINH / "ch08-ban-ke-sha256.json"
# Bản kê phải đến từ bản phát hành tin cậy, không tự tạo lại trong lúc nạp.
BAN_KE = json.loads(FILE_BAM.read_text(encoding="utf-8"))
BAM_MONG_DOI = BAN_KE[FILE_MO_HINH.name]
BAM_HO_SO = BAN_KE[FILE_HO_SO.name]
OK, LOI_CHUNG, SAI_THAM_SO, KHONG_DOC_DUOC, SAI_HOP_DONG, KHONG_GHI_DUOC = 0, 1, 2, 3, 4, 5
TOI_DA_LOI_IN = 5
COT_DEM = {"so_lan_dang_nhap", "tong_thoi_luong_phut", "so_bai_tap_nop",
           "so_lan_dien_dan"}
# Miền nghiệp vụ, không phải miền quan sát được trên tập huấn luyện: 104 lần đăng nhập
# là hợp lệ dù tập huấn luyện chỉ thấy tới 100, còn tỉ lệ hoàn thành 3,5 thì không hợp lệ
# dù nằm ngoài mọi khoảng đã thấy. Hồ sơ mô hình của Chương 8 chưa mang các cận này.
MIEN_NGHIEP_VU = {"so_lan_dang_nhap": (0, None),
                  "tong_thoi_luong_phut": (0, None),
                  "ty_le_hoan_thanh_video": (0.0, 1.0),
                  "so_bai_tap_nop": (0, None),
                  "diem_tb_bai_tap": (0.0, 10.0),
                  "so_lan_dien_dan": (0, None),
                  "nop_tre_tb_gio": (0, None),
                  "diem_giua_ky": (0.0, 10.0)}
NGUONG_MOI_MO_HINH_DONG_Y = 0.50
# Hai khuôn log cho hai người đọc: bản ra màn hình cho người đang
# theo dõi lần chạy, bản ghi xuống file cho người tra lại sau.
KHUON_MAN_HINH = "%(levelname)-8s %(message)s"
KHUON_TEP = "%(asctime)s %(levelname)-8s %(message)s"
KHUON_GIO = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger("du_doan")
_bo_nho: dict = {}


def dat_log(muc: str, file_log: Path | None) -> None:
    """Log ra stderr, không ra stdout: stdout dành riêng cho kết quả."""
    # Cửa sổ lệnh Windows mặc định cp1252; không đặt lại thì một
    # dòng log mang chữ Việt làm chương trình dừng với
    # UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    man_hinh = logging.StreamHandler(sys.stderr)
    man_hinh.setFormatter(logging.Formatter(KHUON_MAN_HINH))
    xu_ly: list[logging.Handler] = [man_hinh]
    if file_log is not None:
        tep = logging.FileHandler(file_log, encoding="utf-8")
        tep.setFormatter(logging.Formatter(KHUON_TEP, KHUON_GIO))
        xu_ly.append(tep)
    logging.basicConfig(level=getattr(logging, muc), handlers=xu_ly)


def nap_mo_hinh(kiem_bam: bool = True) -> dict:
    """Nạp một lần cho cả tiến trình; lần gọi sau dùng lại đối tượng đã nạp."""
    if "mo_hinh" not in _bo_nho:
        if kiem_bam:
            bam = hashlib.sha256(
                FILE_MO_HINH.read_bytes()).hexdigest()
            if bam != BAM_MONG_DOI:
                thong_bao = "mã băm file mô hình không khớp: %s"
                raise RuntimeError(thong_bao % bam[:16])
            if hashlib.sha256(FILE_HO_SO.read_bytes()).hexdigest() != BAM_HO_SO:
                raise RuntimeError("mã băm hồ sơ không khớp")
        mo_hinh = joblib.load(FILE_MO_HINH)
        ho_so = json.loads(FILE_HO_SO.read_text(encoding="utf-8"))
        if float(mo_hinh.threshold) != float(ho_so["nguong_quyet_dinh"]):
            raise RuntimeError("ngưỡng mô hình không khớp hồ sơ")
        _bo_nho.update(mo_hinh=mo_hinh, ho_so=ho_so)
        tien = _bo_nho["mo_hinh"].estimator.estimator.named_steps["tien"]
        _bo_nho["cot_so"] = list(tien.transformers_[0][2])
        _bo_nho["muc"] = dict(zip(tien.transformers_[1][2],
                                  tien.named_transformers_["pl"][-1].categories_))
        log.info("đã nạp %s (%.1f KB), ngưỡng %.4f, %d cột vào",
                 FILE_MO_HINH.name,
                 FILE_MO_HINH.stat().st_size / 1024,
                 _bo_nho["ho_so"]["nguong_quyet_dinh"],
                 len(_bo_nho["ho_so"]["cot_vao_theo_thu_tu"]))
    return _bo_nho


def doc_bang(duong_dan: Path) -> pd.DataFrame:
    """Đọc file vào. Mọi lỗi đọc đều thành KHONG_DOC_DUOC, không để thông báo truy vết lọt ra."""
    if not duong_dan.is_file():
        raise FileNotFoundError("không thấy file vào: %s" % duong_dan)
    bang = pd.read_csv(duong_dan)
    if bang.shape[1] == 1:
        raise ValueError("file chỉ có 1 cột, dấu phân cách có thể không phải dấu phẩy")
    log.info("đọc %s: %d dòng, %d cột", duong_dan.name, len(bang), bang.shape[1])
    return bang


def kiem_hop_dong(bang: pd.DataFrame, goi: dict) -> tuple[list, list]:
    """Trả về (lỗi, cảnh báo); mỗi phần tử là (dòng trong file, cột, giá trị, lý do)."""
    cot = goi["ho_so"]["cot_vao_theo_thu_tu"]
    loi: list[tuple] = []
    canh_bao: list[tuple] = []
    thieu = [c for c in cot if c not in bang.columns]
    if thieu:
        return [(0, ", ".join(thieu), "", "thiếu cột trong hợp đồng")], canh_bao
    la = [c for c in bang.columns if c not in cot and c != "ma_sv"]
    if la:
        canh_bao.append((0, ", ".join(la), "", "cột ngoài hợp đồng, sẽ bị loại"))
    for c in goi["cot_so"]:
        gt = pd.to_numeric(bang[c], errors="coerce")
        for i in bang.index[gt.isna() & bang[c].notna()]:
            loi.append((i + 2, c, repr(bang.at[i, c]),
                        "không đọc được thành số"))
        huu_han = pd.Series(np.isfinite(gt.to_numpy(dtype=float, na_value=np.nan)),
                             index=bang.index)
        for i in bang.index[gt.notna() & ~huu_han]:
            loi.append((i + 2, c, repr(bang.at[i, c]), "phải là số hữu hạn"))
        if c in COT_DEM:
            for i in bang.index[huu_han & gt.where(huu_han).mod(1).ne(0).fillna(False)]:
                loi.append((i + 2, c, repr(bang.at[i, c]), "phải là số nguyên"))
        duoi, tren = MIEN_NGHIEP_VU[c]
        ly_do_mien = "ngoài miền nghiệp vụ [%s; %s]" % (duoi, tren)
        ngoai = gt < duoi if tren is None else (gt < duoi) | (gt > tren)
        for i in bang.index[ngoai.fillna(False) & huu_han]:
            loi.append((i + 2, c, gt[i], ly_do_mien))
    for c, muc in goi["muc"].items():
        for i in bang.index[~bang[c].isin(list(muc)) & bang[c].notna()]:
            canh_bao.append((i + 2, c, repr(bang.at[i, c]), "mức chưa có trong mô hình"))
    return loi, canh_bao


def ghi_loi(muc, danh_sach: list) -> None:
    """In tối đa TOI_DA_LOI_IN dòng rồi cho biết còn bao nhiêu dòng nữa."""
    for dong, cot, gt, ly_do in danh_sach[:TOI_DA_LOI_IN]:
        vi_tri = "cả file" if dong == 0 else "dòng %d" % dong
        muc("%s %s=%s: %s", vi_tri, cot, gt, ly_do)
    if len(danh_sach) > TOI_DA_LOI_IN:
        muc("còn %d dòng cùng loại, đã lược",
            len(danh_sach) - TOI_DA_LOI_IN)


def muc_uu_tien(p: float, nguong: float) -> str:
    """Ba mức theo hai điểm cắt: ngưỡng đã chọn ở Chương 7 và điểm cắt mặc định 0,5."""
    if p < nguong:
        return "Chưa cần"
    if p < NGUONG_MOI_MO_HINH_DONG_Y:
        return "Cần xem lại"
    return "Liên hệ ngay"


def du_doan(bang: pd.DataFrame, goi: dict) -> pd.DataFrame:
    cot = goi["ho_so"]["cot_vao_theo_thu_tu"]
    nguong = goi["ho_so"]["nguong_quyet_dinh"]
    p = goi["mo_hinh"].predict_proba(bang[cot])[:, 1]
    log.debug("xác suất nhỏ nhất %.4f, lớn nhất %.4f", p.min(), p.max())
    ra = pd.DataFrame({
        "ma_sv": bang["ma_sv"] if "ma_sv" in bang.columns else bang.index + 2,
        "nguy_co_phan_tram": (p * 100).round(1),
        "can_lien_he": ["Có" if x >= nguong else "Không" for x in p],
        "muc_uu_tien": [muc_uu_tien(x, nguong) for x in p],
    }).sort_values("nguy_co_phan_tram", ascending=False)
    log.info("dự đoán xong %d dòng, %d dòng vượt ngưỡng %.4f",
             len(ra), int((p >= nguong).sum()), nguong)
    return ra


def ghi_ket_qua(ra: pd.DataFrame, duong_dan: Path | None) -> None:
    """utf-8-sig cho file để Excel mở đúng phông; utf-8 trần cho stdout để dẫn sang lệnh khác."""
    if duong_dan is None:
        ra.to_csv(sys.stdout, index=False, lineterminator="\n")
        return
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    ra.to_csv(duong_dan, index=False, encoding="utf-8-sig")
    log.info("đã ghi %s, %d dòng", duong_dan.name, len(ra))


def doc_tham_so(dong_lenh=None):
    mo_ta = "Dự đoán nguy cơ không đạt cho một lô sinh viên."
    bo = argparse.ArgumentParser(prog="du_doan", description=mo_ta)
    bo.add_argument("--vao", type=Path, required=True,
                    help="file CSV lô mới")
    bo.add_argument(
        "--ra", type=Path, default=None,
        help="file CSV kết quả; bỏ trống thì in ra stdout")
    bo.add_argument("--log-muc", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    bo.add_argument("--file-log", type=Path, default=None)
    bo.add_argument(
        "--bo-dong-loi", action="store_true",
        help="bỏ các dòng sai hợp đồng rồi dự đoán phần còn lại")
    return bo.parse_args(dong_lenh)


def main(dong_lenh=None) -> int:
    ts = doc_tham_so(dong_lenh)
    dat_log(ts.log_muc, ts.file_log)
    try:
        bang = doc_bang(ts.vao)
    except (OSError, UnicodeDecodeError, ValueError,
            pd.errors.ParserError) as loi:
        log.error("không đọc được file vào: %s", loi)
        return KHONG_DOC_DUOC
    if bang.empty:
        log.error("lô dữ liệu không có dòng để dự đoán")
        return SAI_HOP_DONG
    goi = nap_mo_hinh()
    loi, canh_bao = kiem_hop_dong(bang, goi)
    ghi_loi(log.warning, canh_bao)
    if loi:
        ghi_loi(log.error, loi)
        if not ts.bo_dong_loi or loi[0][0] == 0:
            return SAI_HOP_DONG
        bo = sorted({d - 2 for d, *_ in loi})
        log.warning("bỏ %d dòng sai, dự đoán %d dòng còn lại",
                    len(bo), len(bang) - len(bo))
        bang = bang.drop(index=bo)
        if bang.empty:
            log.error("không còn dòng hợp lệ để dự đoán")
            return SAI_HOP_DONG
    try:
        ghi_ket_qua(du_doan(bang, goi), ts.ra)
    except OSError as loi:
        log.error("không ghi được file ra: %s", loi)
        return KHONG_GHI_DUOC
    return OK


if __name__ == "__main__":
    sys.exit(main())
