"""Kiểm thử tự động cho API dự đoán (Chương 10, mục 10.3.2).

Chạy từ gốc dự án, không cần khởi động máy chủ và không cần mạng:
    python code/76_kiem_thu_api.py

TestClient dựng ứng dụng trong cùng tiến trình nên nó chạy cả phần vòng đời:
dùng câu lệnh "with" thì mô hình được nạp, không dùng thì không.
"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

GOC_DU_AN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC_DU_AN / "code"))
api = importlib.import_module("75_api_du_doan")

LO = pd.read_csv(GOC_DU_AN / "data" / "raw" / "ch09-lo-moi.csv")
HO_SO = json.loads(LO.to_json(orient="records"))
# Bản ghi mốc: sát trên ngưỡng nhất trong lô 40, nên mọi sai lệch đều lộ ra ở nhãn.
MOC = next(h for h in HO_SO if h["ma_sv"] == "SV2601614")
XAC_SUAT_MOC = 0.331912
KHOA_PHAN_HOI_422 = {"ban_ghi", "truong", "ly_do", "loai"}
PHAN_BO_UU_TIEN = {"Chưa cần": 33, "Cần xem lại": 3, "Liên hệ ngay": 4}


def doi(**kw) -> dict:
    return {**MOC, **kw}


def kiem_thu_suc_khoe(c) -> None:
    t = c.get("/suc-khoe").json()
    assert t["trang_thai"] == "ok", t
    assert t["mo_hinh"] == api.TEN_PHIEN_BAN, t["mo_hinh"]
    assert t["so_cot_vao"] == 14 and t["nguong"] == 0.2727, t


def kiem_thu_xac_suat_ban_ghi_moc(c) -> None:
    """Khóa GIÁ TRỊ xác suất, không khóa kết luận "trên ngưỡng"."""
    t = c.post("/du-doan", json=MOC).json()
    assert abs(t["xac_suat"] - XAC_SUAT_MOC) < 5e-7, t["xac_suat"]
    assert t["ngoai_khoang_huan_luyen"] == [], t


def kiem_thu_so_gui_dang_chuoi_khong_doi_ket_qua(c) -> None:
    t = c.post("/du-doan", json=doi(diem_giua_ky=str(MOC["diem_giua_ky"]),
                                    so_lan_dang_nhap=str(MOC["so_lan_dang_nhap"]))).json()
    assert t["xac_suat"] == c.post("/du-doan", json=MOC).json()["xac_suat"], t


def kiem_thu_thieu_truong_tra_422(c) -> None:
    d = {k: v for k, v in MOC.items() if k != "diem_giua_ky"}
    r = c.post("/du-doan", json=d)
    assert r.status_code == 422, r.status_code
    assert r.json()["detail"][0]["truong"] == "diem_giua_ky", r.text


def kiem_thu_ngoai_mien_nghiep_vu_tra_422(c) -> None:
    for sua, phan in [(doi(ty_le_hoan_thanh_video=3.5), "ngoài miền nghiệp vụ"),
                      (doi(tong_thoi_luong_phut=99999), "vượt cận trên nghiệp vụ")]:
        r = c.post("/du-doan", json=sua)
        assert r.status_code == 422, (sua, r.status_code)
        assert phan in r.text, r.text


def kiem_thu_nan_bi_tu_choi(c) -> None:
    """NaN phải gửi bằng thân thô: httpx dùng json.dumps(allow_nan=False) nên nó
    từ chối dựng thân yêu cầu chứa NaN, còn json.dumps mặc định thì ghi ra NaN."""
    assert json.dumps(float("nan")) == "NaN"
    tho = json.dumps(MOC, ensure_ascii=False)
    gia = '"diem_giua_ky": %s' % MOC["diem_giua_ky"]
    assert gia in tho, tho[-120:]
    r = c.post("/du-doan", content=tho.replace(gia, '"diem_giua_ky": NaN').encode("utf-8"),
               headers={"content-type": "application/json"})
    assert r.status_code == 422, r.status_code
    assert "finite" in r.text, r.text


def kiem_thu_phan_hoi_422_khong_chua_du_lieu_goi_len(c) -> None:
    """Phép so TẬP KHÓA là phép kín; phép tìm giá trị chỉ là phép phụ.

    Phép tìm giá trị hở cả hai đầu: rào bằng một ngưỡng độ dài thì nó bỏ lọt mọi
    giá trị ngắn hơn ngưỡng ấy; bỏ rào đi thì nó trượt oan vì giá trị 8 của
    so_bai_tap_nop trùng chữ số 8 trong thông điệp lỗi.
    """
    d = {k: v for k, v in MOC.items() if k != "diem_giua_ky"}
    r = c.post("/du-doan", json=d)
    for e in r.json()["detail"]:           # phép kín: dạng của phần tử detail
        assert set(e) <= KHOA_PHAN_HOI_422, e
    lot = [str(v) for v in d.values() if len(str(v)) >= 4 and str(v) in r.text]
    assert not lot, lot                    # phép phụ: hở với giá trị dưới 4 ký tự


def kiem_thu_truong_la_bi_tu_choi(c) -> None:
    """Trường lạ phải thành 422, kể cả khi nó là tên gõ sai của một trường có mặc định:
    extra="ignore" sẽ bỏ qua im lặng và cờ ma_sv thành null mà không ai biết."""
    for la in ({"diem_cuoi_ky": 3.0}, {"ma_sinh_vien": "SV9"}):
        r = c.post("/du-doan", json=doi(**la))
        assert r.status_code == 422, (la, r.status_code)
        assert r.json()["detail"][0]["loai"] == "extra_forbidden", r.text


def kiem_thu_co_ngoai_suy(c) -> None:
    """Phép khóa giá trị là phép đầu; phép so nhãn ở dòng cuối chỉ là phép phụ.

    Bước 2 Quy trình 10.3 dặn khóa giá trị thay vì khóa kết luận, và ở đây giá trị
    đã được khóa bằng danh sách trường ngoại suy. Dòng so nhãn thêm vào để nói rõ
    một điều của mục 10.2.2: giá trị ngoài khoảng quan sát được NHẬN chứ không bị
    từ chối, nên bản ghi vẫn có nhãn.
    """
    t = c.post("/du-doan", json=doi(so_lan_dang_nhap=104)).json()
    assert t["ngoai_khoang_huan_luyen"] == ["so_lan_dang_nhap"], t
    assert t["can_lien_he"] is False, t


def kiem_thu_lo_40_ban_ghi(c) -> None:
    """Quét TỪNG TRƯỜNG của phản hồi, không chỉ trường can_lien_he: hai phép cuối
    phủ muc_uu_tien, trường mà bản đầu của bộ kiểm thử này bỏ trống nên một lần
    phá mã đi qua được cả mười hai phép."""
    kq = c.post("/du-doan-lo", json={"ho_so": HO_SO}).json()
    assert len(kq) == 40, len(kq)
    assert sum(k["can_lien_he"] for k in kq) == 7, kq
    assert [k["ma_sv"] for k in kq] == [h["ma_sv"] for h in HO_SO]
    dem = Counter(k["muc_uu_tien"] for k in kq)
    assert dem == Counter(PHAN_BO_UU_TIEN), dem
    assert all(k["can_lien_he"] == (k["muc_uu_tien"] != "Chưa cần") for k in kq), kq


def kiem_thu_chuong_trinh_dong_lenh(c) -> None:
    """main của Chương 9 trả về một số nguyên nên gọi được ngay trong một phép kiểm thử,
    không cần tiến trình con. Ba mã trả về của Bảng 9.7 kiểm được bằng ba lời gọi."""
    ra = GOC_DU_AN / "output" / "ch10-kiem-thu-dong-lenh.csv"
    ts = ["--ra", str(ra), "--log-muc", "ERROR"]
    tep = GOC_DU_AN / "data" / "raw"
    assert api.cli.main(["--vao", str(tep / "ch09-lo-moi.csv")] + ts) == 0
    assert api.cli.main(["--vao", str(tep / "ch09-lo-loi.csv")] + ts) == 4
    assert api.cli.main(["--vao", str(tep / "khong-co-file-nay.csv")] + ts) == 3
    assert len(pd.read_csv(ra)) == 40


def kiem_thu_tran_kich_thuoc_lo(c) -> None:
    for n, phan in [(0, "at least 1"), (api.TOI_DA_MOT_LO + 1, "at most 500")]:
        r = c.post("/du-doan-lo", json={"ho_so": (HO_SO * 20)[:n]})
        assert r.status_code == 422, (n, r.status_code)
        assert phan in r.text, r.text
    assert c.post("/du-doan-lo",
                  json={"ho_so": (HO_SO * 20)[:api.TOI_DA_MOT_LO]}).status_code == 200


def kiem_thu_cli_so_huu_han_va_so_nguyen(c) -> None:
    goi = api._goi
    for gia_tri in (float("inf"), float("-inf"), 1.5):
        loi, _ = api.cli.kiem_hop_dong(pd.DataFrame([doi(so_lan_dang_nhap=gia_tri)]), goi)
        assert any(t[1] == "so_lan_dang_nhap" for t in loi), gia_tri
    loi, _ = api.cli.kiem_hop_dong(pd.DataFrame([doi(so_lan_dang_nhap=1)]), goi)
    assert not loi, loi


def kiem_thu_cli_lo_rong(c) -> None:
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as thu_muc:
        p = Path(thu_muc) / "rong.csv"
        LO.iloc[:0].to_csv(p, index=False)
        assert api.cli.main(["--vao", str(p), "--log-muc", "ERROR"]) == 4


def kiem_thu_toan_ven_mo_hinh_va_ho_so(c) -> None:
    from tempfile import TemporaryDirectory
    cli = api.cli
    model_cu, ho_so_cu, bo_nho_cu = cli.FILE_MO_HINH, cli.FILE_HO_SO, cli._bo_nho
    try:
        with TemporaryDirectory() as thu_muc:
            for ten in ("mo_hinh", "ho_so"):
                cli.FILE_MO_HINH = Path(thu_muc) / model_cu.name
                cli.FILE_HO_SO = Path(thu_muc) / ho_so_cu.name
                cli.FILE_MO_HINH.write_bytes(model_cu.read_bytes())
                cli.FILE_HO_SO.write_bytes(ho_so_cu.read_bytes())
                p = cli.FILE_MO_HINH if ten == "mo_hinh" else cli.FILE_HO_SO
                p.write_bytes(p.read_bytes() + b" ")
                cli._bo_nho = {}
                try:
                    cli.nap_mo_hinh()
                except RuntimeError:
                    assert not cli._bo_nho
                else:
                    raise AssertionError("không phát hiện tệp bị thay đổi: " + ten)
    finally:
        cli.FILE_MO_HINH, cli.FILE_HO_SO, cli._bo_nho = model_cu, ho_so_cu, bo_nho_cu


def chay_het() -> int:
    # KHÔNG tắt cảnh báo ở đây. Một lệnh warnings.filterwarnings("ignore") sẽ che
    # InconsistentVersionWarning của joblib.load, tức dấu hiệu duy nhất của việc nạp
    # mô hình bằng một phiên bản scikit-learn khác (mục 8.1.2). Dòng
    # StarletteDeprecationWarning in ra lúc nhập fastapi.testclient là dòng phải để
    # nhìn thấy, không phải dòng phải dập (mục 10.3.2).
    sys.stdout.reconfigure(encoding="utf-8")
    bai = [v for k, v in sorted(globals().items()) if k.startswith("kiem_thu_")]
    truot = []
    with TestClient(api.app) as c:
        for ham in bai:
            try:
                ham(c)
            except Exception as loi:        # ngoại lệ cũng là trượt, không chỉ AssertionError:
                truot.append((ham.__name__, loi))   # ngoại lệ làm dịch vụ sập cũng tính là trượt
                print("TRƯỢT %s: %s: %s" % (ham.__name__, type(loi).__name__, loi))
            else:
                print("ĐẠT   %s" % ham.__name__)
    print("%d đạt, %d trượt trên %d phép kiểm thử" % (len(bai) - len(truot), len(truot), len(bai)))
    return 1 if truot else 0


if __name__ == "__main__":
    sys.exit(chay_het())
