"""Đo các con số của Chương 10 (bảng trong mục 10.1, 10.2 và 10.3).

Chạy từ gốc dự án:
    python code/78_do_api.py lop-kiem      # ba lớp kiểm và ca đi lọt
    python code/78_do_api.py lo-hay-le      # một lô so với nhiều yêu cầu lẻ
    python code/78_do_api.py phan-hoi-422   # kích thước và nội dung phản hồi 422
    python code/78_do_api.py cho-theo-giai-doan   # mức chờ áp cho từng giai đoạn
    python code/78_do_api.py nap khoi-dong  # nạp mô hình khi khởi động
    python code/78_do_api.py nap trong-yeu-cau
Hai lệnh "nap" phải chạy mỗi lệnh trong một tiến trình mới, vì thứ chúng đo là chi phí
chỉ trả một lần cho mỗi tiến trình.
"""
import importlib
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

GOC_DU_AN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC_DU_AN / "code"))
api = importlib.import_module("75_api_du_doan")

LO = pd.read_csv(GOC_DU_AN / "data" / "raw" / "ch09-lo-moi.csv")
HO_SO = json.loads(LO.to_json(orient="records"))
MOC = next(h for h in HO_SO if h["ma_sv"] == "SV2601614")


class HoSoLong(BaseModel):
    """Bản khai LỎNG: float trần, không chặn NaN, không chặn trường lạ. Dùng để đo
    xem lớp khai báo chặt của Mã 10.4 thật sự chặn thêm được những gì."""
    ma_sv: str | None = None
    nganh: str
    gioi_tinh: str
    nhom_tuoi: str
    vua_lam_vua_hoc: str
    hinh_thuc_hoc: str
    thiet_bi_chinh: str
    so_lan_dang_nhap: int
    tong_thoi_luong_phut: int
    ty_le_hoan_thanh_video: float
    so_bai_tap_nop: int
    diem_tb_bai_tap: float
    so_lan_dien_dan: int
    nop_tre_tb_gio: float
    diem_giua_ky: float


def ba_con_so(ten: str, sua, goi: dict) -> None:
    """Ba con số cho một ca: bao nhiêu nhãn đổi, lệch xác suất lớn nhất, số cần liên hệ."""
    cot, nguong = goi["ho_so"]["cot_vao_theo_thu_tu"], goi["ho_so"]["nguong_quyet_dinh"]
    p0 = goi["mo_hinh"].predict_proba(LO[cot])[:, 1]
    p1 = goi["mo_hinh"].predict_proba(sua(LO.copy())[cot])[:, 1]
    print("  %-30s đổi %d/40 nhãn | lệch lớn nhất %.4f | cần liên hệ %d -> %d"
          % (ten, int(((p0 >= nguong) != (p1 >= nguong)).sum()), abs(p0 - p1).max(),
             int((p0 >= nguong).sum()), int((p1 >= nguong).sum())))


def lop_kiem() -> None:
    goi = api.cli.nap_mo_hinh()
    print("Nếu đi lọt tới mô hình thì hậu quả đo được là:")
    ba_con_so("tong_thoi_luong_phut=99999", lambda b: b.assign(tong_thoi_luong_phut=99999), goi)
    ba_con_so("diem_giua_ky=NaN", lambda b: b.assign(diem_giua_ky=float("nan")), goi)
    ba_con_so("so_lan_dang_nhap=104", lambda b: b.assign(so_lan_dang_nhap=104), goi)
    ba_con_so("nganh='Trí tuệ nhân tạo'", lambda b: b.assign(nganh="Trí tuệ nhân tạo"), goi)
    ba_con_so("gioi_tinh viết hoa", lambda b: b.assign(gioi_tinh=b["gioi_tinh"].str.upper()), goi)
    ba_con_so("số gửi dạng chuỗi có dấu chấm",
              lambda b: b.assign(diem_giua_ky=b["diem_giua_ky"].astype(str)), goi)
    print("Lớp nào chặn ca nào:")
    with TestClient(api.app) as c:
        for ten, d in [("tong_thoi_luong_phut=99999", {"tong_thoi_luong_phut": 99999}),
                       ("so_lan_dang_nhap=104", {"so_lan_dang_nhap": 104}),
                       ("nganh='Trí tuệ nhân tạo'", {"nganh": "Trí tuệ nhân tạo"}),
                       ("ty_le_hoan_thanh_video=3.5", {"ty_le_hoan_thanh_video": 3.5}),
                       ("thêm trường diem_cuoi_ky", {"diem_cuoi_ky": 3.0})]:
            r = c.post("/du-doan", json={**MOC, **d})
            print("  %-30s %s %s" % (ten, r.status_code, r.text[:95]))


def lo_hay_le() -> None:
    with TestClient(api.app) as c:
        c.post("/du-doan", json=MOC)
        for lan in range(3):
            t0 = time.perf_counter()
            assert c.post("/du-doan-lo", json={"ho_so": HO_SO}).status_code == 200
            t_lo = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            for h in HO_SO:
                assert c.post("/du-doan", json=h).status_code == 200
            t_le = (time.perf_counter() - t0) * 1000
            # Tỉ số tính từ GIÁ TRỊ GỐC, không từ hai con số đã làm tròn in ở cùng dòng, nên
            # chia lại hai con số ấy có thể ra khác: 731,5 chia 14,6 cho 50,10 trong khi tỉ
            # số thật là 50,31. Bảng in trong sách nói rõ điều đó ở chú thích bảng.
            print("  lần %d: một lô 40 bản ghi %.1f ms | 40 yêu cầu lẻ %.1f ms | tỉ số %.1f"
                  % (lan + 1, t_lo, t_le, t_le / t_lo))
        than = json.dumps({"ho_so": HO_SO}).encode()
        print("  thân yêu cầu lô 40: %d byte, %d byte mỗi bản ghi; trần %d bản ghi là %.0f KB"
              % (len(than), len(than) // 40, api.TOI_DA_MOT_LO,
                 len(than) / 40 * api.TOI_DA_MOT_LO / 1024))


def phan_hoi_422() -> None:
    phu = FastAPI()

    @phu.post("/du-doan")
    def mot(ho_so: HoSoLong) -> dict:
        return {"diem_giua_ky_nhan_duoc": ho_so.diem_giua_ky}

    thieu = {k: v for k, v in MOC.items() if k != "diem_giua_ky"}
    tho = json.dumps(MOC, ensure_ascii=False)
    nan_tho = tho.replace('"diem_giua_ky": %s' % MOC["diem_giua_ky"], '"diem_giua_ky": NaN')
    assert nan_tho != tho
    tieu_de = {"content-type": "application/json"}
    with TestClient(phu, raise_server_exceptions=False) as c:
        r = c.post("/du-doan", json=thieu)
        e = r.json()["detail"][0]
        print("  bản khai lỏng, thiếu 1 trường : %s, %d byte, khóa %s, input mang %d giá trị"
              % (r.status_code, len(r.content), sorted(e), len(e["input"])))
        print("  bản khai lỏng, thân chứa NaN  : %s %s"
              % (c.post("/du-doan", content=nan_tho.encode("utf-8"), headers=tieu_de).status_code,
                 c.post("/du-doan", content=nan_tho.encode("utf-8"), headers=tieu_de).text[:60]))
        ca_hai = json.dumps({**thieu, "nop_tre_tb_gio": float("nan")})
        print("  bản khai lỏng, thiếu 1 trường VÀ có NaN ở trường khác: %s"
              % c.post("/du-doan", content=ca_hai.encode(), headers=tieu_de).status_code)
    with TestClient(api.app) as c:
        r = c.post("/du-doan", json=thieu)
        print("  bản khai chặt, thiếu 1 trường : %s, %d byte, %s"
              % (r.status_code, len(r.content), r.text))
        print("  bản khai chặt, thân chứa NaN  : %s %s"
              % (c.post("/du-doan", content=nan_tho.encode("utf-8"), headers=tieu_de).status_code,
                 c.post("/du-doan", content=nan_tho.encode("utf-8"), headers=tieu_de).text[:85]))
        for ten, goi in [("GET /du-doan", lambda: c.get("/du-doan")),
                         ("POST /du-doann", lambda: c.post("/du-doann", json=MOC)),
                         ("POST /du-doan thân rỗng", lambda: c.post("/du-doan")),
                         ("POST /du-doan thân không phải JSON",
                          lambda: c.post("/du-doan", content=b"xin chao", headers=tieu_de))]:
            r = goi()
            print("  %-36s %s %s" % (ten, r.status_code, r.text[:85]))


def nap(cach: str) -> None:
    if cach == "khoi-dong":
        ung_dung = api.app
    else:
        ung_dung = FastAPI()

        @ung_dung.post("/du-doan")
        def mot(ho_so: api.HoSoSinhVien) -> dict:
            api._goi.clear()
            api.cli._bo_nho.clear()
            api._goi.update(api.cli.nap_mo_hinh())
            return api.du_doan_api([ho_so])[0].model_dump()

    t0 = time.perf_counter()
    c = TestClient(ung_dung)
    c.__enter__()
    t_khoi_dong = (time.perf_counter() - t0) * 1000
    t = []
    for _ in range(6):
        t0 = time.perf_counter()
        assert c.post("/du-doan", json=MOC).status_code == 200
        t.append((time.perf_counter() - t0) * 1000)
    c.__exit__(None, None, None)
    print("  %-13s khởi động %.0f ms | yêu cầu đầu %.1f ms | năm yêu cầu sau %.1f tới %.1f ms"
          % (cach, t_khoi_dong, t[0], min(t[1:]), max(t[1:])))


def _mo_cong() -> tuple[socket.socket, int]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def _tra_loi(c: socket.socket, than: bytes) -> None:
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(than) + than)


def _may_chu_nho_giot(s: socket.socket, than: bytes, nghi: float) -> None:
    """Trả lời đúng một lời gọi, gửi từng byte một, cách nhau nghi giây."""
    c, _ = s.accept()
    c.recv(65536)
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(than))
    for b in than:
        time.sleep(nghi)
        c.sendall(bytes([b]))
    c.close()
    s.close()


def _may_chu_ngu(s: socket.socket, ngu: float) -> None:
    """Trả lời đúng một lời gọi, nhưng im lặng ngu giây trước khi gửi gì."""
    c, _ = s.accept()
    c.recv(65536)
    time.sleep(ngu)
    _tra_loi(c, b"{}")
    c.close()
    s.close()


def cho_theo_giai_doan() -> None:
    """Mức chờ của httpx áp cho TỪNG GIAI ĐOẠN, không cho tổng một lời gọi.

    Hai phản ví dụ, mỗi cái dựng bằng một máy chủ socket thô vì không máy chủ thật
    nào chịu chậm theo yêu cầu: (1) gửi nhỏ giọt, khe giữa hai byte luôn dưới mức
    chờ, nên mức chờ đọc được tính lại mỗi lần có byte về và cả lời gọi vượt xa mức
    chờ; (2) timeout=None truyền ở lời gọi đè mức chờ của Client, nên một lời gọi
    "có chữ timeout" và "đi qua Client đã khai timeout" vẫn không có mức chờ nào.
    """
    than, nghi, cho = b'{"x":123456789}', 1.0, 3.0
    s, cong = _mo_cong()
    threading.Thread(target=_may_chu_nho_giot, args=(s, than, nghi),
                     daemon=True).start()
    t0 = time.perf_counter()
    r = httpx.get("http://127.0.0.1:%d/" % cong, timeout=cho)
    print("  nhỏ giọt %d byte, khe %.1f s (dưới mức chờ %.1f s): mã %d sau %.1f s"
          % (len(than), nghi, cho, r.status_code, time.perf_counter() - t0))

    ngu = 8.0
    s2, cong2 = _mo_cong()
    threading.Thread(target=_may_chu_ngu, args=(s2, ngu), daemon=True).start()
    with httpx.Client(base_url="http://127.0.0.1:%d" % cong2, timeout=cho) as khach:
        t0 = time.perf_counter()
        r = khach.get("/", timeout=None)
        print("  khach.timeout = %s, lời gọi truyền timeout=None: mã %d sau %.1f s"
              % (khach.timeout, r.status_code, time.perf_counter() - t0))
    print("  httpx.Timeout(None) theo từng giai đoạn: %s"
          % {k: getattr(httpx.Timeout(None), k) for k in
             ("connect", "read", "write", "pool")})


if __name__ == "__main__":
    # KHÔNG tắt cảnh báo: cùng lệnh tắt sẽ che InconsistentVersionWarning của
    # joblib.load, dấu hiệu duy nhất của việc nạp mô hình bằng một phiên bản
    # scikit-learn khác (mục 8.1.2 và mục 10.3.2).
    sys.stdout.reconfigure(encoding="utf-8")
    phan = sys.argv[1] if len(sys.argv) > 1 else "tat-ca"
    if phan == "nap":
        nap(sys.argv[2] if len(sys.argv) > 2 else "khoi-dong")
    else:
        for ten, ham in [("lop-kiem", lop_kiem), ("lo-hay-le", lo_hay_le),
                         ("phan-hoi-422", phan_hoi_422),
                         ("cho-theo-giai-doan", cho_theo_giai_doan)]:
            if phan in ("tat-ca", ten):
                print("=== %s ===" % ten)
                ham()
