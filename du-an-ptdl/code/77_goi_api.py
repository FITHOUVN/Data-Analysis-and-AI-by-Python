"""Gọi API dự đoán từ một chương trình Python khác (Chương 10, mục 10.4).

Chạy API ở một cửa sổ lệnh khác trước:
    uvicorn --app-dir code 75_api_du_doan:app --port 8000
rồi:
    python code/77_goi_api.py --vao data/raw/ch09-lo-moi.csv --ra output/ch10-ket-qua.csv
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

GOC_DU_AN = Path(__file__).resolve().parent.parent
DIA_CHI = "http://127.0.0.1:8000"
# Mốc là lần chậm nhất quan sát được: 197,5 ms cho lô 40 bản ghi ở lần gọi đầu sau một
# quãng nghỉ (Bảng 10.6). Đây KHÔNG phải trần cho tổng thời gian một lời gọi: mức chờ của
# httpx áp cho từng giai đoạn, xem mục 10.4.1.
CHO_TOI_DA = 3.0        # giây
SO_LAN_THU = 3
GIAN_CACH = 0.5         # giây, nhân đôi sau mỗi lần thất bại
OK, KHONG_GOI_DUOC, API_TU_CHOI = 0, 6, 7
log = logging.getLogger("goi_api")


def goi_mot_lan(khach: httpx.Client, ho_so: list[dict]) -> httpx.Response:
    return khach.post("/du-doan-lo", json={"ho_so": ho_so})


def goi_co_thu_lai(ho_so: list[dict], dia_chi: str = DIA_CHI) -> list[dict]:
    """Thử lại chỉ với lỗi mạng và lỗi phía dịch vụ; 4xx thì thử lại cũng vô ích.

    Thử lại an toàn vì điểm cuối dự đoán không đổi trạng thái nào trên dịch vụ:
    gọi hai lần cho đúng một kết quả, không sinh bản ghi thừa.
    """
    cho = GIAN_CACH
    with httpx.Client(base_url=dia_chi, timeout=CHO_TOI_DA) as khach:
        for lan in range(1, SO_LAN_THU + 1):
            try:
                tl = goi_mot_lan(khach, ho_so)
            except (httpx.ConnectError, httpx.TimeoutException) as loi:
                log.warning("lần %d: %s: %s", lan, type(loi).__name__, loi)
            else:
                if tl.status_code < 500:
                    tl.raise_for_status()
                    return tl.json()
                log.warning("lần %d: dịch vụ trả %d", lan, tl.status_code)
            if lan < SO_LAN_THU:
                time.sleep(cho)
                cho *= 2
    raise httpx.ConnectError("không gọi được %s sau %d lần" % (dia_chi, SO_LAN_THU))


def main(dong_lenh=None) -> int:
    ts = doc_tham_so(dong_lenh)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    bang = pd.read_csv(ts.vao)
    ho_so = json.loads(bang.to_json(orient="records"))
    t0 = time.perf_counter()
    try:
        kq = goi_co_thu_lai(ho_so, ts.dia_chi)
    except httpx.HTTPStatusError as loi:
        log.error("dịch vụ từ chối %d: %s", loi.response.status_code, loi.response.text[:300])
        return API_TU_CHOI
    except (httpx.ConnectError, httpx.TimeoutException) as loi:
        log.error("thất bại: %s", loi)
        return KHONG_GOI_DUOC
    log.info("%d bản ghi, %.0f ms, mô hình %s", len(kq),
             (time.perf_counter() - t0) * 1000, kq[0]["mo_hinh"])
    ra = pd.DataFrame(kq)
    ngoai = ra["ngoai_khoang_huan_luyen"].map(len) > 0
    if ngoai.any():
        log.warning("%d bản ghi ngoại suy ngoài khoảng huấn luyện", int(ngoai.sum()))
    ts.ra.parent.mkdir(parents=True, exist_ok=True)
    ra.to_csv(ts.ra, index=False, encoding="utf-8-sig")
    log.info("đã ghi %s, %d dòng cần liên hệ", ts.ra.name, int(ra["can_lien_he"].sum()))
    return OK


def doc_tham_so(dong_lenh=None):
    bo = argparse.ArgumentParser(prog="goi_api", description="Gọi API dự đoán cho một lô.")
    bo.add_argument("--vao", type=Path, required=True)
    bo.add_argument("--ra", type=Path, default=GOC_DU_AN / "output" / "ch10-ket-qua.csv")
    bo.add_argument("--dia-chi", default=DIA_CHI, help="địa chỉ dịch vụ dự đoán")
    return bo.parse_args(dong_lenh)


if __name__ == "__main__":
    sys.exit(main())
