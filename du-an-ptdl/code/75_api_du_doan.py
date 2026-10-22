"""API dự đoán: bọc đúng chương trình dòng lệnh của Chương 9 (Chương 10, mục 10.2).

Chạy từ gốc dự án:
    uvicorn --app-dir code 75_api_du_doan:app --port 8000
Tài liệu tự sinh: http://127.0.0.1:8000/docs

Mô hình, hợp đồng dữ liệu vào, mã băm và ngưỡng 0,2727 đều lấy từ 69_du_doan.py,
không khai lại ở đây: một ngưỡng viết ở hai chỗ là một ngưỡng sẽ lệch nhau.
"""
import importlib.util
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_CLI = GOC_DU_AN / "code" / "69_du_doan.py"
TOI_DA_MOT_LO = 500          # chặn ở lớp khai báo, trước khi dựng DataFrame
# Cận trên nghiệp vụ mà MIEN_NGHIEP_VU của Chương 9 để ngỏ. Căn cứ: một học kỳ 15 tuần,
# không ai học quá 8 giờ mỗi ngày suốt 15 tuần (15*7*8*60 = 50.400 phút) và không ai
# nộp muộn hơn cả học kỳ (15*7*24 = 2.520 giờ). Chỗ đúng của hai cận này là hồ sơ mô hình.
CAN_TREN_NGHIEP_VU = {"tong_thoi_luong_phut": 50400, "nop_tre_tb_gio": 2520}
# Khoảng quan sát trên 2.409 dòng huấn luyện (mục 8.2.1). KHÔNG dùng để từ chối yêu cầu:
# chỉ dùng để đánh dấu trong phản hồi rằng dự đoán đang ngoại suy.
KHOANG_HUAN_LUYEN = {"so_lan_dang_nhap": (1, 100), "tong_thoi_luong_phut": (20, 2198),
                     "ty_le_hoan_thanh_video": (0.0, 1.0), "so_bai_tap_nop": (0, 10),
                     "diem_tb_bai_tap": (0.0, 10.0), "so_lan_dien_dan": (0, 14),
                     "nop_tre_tb_gio": (0.0, 644.0), "diem_giua_ky": (0.0, 10.0)}
log = logging.getLogger("api_du_doan")


def nap_mo_dun(duong_dan: Path, ten: str):
    """Nạp một file .py mà lệnh import không dùng được vì tên bắt đầu bằng chữ số."""
    dac_ta = importlib.util.spec_from_file_location(ten, duong_dan)
    mo_dun = importlib.util.module_from_spec(dac_ta)
    sys.modules[ten] = mo_dun
    dac_ta.loader.exec_module(mo_dun)
    return mo_dun


cli = nap_mo_dun(FILE_CLI, "du_doan_cli")
TEN_PHIEN_BAN = "%s@%s" % (cli.FILE_MO_HINH.name, cli.BAM_MONG_DOI[:8])
_goi: dict = {}


@asynccontextmanager
async def vong_doi(app: FastAPI):
    """Nạp mô hình một lần khi tiến trình khởi động, giải phóng khi tắt."""
    _goi.update(cli.nap_mo_hinh())
    log.info("nạp khi khởi động: %s", TEN_PHIEN_BAN)
    yield
    _goi.clear()


app = FastAPI(title="API dự đoán nguy cơ không đạt", version="1.0", lifespan=vong_doi)


class HoSoSinhVien(BaseModel):
    """14 trường của hợp đồng dữ liệu vào (mục 8.2.1), cộng một mã sinh viên tùy chọn."""
    model_config = ConfigDict(extra="forbid")

    ma_sv: str | None = None
    nganh: str
    gioi_tinh: str
    nhom_tuoi: str
    vua_lam_vua_hoc: str
    hinh_thuc_hoc: str
    thiet_bi_chinh: str
    so_lan_dang_nhap: int
    tong_thoi_luong_phut: int
    ty_le_hoan_thanh_video: float = Field(allow_inf_nan=False)
    so_bai_tap_nop: int
    diem_tb_bai_tap: float = Field(allow_inf_nan=False)
    so_lan_dien_dan: int
    nop_tre_tb_gio: float = Field(allow_inf_nan=False)
    diem_giua_ky: float = Field(allow_inf_nan=False)


class YeuCauLo(BaseModel):
    ho_so: list[HoSoSinhVien] = Field(min_length=1, max_length=TOI_DA_MOT_LO)


class KetQua(BaseModel):
    ma_sv: str | None
    xac_suat: float
    can_lien_he: bool
    muc_uu_tien: str
    ngoai_khoang_huan_luyen: list[str]
    nguong: float
    mo_hinh: str


def kiem_can_tren(bang: pd.DataFrame) -> list[dict]:
    """Cận trên nghiệp vụ: giá trị không thể có thật thì từ chối cả yêu cầu."""
    return [{"ban_ghi": int(i), "truong": c, "ly_do": "vượt cận trên nghiệp vụ %s" % tren}
            for c, tren in CAN_TREN_NGHIEP_VU.items() if c in bang.columns
            for i in bang.index[bang[c] > tren]]


def ngoai_khoang(h: "HoSoSinhVien") -> list[str]:
    """Các trường nằm ngoài khoảng quan sát khi huấn luyện: dự đoán đang ngoại suy."""
    return [c for c, (duoi, tren) in KHOANG_HUAN_LUYEN.items()
            if not duoi <= getattr(h, c) <= tren]


def du_doan_api(ho_so: list[HoSoSinhVien]) -> list[KetQua]:
    """Dựng bảng, kiểm hợp đồng bằng mã của Chương 9, rồi dự đoán cả lô một lần."""
    bang = pd.DataFrame([h.model_dump() for h in ho_so])
    loi, canh_bao = cli.kiem_hop_dong(bang, _goi)
    for dong, cot, _, ly_do in canh_bao:
        log.warning("bản ghi %s, trường %s: %s", dong - 2, cot, ly_do)
    # Không đưa giá trị người gọi gửi lên vào thông điệp: nó là dữ liệu cá nhân.
    # dong == 0 là quy ước "cả file" của Chương 9; trong JSON không có số dòng nên nó thành null.
    chi_tiet = [{"ban_ghi": None if dong == 0 else dong - 2, "truong": cot, "ly_do": ly_do}
                for dong, cot, _, ly_do in loi] + kiem_can_tren(bang)
    if chi_tiet:
        raise HTTPException(status_code=422, detail=chi_tiet)
    cot = _goi["ho_so"]["cot_vao_theo_thu_tu"]
    nguong = _goi["ho_so"]["nguong_quyet_dinh"]
    p = _goi["mo_hinh"].predict_proba(bang[cot])[:, 1]
    return [KetQua(ma_sv=h.ma_sv, xac_suat=round(float(x), 6),
                   can_lien_he=bool(x >= nguong),
                   muc_uu_tien=cli.muc_uu_tien(float(x), nguong),
                   ngoai_khoang_huan_luyen=ngoai_khoang(h),
                   nguong=round(nguong, 4), mo_hinh=TEN_PHIEN_BAN)
            for h, x in zip(ho_so, p)]


@app.get("/suc-khoe")
def suc_khoe() -> dict:
    """Kiểm tra dịch vụ: có mô hình trong bộ nhớ hay không, và là mô hình nào."""
    da_nap = "mo_hinh" in _goi
    return {"trang_thai": "ok" if da_nap else "chua-nap",
            "mo_hinh": TEN_PHIEN_BAN,
            "nguong": round(_goi["ho_so"]["nguong_quyet_dinh"], 4) if da_nap else None,
            "so_cot_vao": len(_goi["ho_so"]["cot_vao_theo_thu_tu"]) if da_nap else None,
            "toi_da_mot_lo": TOI_DA_MOT_LO}


@app.post("/du-doan", response_model=KetQua)
def du_doan_mot(ho_so: HoSoSinhVien) -> KetQua:
    return du_doan_api([ho_so])[0]


@app.post("/du-doan-lo", response_model=list[KetQua])
def du_doan_lo(yeu_cau: YeuCauLo) -> list[KetQua]:
    return du_doan_api(yeu_cau.ho_so)


@app.exception_handler(RequestValidationError)
async def gon_loi_422(yeu_cau: Request, loi: RequestValidationError):
    """Bỏ khóa "input" khỏi phản hồi 422: mặc định nó mang nguyên dữ liệu người gọi gửi."""
    gon = [{"truong": ".".join(str(k) for k in e["loc"][1:]) or "thân yêu cầu",
            "ly_do": e["msg"], "loai": e["type"]} for e in loi.errors()]
    log.warning("422: %d trường không hợp lệ: %s", len(gon),
                ", ".join(e["truong"] for e in gon))
    return JSONResponse(status_code=422, content={"detail": gon})
