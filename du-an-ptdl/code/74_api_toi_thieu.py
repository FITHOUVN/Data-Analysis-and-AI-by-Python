"""Ứng dụng FastAPI tối thiểu: ba điểm cuối, chưa có mô hình (Chương 10, mục 10.1.3).

Chạy từ gốc dự án:
    uvicorn --app-dir code 74_api_toi_thieu:app --port 8000 --reload
Rồi mở http://127.0.0.1:8000/docs

Tên file bắt đầu bằng chữ số nên câu lệnh "import 74_api_toi_thieu" là lỗi cú pháp,
nhưng uvicorn nạp mô đun bằng importlib nên chuỗi "74_api_toi_thieu:app" vẫn dùng được.
"""
import platform

import fastapi
from fastapi import FastAPI

app = FastAPI(title="Thử FastAPI", version="0.1")


@app.get("/")
def goc() -> dict:
    """Điểm cuối đơn giản nhất: không tham số, trả về một từ điển."""
    return {"loi_chao": "API đang chạy"}


@app.get("/phien-ban")
def phien_ban() -> dict:
    """Phiên bản của môi trường đang chạy, dùng để đối chiếu khi kết quả lệch."""
    return {"python": platform.python_version(), "fastapi": fastapi.__version__}


@app.get("/binh-phuong/{n}")
def binh_phuong(n: int) -> dict:
    """n khai kiểu int nên FastAPI tự đổi chuỗi trong đường dẫn thành số nguyên,
    và tự trả mã 422 khi đổi không được."""
    return {"n": n, "binh_phuong": n * n}
