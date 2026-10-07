"""Xác định gốc dự án từ vị trí của chính file mã, không phụ thuộc thư mục đang đứng."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GOC_DU_AN = Path(__file__).resolve().parent.parent
DU_LIEU_GOC = GOC_DU_AN / "data" / "raw"
KET_QUA = GOC_DU_AN / "output"

print("Thư mục đang đứng:", Path.cwd().name)
print("Gốc dự án        :", GOC_DU_AN.name)
print("Có file dữ liệu  :", (DU_LIEU_GOC / "sinhvien-truc-tuyen.csv").exists())
KET_QUA.mkdir(exist_ok=True)
print("Thư mục output   :", KET_QUA.is_dir())
