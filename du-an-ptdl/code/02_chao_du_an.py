"""In thông tin môi trường và kiểm tra sự tồn tại của file dữ liệu."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_DU_LIEU = GOC_DU_AN / "data" / "raw" / "sinhvien-truc-tuyen.csv"


def main() -> None:
    print("Dự án           :", GOC_DU_AN.name)
    print("Phiên bản Python:", sys.version.split()[0])
    print("Có file dữ liệu :", FILE_DU_LIEU.exists())


if __name__ == "__main__":
    main()
