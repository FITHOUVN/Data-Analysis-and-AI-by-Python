"""Mức lạ xuất hiện khi dự đoán dữ liệu mới (mục 4.2.4)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8")
GOC_DU_AN = Path(__file__).resolve().parent.parent
VAO = GOC_DU_AN / "data" / "processed" / "du-lieu-sach.csv"

COT = ["nganh", "thiet_bi_chinh"]
# Một dòng dữ liệu mới có hai mức chưa từng thấy trong dữ liệu huấn luyện
MOI = pd.DataFrame({"nganh": ["Y khoa"], "thiet_bi_chinh": ["Tivi thông minh"]})


def bao_loi(d: pd.DataFrame) -> None:
    """Giá trị mặc định handle_unknown='error' dừng chương trình khi gặp mức lạ."""
    bo = OneHotEncoder(sparse_output=False).fit(d[COT])
    print("handle_unknown mặc định:", bo.handle_unknown)
    try:
        bo.transform(MOI)
    except ValueError as loi:
        print("ValueError:", str(loi).split("\n")[0])


def hai_cach_khong_dung(d: pd.DataFrame) -> None:
    """Hai cách xử lý mức lạ mà không dừng chương trình."""
    bo_qua = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    bo_qua.set_output(transform="pandas")
    ra = bo_qua.fit(d[COT]).transform(MOI)
    print("Tổng theo từng nhóm cột:",
          {c: float(ra[[k for k in ra.columns if k.startswith(c + "_")]].sum(axis=1).iloc[0])
           for c in COT})
    gop = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist",
                        min_frequency=100)
    gop.set_output(transform="pandas")
    ra2 = gop.fit(d[COT]).transform(MOI)
    for cot, hiem in zip(COT, gop.infrequent_categories_):
        print("  mức hiếm của %-15s %s" % (cot, list(hiem) if hiem is not None else "không có"))
    print("Ô khác 0 của dòng mới:", {c: float(v) for c, v in ra2.iloc[0].items() if v})


def main() -> None:
    d = pd.read_csv(VAO, parse_dates=["ngay_dang_ky"])
    print("Mức đã thấy khi khớp:", {c: int(d[c].nunique()) for c in COT})
    print("Dòng dữ liệu mới   :", MOI.iloc[0].to_dict())
    print("\n-- Cách 1: báo lỗi, giá trị mặc định --")
    bao_loi(d)
    print("\n-- Cách 2 và cách 3: ignore và infrequent_if_exist --")
    hai_cach_khong_dung(d)


if __name__ == "__main__":
    main()
