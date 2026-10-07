# Dự án mẫu đi kèm sách

**Phân tích dữ liệu và triển khai mô hình AI với Python**
Khoa Công nghệ thông tin, Trường Đại học Mở Hà Nội

Thư mục này là **bản hoàn chỉnh của dự án mà sách hướng dẫn dựng từng bước**. Sinh viên nên tự dựng
dự án của mình theo sách, và chỉ mở thư mục này để đối chiếu khi kết quả trên máy khác với kết quả
in trong sách.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS, Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sách được viết và kiểm thử trên **Python 3.11.9**. Phiên bản của từng thư viện được ghim trong
`requirements.txt`; cú pháp các thư viện này đổi theo phiên bản, nên cài đúng bộ đã ghim thì kết quả
trên máy mới khớp với kết quả in trong sách.

## Cây thư mục

| Thư mục | Nội dung |
|---|---|
| `data/raw/` | dữ liệu gốc, **không bao giờ sửa** |
| `data/processed/` | dữ liệu đã xử lý, xóa đi và tạo lại được bằng cách chạy lại mã |
| `code/` | chương trình Python, đánh số theo thứ tự chạy |
| `models/` | mô hình đã huấn luyện |
| `output/` | hình và bảng kết quả |

## Dữ liệu

`data/raw/sinhvien-truc-tuyen.csv` – 3.050 dòng, 19 cột, mô tả kết quả học tập một học phần trực
tuyến. Bản Excel cùng nội dung: `sinhvien-truc-tuyen.xlsx`. Bảng tra vùng miền: `tinh-thanh.csv`.

**Toàn bộ số liệu là dữ liệu mô phỏng, không phải dữ liệu thật của bất kỳ sinh viên nào.** Dữ liệu
được sinh bằng `../cong-cu/sinh_du_lieu.py` với hạt ngẫu nhiên cố định, nên chạy lại chương trình đó
luôn cho ra đúng bộ dữ liệu này và mọi con số in trong sách đều tái lập được.

Dữ liệu đã được gài sẵn bốn loại lỗi mà một bộ dữ liệu thật thường mắc: giá trị thiếu, giá trị ngoại
lai, định dạng không nhất quán và dòng trùng lặp. Chương 2 tìm ra chúng, chương 3 xử lý chúng.

## Thứ tự chạy

Các chương trình trong `code/` đánh số theo thứ tự chạy. Chương trình sau dùng kết quả của chương
trình trước, nên chạy theo đúng thứ tự số.

Cách gọn nhất là chạy hết một lượt theo thứ tự tên file:

```bash
for f in code/*.py; do python "$f"; done     # macOS, Linux
Get-ChildItem code\*.py | ForEach-Object { python $_.FullName }   # Windows PowerShell
```

Chương 1 dùng bốn file `00_` tới `03_`; chương 2 dùng chín file `04_` tới `12_`; chương 3 dùng
mười hai file `13_` tới `24_`; chương 4 dùng mười một file `25_` tới `35_`:

| File | Chương, mục | Sinh ra gì |
|---|---|---|
| `00_kiem_tra_moi_truong.py` | 1.5.3 | – |
| `01_duong_dan.py` | 1.6.3 | thư mục `output/` |
| `02_chao_du_an.py` | 1.7.1 | – |
| `03_doc_du_lieu.py` | 1.7.2 | – |
| `04_cau_truc.py` | 2.2.1 | – |
| `05_doc_csv.py` | 2.2.2 | `data/processed/ma-lop.csv`, `sep.csv` |
| `06_doc_excel.py` | 2.2.3 | – |
| `07_kham_pha.py` | 2.3.1, 2.3.3, 2.3.4 | – |
| `08_thong_ke.py` | 2.4.1, 2.4.3 | – |
| `09_ve_bieu_do.py` | 2.5.1 tới 2.5.3 | 5 file PNG trong `output/` |
| `10_thieu_va_canh.py` | 2.6.1 | – |
| `11_ngoai_lai_va_nhan.py` | 2.6.2 | – |
| `12_danh_sach_van_de.py` | 2.6.3 | `data/processed/ch02-bao-cao-kham-pha.csv` |
| `13_thieu_chan_doan.py` | 3.1.1 | – |
| `14_thieu_xoa_hay_dien.py` | 3.1.2 | – |
| `15_thieu_dien_gia_tri.py` | 3.1.3 | – |
| `16_thieu_he_qua.py` | 3.1.4 | – |
| `17_ngoai_lai_phan_biet.py` | 3.2.1 | – |
| `18_ngoai_lai_nguong.py` | 3.2.2 | – |
| `19_ngoai_lai_bon_cach.py` | 3.2.3 | `output/ch03-box-nop-tre.png` |
| `20_gia_tri_canh.py` | 3.2.4 | – |
| `21_chuan_hoa_nhan.py` | 3.3.1 | – |
| `22_doi_kieu_so_va_ngay.py` | 3.3.2, 3.3.3 | – |
| `23_dong_trung.py` | 3.4.1, 3.4.2 | `output/ch03-trung-ma-sv.csv` |
| `24_lam_sach.py` | 3.4.3 | `data/processed/du-lieu-sach.csv` |
| `25_thang_do.py` | 4.1.1 | – |
| `26_hai_bo_chuan_hoa.py` | 4.1.2 | – |
| `27_chon_bo_chuan_hoa.py` | 4.1.3 | – |
| `28_khop_tren_tap_huan_luyen.py` | 4.1.4 | – |
| `29_ma_hoa_thu_bac_va_mot_nong.py` | 4.2.1, 4.2.2 | – |
| `30_bien_nhieu_muc.py` | 4.2.3 | – |
| `31_muc_la.py` | 4.2.4 | – |
| `32_chia_tap.py` | 4.3.1, 4.3.2 | – |
| `33_ro_ri_du_lieu.py` | 4.3.3 | – |
| `34_dac_trung_khong_duoc_dung.py` | 4.3.4 | – |
| `35_luu_du_lieu_da_xu_ly.py` | 4.4.1 | 6 file trong `data/processed/`, `models/bo-tien-xu-ly.joblib` |

## Lưu ý

- Mọi đường dẫn trong mã được dựng từ `__file__`, nên chạy được từ bất kỳ thư mục nào.
- Chương trình in tiếng Việt đều đặt `sys.stdout.reconfigure(encoding="utf-8")` ở đầu file. Thiếu dòng
  này, console Windows báo `UnicodeEncodeError`.
- `data/raw/` chỉ đọc. Mọi kết quả xử lý ghi vào `data/processed/`, `models/` hoặc `output/`.
