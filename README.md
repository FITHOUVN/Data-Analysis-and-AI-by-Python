# Phân tích dữ liệu và triển khai mô hình AI với Python

Mã nguồn và dữ liệu đi kèm sách chuyên khảo **Phân tích dữ liệu và triển khai mô hình AI với Python**.

Khoa Công nghệ thông tin, Trường Đại học Mở Hà Nội.
Chủ biên: TS. Đinh Tuấn Long. Tham gia biên soạn: ThS. Lê Ngọc An, CN. Nguyễn Đình Dũng, CN. Ngọ Văn Sơn.

> **Sách đang trong quá trình biên soạn.** Mã nguồn của từng chương được bổ sung dần, sau khi chương
> đó đã viết xong và qua rà soát. Hiện có mã của **chương 1 và chương 2**.

## Kho này có gì

```
du-an-ptdl/          dự án mẫu hoàn chỉnh, đúng cây thư mục mà sách hướng dẫn dựng
├── data/raw/        dữ liệu gốc, không bao giờ sửa
├── data/processed/  dữ liệu đã xử lý, xóa đi và tạo lại được
├── code/            chương trình Python, đánh số theo thứ tự chạy
├── models/          mô hình đã huấn luyện
├── output/          hình và bảng kết quả
└── requirements.txt

cong-cu/
└── sinh_du_lieu.py  chương trình sinh bộ dữ liệu mô phỏng của sách
```

Sinh viên nên **tự dựng dự án của mình theo sách**, và chỉ mở kho này để đối chiếu khi kết quả trên
máy khác với kết quả in trong sách.

## Vì sao mã trong sách ngắn hơn mã ở đây

Sách chỉ in **phần mang bài học** của mỗi chương trình, tối đa khoảng 15 dòng. Phần khung lặp lại ở
mọi file (nhập thư viện, đặt lại bảng mã cho đầu ra, dựng đường dẫn, hàm `main`) và phần trang trí
biểu đồ được để ở kho này. Dòng `# ...` trong sách đánh dấu chỗ đã lược.

Mọi đoạn mã in trong sách khớp **từng ký tự** với file tương ứng ở đây.

## Cài đặt

```bash
git clone https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python
cd Data-Analysis-and-AI-by-Python/du-an-ptdl

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS, Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sách được viết và kiểm thử trên **Python 3.11.9**. Phiên bản từng thư viện được ghim trong
`requirements.txt`. Cú pháp các thư viện này đổi theo phiên bản, nên cài đúng bộ đã ghim thì kết quả
trên máy mới khớp với kết quả in trong sách.

| Thư viện | Phiên bản | | Thư viện | Phiên bản |
|---|---|---|---|---|
| pandas | 3.0.3 | | scikit-learn | 1.9.0 |
| numpy | 2.4.6 | | imbalanced-learn | 0.14.2 |
| matplotlib | 3.11.0 | | joblib | 1.5.3 |
| seaborn | 0.13.2 | | openpyxl | 3.1.5 |
| FastAPI | 0.138.1 | | uvicorn | 0.49.0 |
| Pydantic | 2.13.4 | | httpx | 0.28.1 |

## Chạy

Các chương trình trong `code/` đánh số theo thứ tự chạy; chương trình sau dùng kết quả của chương
trình trước.

```bash
python code/00_kiem_tra_moi_truong.py
python code/01_duong_dan.py
# ...
```

Chạy hết một lượt:

```bash
for f in code/*.py; do python "$f"; done                          # macOS, Linux
Get-ChildItem code\*.py | ForEach-Object { python $_.FullName }    # Windows PowerShell
```

## Dữ liệu

`du-an-ptdl/data/raw/sinhvien-truc-tuyen.csv` mô tả kết quả học tập một học phần trực tuyến của
3.000 sinh viên, 3.050 dòng và 19 cột.

**Toàn bộ số liệu là dữ liệu mô phỏng, không phải dữ liệu thật của bất kỳ sinh viên nào.** Dữ liệu
sinh bằng `cong-cu/sinh_du_lieu.py` với hạt ngẫu nhiên cố định, nên chạy lại chương trình đó luôn cho
ra đúng bộ dữ liệu này và mọi con số in trong sách đều tái lập được.

Dữ liệu được gài sẵn bốn loại lỗi mà một bộ dữ liệu thật thường mắc: giá trị thiếu, giá trị ngoại lai,
định dạng không nhất quán và dòng trùng lặp. Chương 2 tìm ra chúng, chương 3 xử lý chúng.

## Lưu ý khi chạy

- Mọi đường dẫn trong mã dựng từ `__file__`, nên chạy được từ bất kỳ thư mục nào.
- Chương trình in tiếng Việt đều đặt `sys.stdout.reconfigure(encoding="utf-8")` ở đầu file. Thiếu dòng
  này, console Windows báo `UnicodeEncodeError`.
- `data/raw/` chỉ đọc. Mọi kết quả xử lý ghi vào `data/processed/`, `models/` hoặc `output/`.

## Phạm vi của kho này

Kho chỉ chứa **mã nguồn và dữ liệu do nhóm biên soạn tự tạo ra** để sinh viên tra cứu: chương trình
Python của từng chương, bộ dữ liệu mô phỏng cùng chương trình sinh ra nó, và các kết quả do chính
những chương trình đó tạo ra.

Kho **không chứa** nội dung sách, và **không chứa** bất kỳ phần nào của các tài liệu tham khảo có bản
quyền. Danh mục tài liệu tham khảo in trong sách; tài liệu nào truy cập mở thì sách ghi kèm địa chỉ
để sinh viên tự tra.

## Giấy phép

Xem `LICENSE`.
