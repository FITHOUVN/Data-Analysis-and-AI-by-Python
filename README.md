# Phân tích dữ liệu và triển khai mô hình AI với Python

Mã nguồn và dữ liệu đi kèm sách chuyên khảo **Phân tích dữ liệu và triển khai mô hình AI với Python**.

Khoa Công nghệ thông tin, Trường Đại học Mở Hà Nội.
Chủ biên: TS. Đinh Tuấn Long. Tham gia biên soạn: ThS. Lê Ngọc An, CN. Nguyễn Đình Dũng, CN. Ngọ Văn Sơn.

> **Sách đang trong quá trình biên soạn.** Mã nguồn của từng chương được bổ sung dần, sau khi chương
> đó đã viết xong và qua rà soát. Hiện có mã của **cả mười một chương**, tệp `01_*.py` tới `81_*.py`.

## Kho này có gì

```
du-an-ptdl/          dự án mẫu hoàn chỉnh, đúng cây thư mục
                     mà sách hướng dẫn dựng
├── data/raw/        dữ liệu gốc, không bao giờ sửa
├── data/processed/  dữ liệu đã xử lý, xóa đi và tạo lại được
├── code/            chương trình Python, đánh số theo thứ tự chạy
├── models/          mô hình đã huấn luyện
├── output/          hình và bảng kết quả
└── requirements.txt

cong-cu/
└── sinh_du_lieu.py  chương trình sinh bộ dữ liệu mô phỏng của sách

bai-tap/             đề bài tập bốn tầng, một tệp mỗi chương (chương 2 tới 10)
tai-lieu-kem/        tài liệu kèm sách chính, một tệp mỗi chương (chương 1 tới 11):
                     đáp án luyện tập, tài liệu đọc thêm, thực hành với AI,
                     chú giải các đoạn mã trích
```

## Ở sách thì tra ở đâu

| Trong sách gặp | Mở tệp nào trong kho |
|---|---|
| đoạn mã `**▪ Mã N.k.**`, muốn xem trọn chương trình | `du-an-ptdl/code/` |
| dòng `# ...` đánh dấu chỗ đã lược | `tai-lieu-kem/chNN.md`, mục *Chú giải các đoạn mã trích* |
| mục `N.k. Bài tập` | `bai-tap/chNN-bai-tap.md` |
| khối `✓ Luyện tập N.k`, muốn đối chiếu đáp án | `tai-lieu-kem/chNN.md`, mục *Đáp án luyện tập* |
| khối `◈ Học cùng AI` | `tai-lieu-kem/chNN.md`, mục *Thực hành với AI* |
| muốn đọc thêm về một chủ đề | `tai-lieu-kem/chNN.md`, mục *Tài liệu đọc thêm* |

Sinh viên nên **tự dựng dự án của mình theo sách**, và chỉ mở kho này để đối chiếu khi kết quả trên
máy khác với kết quả in trong sách.

## Vì sao mã trong sách ngắn hơn mã ở đây

Sách chỉ in **phần mang bài học** của mỗi chương trình, tối đa khoảng 15 dòng. Phần khung lặp lại ở
mọi file (nhập thư viện, đặt lại bảng mã cho đầu ra, dựng đường dẫn, hàm `main`) và phần trang trí
biểu đồ được để ở kho này. Dòng `# ...` trong sách đánh dấu chỗ đã lược và nói rõ đã lược cái gì.

Đoạn mã in trong sách và file ở đây dùng **cùng tên định danh, cùng thứ tự dòng**; khác nhau ở phần
khung đã lược và ở thụt lề, vì nhiều đoạn được trích từ trong thân một hàm. Chỗ nào tên trong sách
buộc phải khác tên trong file (khi một mạch kể in liền nhiều mô hình mà file gọi lại cùng một hàm)
thì dòng `# ...` của sách nói ra điều đó.

Mọi tham số ảnh hưởng tới con số in trong sách đều có mặt trong sách, nên gõ lại đoạn trích thì ra
đúng con số sách in.

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
| pyarrow | 25.0.1 | | | |

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
# macOS, Linux
for f in code/*.py; do python "$f"; done

# Windows PowerShell
Get-ChildItem code\*.py | ForEach-Object { python $_.FullName }
```

## Dữ liệu

`du-an-ptdl/data/raw/sinhvien-truc-tuyen.csv` mô tả kết quả học tập một học phần trực tuyến của
3.000 sinh viên, 3.050 dòng và 19 cột.

**Toàn bộ số liệu là dữ liệu mô phỏng, không phải dữ liệu thật của bất kỳ sinh viên nào.** Dữ liệu
sinh bằng `cong-cu/sinh_du_lieu.py` với giá trị khởi tạo ngẫu nhiên cố định, nên chạy lại chương trình đó luôn cho
ra đúng bộ dữ liệu này và mọi con số in trong sách đều tái lập được.

Dữ liệu được gài sẵn bốn loại lỗi mà một bộ dữ liệu thật thường mắc: giá trị thiếu, giá trị ngoại lai,
định dạng không nhất quán và dòng trùng lặp. Chương 2 tìm ra chúng, chương 3 xử lý chúng.

## Lưu ý khi chạy

- Mọi đường dẫn trong mã dựng từ `__file__`, nên chạy được từ bất kỳ thư mục nào.
- Chương trình in tiếng Việt đều đặt `sys.stdout.reconfigure(encoding="utf-8")` ở đầu file. Thiếu dòng
  này, console Windows báo `UnicodeEncodeError`.
- `data/raw/` chỉ đọc. Mọi kết quả xử lý ghi vào `data/processed/`, `models/` hoặc `output/`.

## Phạm vi của kho này

Kho chứa phần tài liệu mà người học cần cầm trên tay khi ngồi trước máy:

- **mã nguồn và dữ liệu do nhóm biên soạn tự tạo ra** – chương trình Python của từng chương, bộ dữ
  liệu mô phỏng cùng chương trình sinh ra nó, và các kết quả do chính những chương trình đó tạo ra;
- **đề bài tập** của chương 2 tới chương 10;
- **tài liệu kèm sách chính** – đáp án luyện tập, tài liệu đọc thêm, thực hành với AI và chú giải
  các đoạn mã trích.

Kho **không chứa** thân sách chính, **không chứa** tập Đáp án và hướng dẫn chấm dành cho giảng viên,
và **không chứa** bất kỳ phần nào của các tài liệu tham khảo có bản quyền. Danh mục tài liệu tham
khảo in trong sách; tài liệu nào truy cập mở thì sách ghi kèm địa chỉ để người học tự tra.

## Giấy phép

Xem `LICENSE`.
