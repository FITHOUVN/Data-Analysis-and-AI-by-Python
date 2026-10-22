# Bài tập – Chương 2

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Trừ khi đề nêu nguồn khác, mọi bài làm trên `data/raw/sinhvien-truc-tuyen.csv` và giải được bằng năm
quy trình của chương. Chương dừng ở việc **phát hiện** vấn đề: không bài nào yêu cầu sửa dữ liệu.

### Củng cố

**Bài 2.1** (★ · MTS2 · ~10 phút · Kiến thức: mục 2.2, 2.4, 2.5 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) `pd.read_csv(f)` cho bảng `(1200, 1)`, tên cột duy nhất là chuỗi `ma_kh;ten_kh;tong_tien`:
A. thiếu dòng tiêu đề; B. sai bảng mã; C. sai dấu phân tách cột; D. có dòng nhiều trường hơn tiêu đề.

(2) Cột `gio_lam` của bảng 5.000 dòng có `count` 5000, `mean` 180,4, `50%` 38,0, `std` 1.420,6,
`max` 99999: A. không có vấn đề vì không thiếu ô; B. có giá trị đánh dấu, phải đếm số dòng mang 99999
trước khi dùng; C. lệch phải, cần lấy logarit; D. phải bỏ cột vì `std` lớn hơn `mean`.

(3) Câu hỏi "phân bố thời lượng học khác nhau thế nào giữa ba loại thiết bị" ứng với: A. ba histogram
chồng lên nhau; B. biểu đồ hộp xếp cạnh nhau; C. biểu đồ cột thời lượng trung bình; D. biểu đồ phân
tán thời lượng theo thiết bị.

**Bài 2.2** (★ · MTS2 · ~10 phút · Kiến thức: mục 2.4, 2.6 · dạng: đúng hoặc sai, có giải thích)

Mỗi phát biểu kèm một câu giải thích. (1) `df.isna().sum().sum()` cho số dòng có ô thiếu. (2) Ô chứa
đúng một dấu cách được `isna()` tính là giá trị thiếu. (3) Cột số bị đọc thành chuỗi thì vắng mặt
khỏi bảng `describe()` mặc định, không có thông báo. (4) Mọi giá trị vượt ngưỡng 1,5 lần khoảng tứ
phân vị đều phải bị loại trước khi dùng.

**Bài 2.3** (★ · MTS2 · ~15 phút · Kiến thức: mục 2.4.1 · dạng: đọc bảng kết quả)

Đi Quy trình 2.2 trên bảng dưới. Nộp ba nghi vấn, mỗi nghi vấn kèm một dòng lệnh kiểm được nó.

**Bảng cho Bài 2.3.** Thống kê mô tả ba cột của một bảng đơn hàng 2.400 dòng.

| Cột | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| `so_luong` | 2400 | 3,18 | 2,04 | 0 | 2 | 3 | 4 | 25 |
| `don_gia` | 2400 | 182.450 | 931.220 | 1.000 | 35.000 | 62.000 | 120.000 | 9.999.999 |
| `giam_gia_pt` | 2185 | 7,42 | 9,15 | −1 | 0 | 5 | 10 | 90 |

**Bài 2.4** (★ · MTS2 · ~10 phút · Kiến thức: mục 2.2.1, 2.2.4 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, kèm cách sửa. Dấu hiệu: (1) `AttributeError: 'DataFrame'
object has no attribute 'str'`; (2) tên tỉnh in ra là `HÃ\xa0 Ná»\x99i` trong khi bảng đủ dòng và cột;
(3) `KeyError: 0` khi lấy dòng đầu của bảng đã lọc; (4) một dòng kết quả của phép tổng là các tên mức
dính liền nhau. Nguyên nhân: A. bảng còn cột chữ; B. thừa một cặp ngoặc vuông khi chọn cột; C. chỉ
mục giữ nhãn gốc sau khi lọc; D. sai bảng mã, không sinh thông báo.

**Bài 2.5** (★ · MTS2 · ~10 phút · Kiến thức: mục 2.6.2 · dạng: tính toán)

Cột `so_don_hang` của bảng 2.400 dòng có $Q_1 = 12$, $Q_3 = 40$, và 196 dòng vượt ngưỡng trên. Tính
$IQR$, hai ngưỡng, tỉ lệ dòng vượt ngưỡng; nêu kết luận bước 3 Quy trình 2.4; giải thích vì sao ngưỡng
dưới âm trong khi số đơn hàng không thể âm.

**Bài 2.6** (★ · MTS2 · ~10 phút · Kiến thức: mục 2.5.4 · dạng: chọn và giải thích)

Chọn loại biểu đồ, kèm một lưu ý khi vẽ. (1) Mỗi mức `nhom_tuoi` có bao nhiêu sinh viên?
(2) `so_bai_tap_nop` và `diem_cuoi_ky` quan hệ hình dạng thế nào? (3) `nop_tre_tb_gio` có đuôi dài về
một phía không? (4) Trong từng mức `hinh_thuc_hoc`, tỉ lệ Đạt và Không đạt chia thế nào?

**Bài 2.7** (★ · MTS2 · ~15 phút · Kiến thức: mục 2.6.1 · dạng: giải thích có căn cứ số)

Cột `so_lan_dang_nhap` có 10 dòng mang −1; `tong_thoi_luong_phut` có 14 dòng mang 99999. (a) Cột nào
bị lệch trung bình và lệch hệ số tương quan nhiều hơn, giải thích theo cách giá trị đánh dấu tham gia công
thức trung bình. (b) Nêu hai phép đo kiểm được dự đoán đó, mỗi phép một dòng lệnh.

### Vận dụng

**Bài 2.8** (★★ · MTS2 · ~20 phút · Kiến thức: mục 2.2.2, 2.3.1 · dạng: viết mã)

Đọc file chỉ với năm cột `ma_sv`, `nhom_tuoi`, `so_lan_dien_dan`, `diem_giua_ky`, `ket_qua` bằng tham
số thu hẹp phạm vi đọc ở Bảng 2.3. Viết trước danh sách cột lẽ ra là số, rồi đi Quy trình 2.1 từ bước
2 tới bước 5. Nộp mã, kích thước bảng, số ô thiếu theo cột, và một câu giải thích vì sao thứ tự cột
nhận được khác thứ tự liệt kê trong lệnh. Tự kiểm: tổng ô thiếu bằng 189.

**Bài 2.9** (★★ · MTS2 · ~25 phút · Kiến thức: mục 2.3.4, 2.4.3 · dạng: viết mã)

Lập bảng thống kê điểm cuối kỳ theo `nganh` bằng tổng hợp có đặt tên: số sinh viên, điểm trung bình,
độ lệch chuẩn. (a) Bảng có bao nhiêu dòng, vì sao khác số mức ở Bảng 2.2? (b) Tổng số dòng thuộc các
mức sai. (c) Mức nào đứng đầu về điểm trung bình, với bao nhiêu sinh viên? (d) Vì sao kết luận "ngành
đứng đầu bảng học tốt nhất" sai? Tự kiểm: tổng cột số sinh viên bằng 3.050.

**Bài 2.10** (★★ · MTS2 · ~30 phút · Kiến thức: mục 2.4.3, 2.6.2, 2.6.3 · dạng: viết mã) **[HTTT]**

Cột `diem_tb_bai_tap` vắng mặt khỏi mọi bảng thống kê của chương vì mang kiểu chuỗi. Khôi phục nó về
dạng số **trong bộ nhớ**, không sửa `df` và không ghi file. (a) Đếm số ô dùng dấu phẩy thập phân, rồi
tạo `Series` số theo hai cách, đổi kiểu thẳng và thay dấu trước khi đổi kiểu; so về số ô thiếu và
trung bình. (b) Đi Quy trình 2.4 cho `Series` đúng: $Q_1$, $Q_3$, $IQR$, hai ngưỡng, số dòng vượt
ngưỡng, kết luận bước 5. (c) Tính Pearson và Spearman với `diem_cuoi_ky`; cột này đứng thứ mấy khi
chen vào bảng xếp hạng ở Mã 2.11? (d) Viết một dòng bổ sung cho Bảng 2.13, đúng sáu cột, ghi rõ hậu
quả đo được ở câu (c). Tự kiểm: cách đúng cho 0 ô thiếu.

**Bài 2.11** (★★ · MTS2 · ~25 phút · Kiến thức: mục 2.5 · dạng: viết mã và vẽ hình)

Hai câu hỏi, mỗi câu một hình, mỗi hình đi đủ năm bước Quy trình 2.3 và lưu vào `output/`. (a) Tỉ lệ
không đạt khác nhau thế nào giữa hai hình thức học? Ghi cỡ nhóm lên hình, trục tung từ 0. (b) Điểm
giữa kỳ có bị chặn ở 10,0 như điểm cuối kỳ không? Vẽ histogram 20 khoảng, in chiều cao hai khoảng
cuối, đếm số ô mang đúng 10,0. Tự kiểm: trung bình hai tỉ lệ ở câu (a), lấy trọng số theo cỡ nhóm,
cho lại 17,9%.

### Thực tế và liên ngành

**Bài 2.12** (★★★ · MTS2 · ~45 phút · Kiến thức: mục 2.2, 2.6.3 · dạng: tình huống doanh nghiệp, Kế toán)

Phòng kế toán Công ty TNHH Minh An xuất bảng công nợ từ phần mềm kế toán trên máy đặt vùng miền Việt
Nam. File `congno-thang9.csv` có 412 dòng; đọc với tham số mặc định cho bảng `(412, 1)`. Ba dòng đầu:

```
ma_kh;ten_kh;ngay_hd;so_hd;tien_truoc_vat;thue_suat;tong_tien
0045;Nhà sách Hoa Sen;15/09/2026;HD-0912;12500000,00;8;13500000,00
0102;Nhà sách Hoa Mai;2026-09-18;HD-0915;;10;0,00
```

(a) Đi Quy trình 2.1 bước 1: số cột mong đợi và căn cứ đếm. (b) Viết lệnh đọc đúng, giải thích từng
tham số theo Bảng 2.3 kèm triệu chứng nó xử lý; mã khách hàng có chữ số 0 ở đầu. (c) Lập miền giá trị
hợp lý cho `thue_suat` và `tong_tien`, viết trước khi xem dữ liệu, kèm căn cứ nghiệp vụ. (d) Lập danh
sách vấn đề sơ bộ theo mẫu Bảng 2.13 cho ba dòng đầu, kèm phép kiểm trên cả 412 dòng. Tự kiểm: bảng
có 7 cột, hai cột tiền mang kiểu số, mã khách giữ chữ số 0 ở đầu.

**Bài 2.13** (★★★ · MTS2 · ~45 phút · Kiến thức: mục 2.6.1, 2.6.2 · dạng: thiết kế phép kiểm, Du lịch)

Một khách sạn ở Đà Nẵng giao bảng đặt phòng một quý, 9.800 dòng, bảy cột: `ma_dat_phong`,
`ngay_nhan`, `ngay_tra`, `so_dem`, `so_khach`, `gia_phong_dem` (đồng), `kenh_dat` (Trực tiếp, Đại lý
trực tuyến, Công ty). Hệ thống cũ ghi `gia_phong_dem` bằng 0 cho đơn thuộc hợp đồng chưa xác định giá.

(a) Lập bảng miền giá trị hợp lý cho năm cột không phải mã, mỗi cột một câu căn cứ nghiệp vụ.
(b) Thiết kế năm phép kiểm, mỗi phép một dòng lệnh, phủ năm bước đầu Quy trình 2.5; ghi rõ kết quả thế
nào thì coi là có vấn đề. (c) Vì sao `isna()` không tìm ra các đơn giá 0, và hậu quả với giá phòng
trung bình dùng để báo cáo?

**Bài 2.14** (★★★ · MTS2 · ~90 phút · Kiến thức: toàn chương · dạng: làm nhóm ba người,
có trình bày, Công nghệ sinh học)

Một phòng thí nghiệm nuôi cấy tế bào ghi quan trắc buồng nuôi mỗi 15 phút một dòng trong 60 ngày:
`thoi_diem`, `ma_buong` (12 buồng), `nhiet_do_C`, `do_am_pt`, `co2_ppm`, `trang_thai_cua`. Cảm biến
ghi −999 khi mất kết nối, và một trong 12 buồng từng được thay cảm biến độ ẩm giữa kỳ bằng loại báo
theo tỉ lệ 0 tới 1 thay vì theo phần trăm.

Chia việc theo ba cụm của Quy trình 2.5: bước 1 và 2, bước 3 và 4, bước 5 kèm bước 6; mỗi thành viên
nộp mã và kết quả phần phụ trách. Sản phẩm: (a) danh sách vấn đề theo đúng sáu cột Bảng 2.13; (b) một
biểu đồ cho thấy dấu hiệu của buồng bị thay cảm biến, kèm lý do chọn loại đó; (c) bài trình bày 5
phút bảo vệ thứ tự ưu tiên, trả lời được: nếu chỉ còn một ngày trước hạn giao báo cáo thì xử lý vấn
đề nào trước, vì sao. Đánh giá: phủ đủ bốn loại lỗi; mỗi dòng nêu được cách kiểm đã xử lý xong; phân
biệt rõ giá trị thiếu với giá trị đánh dấu.

### Thử thách có AI

**Bài 2.15** (★★★★ · MTS2 · ~90 phút · Kiến thức: toàn chương · dạng: phát hiện lỗi trong mã do AI sinh)

**Bài toán.** Trợ lý AI viết đoạn mã khám phá nhanh dưới đây; mã chạy hết, không dừng. Từ kết quả in
ra, nó kết luận ba điều: điểm giữa kỳ không còn ô thiếu; cột `tinh_thanh` đủ dữ liệu ở mọi dòng; độ
phân tán điểm cuối kỳ giữa các nhóm giới tính gần như bằng nhau.

```python
df = pd.read_csv("data/raw/sinhvien-truc-tuyen.csv",
                 encoding="utf-8")
COT_SO = ["so_lan_dang_nhap", "tong_thoi_luong_phut",
          "ty_le_hoan_thanh_video",
          "so_bai_tap_nop", "diem_tb_bai_tap", "so_lan_dien_dan",
          "nop_tre_tb_gio", "diem_giua_ky", "diem_cuoi_ky"]

print("1 | Tổng từng cột số:")
print(df[COT_SO].sum())

print("2 | Điền giá trị thiếu của điểm giữa kỳ bằng 0:")
df["diem_giua_ky"].fillna(0, inplace=True)
print("Số ô thiếu còn lại:", df["diem_giua_ky"].isna().sum())

print("3 | Phân bố tỉnh thành:")
dem = df["tinh_thanh"].value_counts()
print("Đã đếm", dem.sum(), "dòng, cột này không có ô thiếu")

print("4 | Độ phân tán điểm cuối kỳ theo giới tính:")
print(df.groupby("gioi_tinh")["diem_cuoi_ky"].agg(np.std).round(4))
```

**Nhiệm vụ với AI.** Dùng một công cụ AI tạo sinh theo hai hướng: yêu cầu nó rà soát đoạn mã trên và
liệt kê lỗi; yêu cầu nó sinh một đoạn mã khám phá tương đương chỉ từ mô tả bài toán.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, kèm trích đoạn phản
hồi ứng với từng lỗi. Không dán dữ liệu thật vào prompt, chỉ dán tên cột.

**Kiểm chứng.** Tìm **bốn** lỗi; mỗi lỗi nộp dòng mã sai, triệu chứng khi chạy, minh chứng bằng số từ
lần chạy của chính sinh viên, bản sửa và kết quả sau khi sửa. So với phần trợ lý AI tìm ra: lỗi nào
nó bỏ sót, lỗi nào nó báo oan. Cuối cùng chỉ ra ba kết luận của nó sai ở đâu.

**Phản tư** (100 tới 200 từ). Trợ lý AI giúp được gì và không giúp được gì ở bước khám phá dữ liệu;
vì sao lỗi chạy êm mà kết quả sai khó phát hiện hơn lỗi làm chương trình dừng; rủi ro khi để trợ lý AI tự
quyết cách xử lý giá trị thiếu trên dữ liệu học tập của người thật.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất
lượng kiểm chứng phần trợ lý AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng năm
mức chi tiết nằm trong tài liệu hướng dẫn chấm đi kèm sách.
