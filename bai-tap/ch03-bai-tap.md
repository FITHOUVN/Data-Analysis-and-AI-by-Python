# Bài tập – Chương 3

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Trừ khi đề nêu nguồn khác, mọi bài làm trên `data/raw/sinhvien-truc-tuyen.csv` và giải được bằng tám
quy trình của chương; từ tầng hai trở lên nộp mã kèm kết quả in ra.

### Củng cố

**Bài 3.1** (★ · MTS3 · ~10 phút · Kiến thức: mục 3.1.2, 3.3.2 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Cột `thu_nhap` thiếu 9% ô; nhóm thiếu lệch 0,42 độ lệch chuẩn của biến mục tiêu so với nhóm đủ.
Theo Quy trình 3.2: A. điền trung vị, vì 9% xa ngưỡng 30%; B. bỏ cả cột, vì chênh lệch vượt 0,1 độ
lệch chuẩn; C. điền trung vị kèm một cột cờ; D. xóa dòng thiếu, vì 9% là tỉ lệ nhỏ.

(2) `pd.to_datetime(cot, format="mixed", dayfirst=True)` trên cột trộn ba định dạng ngày cho 0 ô
thiếu. Dấu hiệu chứng minh kết quả vẫn sai: A. kiểu cột ra không phải `datetime64[us]`; B. số tháng
khác nhau là 12 thay vì 2; C. lệnh in kèm cảnh báo; D. số dòng của bảng giảm.

**Bài 3.2** (★ · MTS3 · ~10 phút · Kiến thức: mục 3.2.2, 3.3.1, 3.4.2 · dạng: đúng hoặc sai, có giải thích)

(1) `str.capitalize()` chỉ làm số mức của cột nhãn giảm hoặc giữ nguyên, nên số mức không đổi nghĩa là
cột không bị sửa. (2) `dropna()` không tham số bỏ các dòng trống hẳn. (3) Ngưỡng ba độ lệch chuẩn luôn
khoanh nhiều ô hơn ngưỡng khoảng tứ phân vị. (4) Số dòng còn lại sau `drop_duplicates()` không phụ
thuộc thứ tự dòng, nhưng bản được giữ thì phụ thuộc.

**Bài 3.3** (★ · MTS3 · ~10 phút · Kiến thức: mục 3.1.1 · dạng: tính toán)

Một bảng 5.200 dòng có cột `so_gio_tu_hoc` thiếu 412 ô. Biến mục tiêu có độ lệch chuẩn 2,10; nhóm
thiếu đạt trung bình 5,80, nhóm đủ 6,45. Tính tỉ lệ thiếu và tỉ số bước 3 Quy trình 3.1, ghi dòng kết
luận bước 5, chọn cách xử lý theo Quy trình 3.2, cho biết xóa dòng có được phép hay không.

**Bài 3.4** (★ · MTS3 · ~10 phút · Kiến thức: mục 3.2.4, 3.3 · dạng: ghép đôi)

Ghép mỗi triệu chứng với một nguyên nhân, kèm một dòng lệnh sửa. Triệu chứng: (1) bốn cột đếm in ra
`38.0` thay vì `38`; (2) cột ngày có 0 ô thiếu nhưng trải khắp 12 tháng; (3) cột nhãn vẫn đúng 15 mức
mà cả 2.992 ô đều bị viết lại; (4) đổi một cột điểm sang kiểu số sinh ra 151 ô thiếu. Nguyên nhân:
A. dấu phẩy thập phân chưa thay trước khi đổi kiểu; B. `dayfirst` áp cả cho ô dạng ISO; C. phép hạ chữ
áp cho mọi ký tự sau ký tự đầu; D. ô thiếu đặt vào cột số nguyên đẩy cả cột sang số thực.

**Bài 3.5** (★ · MTS3 · ~15 phút · Kiến thức: mục 3.2.2 · dạng: đọc bảng kết quả)

**Bảng cho Bài 3.5.** Bốn cột số của một bảng hợp đồng viễn thông 12.000 dòng.

| Cột | $Q_1$ | $Q_3$ | trung bình | độ lệch chuẩn |
|---|---|---|---|---|
| `so_don_hang` | 8 | 24 | 19,4 | 11,8 |
| `tien_cuoc_dong` | 120.000 | 480.000 | 642.000 | 5.480.000 |
| `so_ngay_cham` | 1 | 4 | 3,6 | 5,4 |
| `diem_danh_gia`, thang 1 tới 5 | 3,0 | 5,0 | 4,1 | 1,2 |

Với từng cột, tính khoảng tứ phân vị, hai ngưỡng theo cả hai cách, tỉ số độ lệch chuẩn chia khoảng tứ
phân vị. Cho biết cột nào phải xử lý giá trị đánh dấu trước khi tính lại ngưỡng, cột nào chọn cách nào
cũng được, và vì sao `diem_danh_gia` không có ô nào vượt ngưỡng trên theo cách thứ nhất.

**Bài 3.6** (★ · MTS3 · ~10 phút · Kiến thức: mục 3.1.2, 3.1.3 · dạng: chọn và giải thích)

Một bảng hồ sơ vay 8.400 dòng, biến mục tiêu `no_qua_han` với tỉ lệ chung 12%. Chọn cách xử lý từng
cột theo Quy trình 3.2, mỗi cột một câu căn cứ số. (1) `ma_the_bao_hiem`, chuỗi định danh, thiếu 34%.
(2) `so_nguoi_phu_thuoc`, biến đếm, thiếu 3,1%, tỉ số 0,02. (3) `tinh_trang_hon_nhan`, phân loại 4
mức, thiếu 7,8%, nhóm thiếu có tỉ lệ nợ quá hạn 31%. (4) `thu_nhap_ban_dau`, biến số, thiếu 1,4%, tỉ
số 0,05, còn 22 ô mang giá trị −1.

**Bài 3.7** (★ · MTS3 · ~15 phút · Kiến thức: mục 3.4.1 · dạng: giải thích có căn cứ số)

Một bảng đơn hàng 4.000 dòng, khóa `ma_don`: `duplicated()` bằng 55, `duplicated(keep=False)` bằng
108, `duplicated(subset=["ma_don"])` bằng 70, cùng phép ấy với `keep=False` bằng 136. (a) Số dòng còn
lại sau `drop_duplicates()`. (b) Nếu mọi nhóm trùng khít chỉ gồm hai dòng thì con số thứ hai bằng bao
nhiêu; từ chênh lệch, suy ra bao nhiêu mã xuất hiện ba lần. (c) Số cặp xung đột phải quyết định bằng
tay, kèm phép tự kiểm bước 1 Quy trình 3.8. (d) Vì sao `keep="first"` không được chọn mặc định.

### Vận dụng

**Bài 3.8** (★★ · MTS3 · ~25 phút · Kiến thức: mục 3.1 · dạng: viết mã)

Chương đã chẩn đoán ba cột có ô thiếu; cột `thiet_bi_chinh` còn lại. (a) Đi Quy trình 3.1, nộp dòng
kết luận, rồi đi Quy trình 3.2 và nêu bước nào không phải chạy. (b) Điền theo Quy trình 3.3 bằng hai
cách, mức hay gặp nhất và mức mới `"Không rõ"`, mỗi cách in số mức và số dòng từng mức; nêu một lý do
chọn cách của chương. (c) Nhóm thiếu có 9 trên 37 sinh viên không đạt, tức 24,3% so với 17,9% chung;
vì sao con số đó không đổi được kết luận bước 3. Tự kiểm: thiếu 1,21%, tỉ số bước 3 bằng 0,068.

**Bài 3.9** (★★ · MTS3 · ~25 phút · Kiến thức: mục 3.2 · dạng: viết mã)

Cột `so_lan_dien_dan` là cột duy nhất mà ngưỡng ba độ lệch chuẩn khoanh nhiều ô hơn ngưỡng khoảng tứ
phân vị. (a) Đi Quy trình 3.4, bước 3 in khoảng cách giữa các giá trị liền nhau của mười hai giá trị
lớn nhất theo đơn vị khoảng tứ phân vị; rồi đi Quy trình 3.5 và nêu bước nào áp dụng. (b) Đo bốn cách
xử lý nhóm vượt ngưỡng theo mẫu Bảng 3.5, cột đo gồm số dòng, trung bình, độ lệch chuẩn, tương quan
với `diem_cuoi_ky`; kết luận một câu về việc chặn biên. Tự kiểm: ngưỡng trên 11,0 với 9 ô vượt, ngưỡng
ba độ lệch chuẩn 10,39 với 17 ô.

**Bài 3.10** (★★ · MTS3 · ~30 phút · Kiến thức: mục 3.3.1 · dạng: viết mã)

Viết hàm `chuan_hoa_nhan(d, nhan_mong_doi, bang_tra)` áp Quy trình 3.6 cho cả tám cột phân loại, in
cho mỗi cột số mức trước và sau, số dòng đã sửa, tập nhãn lạ còn lại. (a) Viết `nhan_mong_doi`
**trước khi** xem dữ liệu, chạy hàm, cho biết năm cột nào không cần sửa gì. (b) Chạy lại với
`str.capitalize()` thay bảng tra; chỉ ra cột nào mà phép đếm số mức báo đạt trong khi phép so tập nhãn
báo sai. Tự kiểm: cách đúng sửa 284 dòng, không để lại nhãn lạ.

**Bài 3.11** (★★ · MTS3 · ~30 phút · Kiến thức: mục 3.3.3, 3.4.3 · dạng: viết mã) **[HTTT]**

Viết chương trình kiểm tra kết quả cuối cùng cho `data/processed/du-lieu-sach.csv`. (a) Bảy phép kiểm: số dòng, số ô
thiếu, dòng trùng khít, dòng trùng `ma_sv`, kiểu của mười cột theo bảng khai trước, số mức của tám cột
phân loại, miền giá trị cột ngày. (b) Đọc file hai lần, mặc định và có `parse_dates`; cho biết số cột
chuỗi mỗi lần và phép kiểm nào phát hiện khác biệt. (c) In bằng chứng cho thấy V7 và V15 được giữ
nguyên có chủ ý. Tự kiểm: 3.012 dòng, 0 ô thiếu, 12 dòng trùng `ma_sv`, cột ngày trải 2 tháng.

### Thực tế và liên ngành

**Bài 3.12** (★★★ · MTS3 · ~40 phút · Kiến thức: mục 3.2 · dạng: bảo vệ phương án, Tài chính và Ngân hàng)

(a) Lập bảng bốn cách xử lý tám dòng của V7, gồm giữ nguyên, chia 60, đổi thành giá trị thiếu rồi điền
trung vị, loại bỏ tám dòng; mỗi cách ghi tương quan với `diem_cuoi_ky` và một câu về phần thông tin bị mất. (b) Chỉ
ra cách nào không lùi lại được sau khi ghi file, và điều kiện duy nhất cho phép chia 60. (c) Một ngân
hàng số giao bảng 26.000 hồ sơ vay; cột `thoi_gian_xu_ly_phut` có 31 ô từ 2.880 tới 14.400 trong khi
99% số ô dưới 240, và hệ thống từng đổi phần mềm giữa kỳ. Đi Quy trình 3.4, nêu phép kiểm bước 4 và
kết luận bước 5. (d) Viết thư ba câu gửi người quản trị hệ thống, chứa đúng một câu hỏi trả lời được
bằng có hoặc không.

**Bài 3.13** (★★★ · MTS3 · ~40 phút · Kiến thức: mục 3.4 · dạng: tình huống doanh nghiệp, Du lịch và Khách sạn)

(a) Tái lập 12 cặp xung đột của V15 từ bản sạch, in cột gây xung đột và mức lệch. (b) Đo bốn cách
quyết định ở bước 5 Quy trình 3.8, gồm giữ cả hai bản, giữ bản đầu, giữ bản cuối, gộp bằng trung bình;
mỗi cách ghi số dòng, trung bình và độ lệch chuẩn của cột gây xung đột, tương quan với `diem_cuoi_ky`,
số giá trị lẻ sinh ra; nêu rủi ro của cách chương đã chọn kèm một câu về việc chia tập ở Chương 4,
dựa vào khối Chú ý ở cuối mục 3.4.2 chứ không cần kiến thức của Chương 4.
(c) Một khách sạn ở Nha Trang giao bảng đặt phòng 18.400 dòng, khóa `ma_dat_phong`, 240 dòng trùng
khít và 262 dòng trùng khóa, các cặp xung đột lệch ở `so_khach` và `kenh_dat`. Cho biết số cặp phải
quyết định bằng tay, vì sao xung đột ở hai cột đổi kết luận bước 4, và cách quyết định đề nghị.

**Bài 3.14** (★★★ · MTS3 · ~90 phút · Kiến thức: toàn chương · dạng: làm nhóm ba người, có trình bày,
Điện tử và Viễn thông)

Bảng nhật ký một quý của 420 trạm thu phát sóng, 96.000 dòng: `thoi_diem_ghi` trộn hai định dạng ngày;
`nhiet_do_C` ghi 999 khi cảm biến mất kết nối; `cong_suat_W` có 214 ô mang 0; `dien_tieu_thu_kwh` dùng
dấu phẩy thập phân nên cả cột mang kiểu chuỗi; `trang_thai` lệch hoa thường; 1.280 dòng trùng khít và
1.355 dòng trùng cặp `ma_tram` với `thoi_diem_ghi`. Chia việc theo ba cụm: thiếu và giá trị đánh dấu;
ngoại lai và định dạng; dòng trùng và chương trình hợp nhất.

Sản phẩm: (a) bảng vấn đề theo đúng sáu cột Bảng 2.13; (b) chương trình làm sạch bảy bước theo thứ tự
chạy ở mục 3.4.3, có phép kiểm sau mỗi bước, kèm bảng đối chiếu kiểu khai trước với kiểu thực tế;
(c) trình bày 5 phút bảo vệ hai quyết định khó nhất, trả lời được: 214 ô `cong_suat_W` bằng 0 là giá
trị đánh dấu hay số đo thật, và căn cứ nào phân biệt hai khả năng đó. Đánh giá: thứ tự bảy bước đúng và
giải thích được từng ràng buộc; mỗi quyết định có một con số làm căn cứ.

### Thử thách có AI

**Bài 3.15** (★★★★ · MTS3 · ~90 phút · Kiến thức: toàn chương · dạng: phát hiện lỗi trong mã do AI sinh)

**Bài toán.** Một trợ lý AI được yêu cầu viết chương trình làm sạch bộ dữ liệu của sách. Mã dưới đây
chạy hết, không dừng, không in cảnh báo, và ba dòng kiểm nó in ra đều đạt: kích thước `(3012, 19)`,
số ô thiếu `0`, số dòng trùng khít `0`.

```python
d = df.drop_duplicates().reset_index(drop=True)
d["diem_tb_bai_tap"] = pd.to_numeric(
    d["diem_tb_bai_tap"], errors="coerce")
d["ngay_dang_ky"] = pd.to_datetime(d["ngay_dang_ky"],
                                   format="mixed", dayfirst=True,
                                   errors="coerce")
for cot in COT_PHAN_LOAI:
    d[cot] = d[cot].str.strip().str.capitalize()
d["tong_thoi_luong_phut"] = d["tong_thoi_luong_phut"].mask(
    d["tong_thoi_luong_phut"] == 99999)
d["so_lan_dang_nhap"] = d["so_lan_dang_nhap"].mask(
    d["so_lan_dang_nhap"] < 0)
d = d.fillna(d.median(numeric_only=True).to_dict())
for cot in ["tinh_thanh", "thiet_bi_chinh"]:
    d[cot] = d[cot].fillna("Không rõ")
d.to_csv("data/processed/du-lieu-sach.csv", index=False,
         encoding="utf-8")
```

**Nhiệm vụ với AI.** Yêu cầu một công cụ AI tạo sinh rà soát đoạn mã và liệt kê lỗi; sau đó yêu cầu nó
tự sinh một chương trình làm sạch chỉ từ mô tả bộ dữ liệu và hai điều kiện kiểm tra kết quả cuối cùng đó.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi
ứng với từng lỗi. Không dán dữ liệu thật vào prompt, chỉ dán tên cột và kiểu cột.

**Kiểm chứng.** Tìm **bốn** lỗi, cả bốn thuộc loại chạy không báo lỗi mà kết quả sai. Mỗi lỗi nộp dòng
mã sai, một phép kiểm phát hiện được nó, số đo trước và sau khi sửa, bản sửa theo quy trình của chương.
Giải thích vì sao ba dòng kiểm kia vẫn đạt, rồi so với phần trợ lý AI tự tìm: lỗi nào nó bỏ sót, lỗi
nào nó báo oan.

**Phản tư** (100 tới 200 từ). Trợ lý AI giúp được gì và không giúp được gì ở bước làm sạch; vì sao một
phép kiểm đếm được không đủ để kiểm tra kết quả cuối cùng của chương trình làm sạch; rủi ro khi để trợ lý AI tự chọn giá
trị điền cho dữ liệu học tập của người thật.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất
lượng kiểm chứng phần trợ lý AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng năm
mức chi tiết nằm trong tài liệu hướng dẫn chấm đi kèm sách.
