# Bài tập – Chương 5

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Trừ khi đề nêu nguồn khác, mọi bài làm trên gói dữ liệu Chương 4 bàn giao và giải bằng bảy quy trình của
chương; từ tầng hai trở lên nộp mã nguồn và kết quả in ra. Mọi so sánh dạng "có đổi hay không" phải in
đủ **ba con số**: số dự đoán đổi giá trị trên tổng, lệch lớn nhất, lệch của độ đo tổng hợp.

### Củng cố

**Bài 5.1** (★ · MTS4 · ~12 phút · Kiến thức: mục 5.3.3, 5.4.1 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một cây hồi quy cho R² bằng 1,0000 trên tập huấn luyện và 0,5910 trên tập kiểm tra. Kết luận hợp lý
nhất: A. mô hình đủ tốt để dùng vì khớp hoàn hảo dữ liệu đã có; B. mô hình học thuộc tập huấn luyện,
phải chặn độ phức tạp trước khi đọc tiếp; C. có một đặc trưng suy ra từ biến mục tiêu; D. chưa khóa giá trị
khởi tạo ngẫu nhiên.

(2) Dòng `rmse = mean_squared_error(y_te, d, squared=False)` dừng với `TypeError`. Dòng viết lại đúng
theo phiên bản thư viện mà chương khai báo: A. `mean_squared_error(y_te, d, squared=True)`;
B. `root_mean_squared_error(y_te, d)`; C. `mean_absolute_error(y_te, d) ** 0.5`;
D. `r2_score(y_te, d) ** 0.5`.

(3) Cần một con số nói được thành câu cho giảng viên không làm kỹ thuật, thang điểm 0 tới 10, khi dữ
liệu còn vài ca ngoại lai chưa xử lý. Theo Bảng 5.6, độ đo nên chọn: A. R²; B. MSE; C. MAE; D. RMSE.

**Bài 5.2** (★ · MTS4 · ~12 phút · Kiến thức: mục 5.1.3, 5.3.1, 5.3.4, 5.4.3 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số. (1) R² bằng −0,05 trên tập kiểm tra
chứng tỏ chương trình có lỗi. (2) Lệch R² giữa hai mô hình chỉ 0,0058 nên chọn mô hình nào cũng được.
(3) Cây hồi quy không bao giờ cho dự đoán nằm ngoài miền giá trị của biến mục tiêu trên tập huấn luyện.
(4) Tăng siêu tham số `min_samples_leaf` luôn làm R² trên tập kiểm tra tăng.

**Bài 5.3** (★ · MTS4 · ~15 phút · Kiến thức: mục 5.3.1 · dạng: tính toán)

Tám sinh viên có số bài tập đã nộp 1 · 2 · 3 · 5 · 6 · 7 · 9 · 10 và điểm cuối kỳ tương ứng
2 · 2 · 4 · 5 · 6 · 8 · 8 · 9. (a) Tính độ không thuần khiết của nút gốc. (b) Tính độ lợi thông tin của
ba điểm phân chia `so_bai_tap_nop <= 3`, `<= 5` và `<= 7`, rồi cho biết thuật toán chọn điểm nào. (c) Nêu giá
trị hai lá của cây một tầng và phép tự kiểm cho biết ba con số độ lợi vừa tính không thể sai.

**Bài 5.4** (★ · MTS4 · ~15 phút · Kiến thức: mục 5.4.1, 5.4.2 · dạng: tính toán)

Sáu sinh viên có điểm cuối kỳ thực tế 1,0 · 2,0 · 3,5 · 4,0 · 6,0 · 8,5 và được một mô hình đoán
2,2 · 2,0 · 3,0 · 5,5 · 5,8 · 7,0. (a) Lập dãy phần dư, rồi tính MAE, MSE, RMSE và tỉ số RMSE trên MAE.
(b) Tính $SSE$, $SST$ và R² theo Định nghĩa 5.7. (c) Kiểm lại RMSE bằng bước 4 Quy trình 5.6, và cho
biết phép tự kiểm ở bước 4 Quy trình 5.2 có được thỏa hay không.

**Bài 5.5** (★ · MTS4 · ~15 phút · Kiến thức: mục 5.2.3 · dạng: đọc bảng hệ số)

Chuỗi cà phê Hoa Sen khớp một mô hình hồi quy tuyến tính dự đoán doanh thu một ngày của một cửa hàng,
đơn vị triệu đồng. Các cột số **chưa** qua chuẩn hóa; biến `ngay_le` hai mức mã hóa một nóng đầy đủ cho
hai hệ số +0,4150 và −0,4150. Hệ số chặn 8,6200.

**Bảng cho Bài 5.5.** Hệ số theo đơn vị gốc và độ lệch chuẩn của bốn cột số.

| Đặc trưng | Hệ số theo đơn vị gốc | Độ lệch chuẩn |
|---|---|---|
| `so_luot_khach` | +0,0180 | 62,5 |
| `nhiet_do_cao_nhat_C` | −0,0450 | 4,0 |
| `so_nhan_vien_truc` | +0,2500 | 0,8 |
| `chi_phi_quang_cao_nghin_dong` | +0,0008 | 450,0 |

(a) Theo bước 1 Quy trình 5.3, cho biết bảng này có so được độ lớn giữa các hệ số hay không, và nêu thứ
tự bốn cột theo trị tuyệt đối của hệ số đơn vị gốc. (b) Tính hệ số theo thang chuẩn hóa, nêu lại thứ tự
rồi so hai thứ tự. (c) Áp bước 4 Quy trình 5.3 cho cặp hệ số của `ngay_le`. (d) Hệ số của nhiệt độ mang
dấu âm: theo bước 2 Quy trình 5.3, nêu hai việc cần làm trước khi coi đó là một phát hiện.

**Bài 5.6** (★ · MTS4 · ~12 phút · Kiến thức: mục 5.1.3, 5.2.3, 5.3.3 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, kèm một phép kiểm phát hiện được nó.

Dấu hiệu: (1) R² huấn luyện 1,0000 và R² kiểm tra 0,5910; (2) R² huấn luyện 0,4687 và R² kiểm tra 0,4385;
(3) R² của mốc trung bình bằng −0,0019 chứ không đúng bằng 0; (4) bỏ bước chuẩn hóa khỏi `Pipeline` làm
bảng hệ số đổi hoàn toàn mà R² không đổi chút nào.

Nguyên nhân: A. mốc học trung bình trên tập huấn luyện, còn R² tính theo trung bình của tập đang đo;
B. phép chuẩn hóa chỉ dịch và co giãn trục số nên không đổi dự đoán, chỉ đổi đơn vị của hệ số; C. mỗi lá
chứa đúng một dòng huấn luyện nên dự đoán của lá bằng giá trị thật của dòng đó; D. họ mô hình quá hẹp so
với quy luật có trong dữ liệu.

**Bài 5.7** (★ · MTS4 · ~15 phút · Kiến thức: mục 5.3.2 · dạng: đọc cây đã in ra)

Một cây hồi quy ba tầng khớp trên **bốn** cột số thang gốc là `ty_le_hoan_thanh_video`,
`so_bai_tap_nop`, `diem_tb_bai_tap` và `so_lan_dang_nhap`, bỏ hẳn `diem_giua_ky` khỏi tập đặc trưng. Số
lá 8; trên tập kiểm tra cho MAE 1,346 · RMSE 1,683 · R² 0,4632.

```
|--- ty_le_hoan_thanh_video <= 0.52
|   |--- ty_le_hoan_thanh_video <= 0.40
|   |   |--- diem_tb_bai_tap <= 6.15        -> value: [2.93]
|   |   |--- diem_tb_bai_tap >  6.15        -> value: [4.60]
|   |--- ty_le_hoan_thanh_video >  0.40
|   |   |--- so_bai_tap_nop <= 4.50         -> value: [3.97]
|   |   |--- so_bai_tap_nop >  4.50         -> value: [5.88]
|--- ty_le_hoan_thanh_video >  0.52
|   |--- diem_tb_bai_tap <= 6.85
|   |   |--- so_bai_tap_nop <= 4.50         -> value: [4.38]
|   |   |--- so_bai_tap_nop >  4.50         -> value: [6.41]
|   |--- diem_tb_bai_tap >  6.85
|   |   |--- ty_le_hoan_thanh_video <= 0.74 -> value: [6.96]
|   |   |--- ty_le_hoan_thanh_video >  0.74 -> value: [8.20]
```

Áp Quy trình 5.4. (a) Bước 1: nêu cột ở nút gốc, đối chiếu Bảng 5.2, cho biết phép tự kiểm có báo lỗi
hay không. (b) Bước 2: nêu dự đoán cho một sinh viên có tỉ lệ hoàn thành video 0,85 · 9 bài tập đã nộp ·
điểm trung bình bài tập 7,2, và viết nhánh đó thành một câu điều kiện. (c) Bước 3: kiểm miền hợp lệ của
dãy tám lá, chỉ ra cặp lá có thứ tự trái mong đợi và giải thích. (d) Bước 4: tính tỉ số dòng huấn luyện
trên lá; nêu một lý do khiến R² 0,4632 thấp hơn hẳn 0,5396 của cây ở Mã 5.10.

### Vận dụng

**Bài 5.8** (★★ · MTS4 · ~45 phút · Kiến thức: mục 5.1.2, 5.1.3, 5.2.2 · dạng: viết mã)

Đo xem sáu cột phân loại có đáng giữ trong tập đặc trưng hay không. (a) Dựng `ColumnTransformer` chỉ gồm
`StandardScaler` cho tám cột số, `remainder="drop"`, rồi đi trọn Quy trình 5.2 cho `LinearRegression`;
in số đặc trưng và bảng hai dòng ba độ đo. (b) In dòng mốc trung bình trên cùng tập kiểm tra và tỉ lệ
phần trăm sai số của mốc mà mô hình cắt được. (c) So bản rút gọn với bản 30 đặc trưng bằng ba con số,
rồi kiểm xem kết luận có bền hay không bằng cách đo **hiệu** R² hai bản trên năm lần chia tập khác
nhau, mỗi lần khớp lại cả bộ tiền xử lý; nêu hiệu có cùng dấu ở mọi lần chia và có lần nào bằng 0
không. (d) In hệ số chặn và tám hệ số;
chỉ ra cột duy nhất **đổi dấu** so với Bảng 5.2 và nêu việc cần làm theo bước 2 Quy trình 5.3.

Tự kiểm: 8 đặc trưng; R² hai tập 0,6832 và 0,6985; hệ số chặn 6,1016; hiệu ở phần (c) âm cả 5/5 lần.

**Bài 5.9** (★★ · MTS4 · ~40 phút · Kiến thức: mục 5.2.2, 5.2.3, 5.4.3 · dạng: viết mã)

Mục 5.2.3 cho thấy cặp hệ số một nóng của biến hai mức đối xứng quanh 0 vì cộng tuyến. (a) Dựng bộ tiền
xử lý như Mã 4.28 nhưng khai thêm `drop="first"` cho `OneHotEncoder`, khớp `LinearRegression`, in số đặc
trưng, hệ số chặn và R² hai tập. (b) In ba con số so với bản mã hóa đầy đủ và kết luận một câu về việc
dự đoán có đổi hay không; in "lệch lớn nhất" ở dạng khoa học `%.4e` thay vì `%.4f`, rồi cho biết hai
cách in dẫn tới hai kết luận khác nhau thế nào. (c) Lập bảng so hệ số hai bản cho ba biến
`hinh_thuc_hoc`, `vua_lam_vua_hoc`, `thiet_bi_chinh`; phát biểu quy tắc nối hệ số hai bản rồi kiểm trên
cả ba biến. (d) Cho biết tám hệ số của các cột số đổi hay không, kèm một lý do.

Tự kiểm: 24 đặc trưng; hệ số chặn 5,1654; `hinh_thuc_hoc_Trực tuyến` nhận 1,1555; lệch lớn nhất
1,6·10⁻¹⁴.

**Bài 5.10** (★★ · MTS4 · ~40 phút · Kiến thức: mục 5.4.1, 5.4.2 · dạng: viết mã)

Làm trọn bộ đo cho cấu hình chốt của chương, `max_depth=8, min_samples_leaf=20`. (a) In ba độ đo trên cả
hai tập, khoảng cách tổng quát hóa, số lá và số giá trị dự đoán khác nhau trên tập kiểm tra. (b) Tính
$SSE$, $SST$, R² bằng tay rồi đối chiếu `r2_score`; kiểm lại RMSE bằng bước 4 Quy trình 5.6. (c) In
trung vị, phân vị 90, giá trị lớn nhất của sai số tuyệt đối, số ca sai quá 2 và quá 3 điểm; so từng con
số với bộ số của mô hình tuyến tính ở Mã 5.14. (d) Đếm số dự đoán rơi ngoài thang điểm 0 tới 10, so với
18 trên 603 của mô hình tuyến tính, rồi giải thích bằng cách cây sinh ra dự đoán.

Tự kiểm: R² hai tập 0,7802 và 0,6938; $SSE$ 974,49; trung vị sai số 0,867.

**Bài 5.11** (★★ · MTS4 · ~60 phút · Kiến thức: mục 5.4.3, 5.5.1 · dạng: viết mã) **[HTTT]**

Ví dụ 5.4 chọn giữa mô hình tuyến tính và rừng ngẫu nhiên. Làm lại cho cặp mô hình tuyến tính và cây
chặn quá khớp. (a) Đi đủ sáu bước Quy trình 5.7, in kết quả và phép tự kiểm của từng bước. (b) Thêm hai
con số nghiệp vụ: số cặp dự đoán lệch quá 1 điểm, và số sinh viên bị hai mô hình xếp về hai phía ngưỡng
đạt 4,0. (c) Áp bước 1 và 2 Quy trình 5.8 cho cả hai mô hình: in phần dư trung bình toàn tập, rồi **hai**
bảng phần dư trung bình, một chia theo năm khoảng điểm thực như Bảng 5.9 và một chia theo năm khoảng dự
đoán như Bảng 5.10. Cho biết bảng nào chẩn đoán được dạng mô hình và vì sao bảng kia không. (d) Kết luận
chọn một mô hình với đủ ba loại căn cứ gồm hiệu năng, mức thay đổi ở từng cá nhân và chi phí; chỉ rõ
khoảng điểm mà cây tệ hơn mô hình tuyến tính nhiều nhất.

Tự kiểm: lệch R² −0,0652 trên lần chia của chương, hiệu âm cả 10/10 lần chia ở bước 6; 63 sinh viên bị
xếp hai phía ngưỡng 4,0.

### Thực tế và liên ngành

**Bài 5.12** (★★★ · MTS4 · ~45 phút · Kiến thức: mục 5.1.3, 5.3.3, 5.5.2 · dạng: bảo vệ phương án, Tài chính và Ngân hàng)

(a) Khớp sáu mô hình của Bảng 5.7, lập bảng gồm R² hai tập, MAE hai tập và **thứ hạng** theo từng cột R².
Chỉ ra mô hình giảm nhiều hạng nhất giữa hai cách xếp, kèm con số chứng minh.

(b) Phòng quản lý rủi ro Ngân hàng Minh An nhận một báo cáo thực tập: "cây quyết định dự đoán số tiền
khách trả nợ tháng tới, R² đạt 0,998, mô hình đã đạt yêu cầu triển khai". Báo cáo không có dòng mốc,
không nêu đo trên tập nào, không khai `max_depth`. Viết **ba** câu hỏi, mỗi câu đủ bác bỏ kết luận đó và
kèm một con số của phần (a) làm căn cứ.

(c) Đi bước 1 và bước 3 Quy trình 5.1 cho bài toán của ngân hàng: chốt biến mục tiêu, miền giá trị hợp
lệ, một độ đo chính và một độ đo phụ kèm lý do nghiệp vụ; nêu mốc hợp lý nhất và giá trị R² mà mốc đó
cho. (d) Bảo vệ phương án trong 5 phút trước hội đồng: biểu đồ cột R² có trục dọc cắt từ 0,95 tới 1,00
làm hội đồng hiểu sai thế nào, và theo mục 5.5.2 phải vẽ lại ra sao.

**Bài 5.13** (★★★ · MTS4 · ~50 phút · Kiến thức: mục 5.2.3, 5.2.4, 5.4.3 · dạng: tình huống doanh nghiệp, Điện tử và Viễn thông)

(a) Mã 5.8 đo hậu quả của **một** dòng mang giá trị đánh dấu ở cột `tong_thoi_luong_phut`. Làm lại phép thử
đó hai lần, mỗi lần đặt 99999 vào một cột khác và gán điểm cuối kỳ cho dòng thêm vào: `nop_tre_tb_gio`
với điểm 10,0; rồi `ty_le_hoan_thanh_video` với điểm 0,0. Mỗi lần in hệ số của chính cột đó trước và sau,
cùng ba con số so với mô hình gốc. (b) Một trong hai lần có lệch R² nhỏ hơn mức chênh thường thấy giữa
hai mô hình khác hẳn nhau của chương, lần kia không. Nêu hệ quả thực hành cho người chỉ theo dõi bảng
hiệu năng, kèm phát biểu bằng lời về quan hệ sai mà bảng hệ số của lần thứ nhất đưa ra.

(c) Công ty Viễn thông Bạch Đằng dự đoán lưu lượng ngày hôm sau của một trạm phát, đơn vị gigabyte, từ bảy
cột đọc tự động từ thiết bị; `nhiet_do_thiet_bi_C` nhận −273 khi cảm biến mất kết nối,
`luu_luong_hom_truoc_GB` nhận 99999 khi bộ đếm tràn, và các cột **không** được chuẩn hóa. Chỉ ra ba kết
luận sai nhóm có thể rút ra từ bảng hệ số, mỗi kết luận gắn với một bước của Quy trình 5.3. (d) Nhóm đề
nghị bỏ qua việc làm sạch vì "R² vẫn trên 0,95": nêu đúng ba con số phải đo thêm kèm phép đo.

Tự kiểm: hệ số `nop_tre_tb_gio` đi từ −0,2079 sang +0,0490; lệch R² hai lần là −0,0137 và −0,0483.

**Bài 5.14** (★★★ · MTS4 · ~180 phút cho cả nhóm · Kiến thức: mục 5.4.3, 5.5.1, 5.5.2 · dạng: làm nhóm ba người, có trình bày, Kiến trúc)

(a) Lập lại bảng ba con số của mục 5.4.3 nhưng lấy **cây chặn quá khớp** làm mô hình chuẩn. Một dòng có
số dự đoán đổi nhỏ hơn 603: chỉ ra dòng đó và giải thích vì sao riêng nó khác các dòng còn lại. (b) Vẽ
lại biểu đồ cột R² theo ba quy ước của mục 5.5.2. Thanh khoảng vẽ kèm phải là khoảng của **hiệu** giữa
rừng ngẫu nhiên và cây chặn quá khớp, đo trên năm lần chia tập; nêu kết luận đọc được từ hình mà bảng
số che mất, và nêu một kết luận sai mà người xem sẽ rút ra nếu thanh đó là dải dao động R² của riêng
cột cây chặn quá khớp.

(c) Văn phòng kiến trúc Thiên Đức dự đoán giá thuê một tháng của căn hộ tại Hà Nội, đơn vị triệu đồng, đạt
MAE 1,45 trên thang giá từ 4 tới 48. Một nhóm khác báo cáo MAE 0,92 cho bài toán dự đoán điểm cuối kỳ và
kết luận mô hình của họ tốt hơn. Nêu lý do kết luận đó vô nghĩa kèm một con số của mục 5.4.1, và nêu độ
đo duy nhất so được hai bài toán. Mô hình giá thuê còn trả về 37 giá trị âm trên 1.200 căn: nêu phép xử
lý của Mã 5.9, hai con số phải báo cùng kết quả, và một tình huống khiến phép xử lý đó làm MAE xấu đi.

(d) Chia ba phần trên cho ba người. Nộp mã nguồn chạy lại được, hai hình, bảng kết quả và bài trình bày
7 phút trả lời một câu hỏi: trong ba khoảng giá thấp, giữa, cao, mô hình phục vụ khoảng nào kém nhất và
căn cứ nào cho biết điều đó. Căn cứ phải là cột MAE theo nhóm giá **thực**, không phải cột phần dư trung
bình theo nhóm giá thực; nêu lý do loại bỏ cột thứ hai, dựa vào đoạn Giải thích của mục 5.5.1.

Tự kiểm: dòng khác biệt có 574 trên 603 dự đoán đổi; rừng hơn cây chặn quá khớp 0,0885 điểm R² trên lần
chia của chương, và hiệu dương cả 5/5 lần chia ở phần (b), từ +0,0800 tới +0,0957.

### Thử thách có AI

**Bài 5.15** (★★★★ · MTS4 · ~90 phút · Kiến thức: toàn chương · dạng: giải tay trước rồi phát hiện lỗi trong báo cáo của AI)

**Bài toán.** Một nhóm nhờ trợ lý AI tạo sinh viết chương trình so hai mô hình hồi quy trên gói dữ liệu
Chương 4 bàn giao, rồi nhận về đoạn mã dưới đây cùng báo cáo hai câu: "cây quyết định không giới hạn độ
sâu là mô hình tốt nhất, R² đạt 1,0000" và "một dòng mang giá trị đánh dấu 99999 không ảnh hưởng tới mô
hình, vì lệch R² chỉ −0,0058".

```python
mo_tt = Pipeline([("tien", clone(tien)),
                  ("mo", LinearRegression(normalize=True))])
mo_tt.fit(X_tr, y_tr)
cay = Pipeline([("tien", clone(tien)),
                ("cay", DecisionTreeRegressor(criterion="mse",
                                              random_state=42))])
cay.fit(X_tr, y_tr)
for ten, mo in (("tuyến tính", mo_tt), ("cây", cay)):
    d = mo.predict(X_tr)
    print("%s: RMSE %.4f | R2 %.4f"
          % (ten, mean_squared_error(y_tr, d, squared=False),
             r2_score(y_tr, d)))

X2 = pd.concat([X_tr, X_tr.iloc[[0]]], ignore_index=True)
X2.loc[len(X2) - 1, "tong_thoi_luong_phut"] = 99999
y2 = pd.concat([y_tr, pd.Series([0.0])], ignore_index=True)
mo2 = Pipeline([("tien", clone(tien)),
                ("mo", LinearRegression())]).fit(X2, y2)
print("lệch R2 sau khi thêm một dòng: %+.4f"
      % (r2_score(y_te, mo2.predict(X_te))
         - r2_score(y_te, mo_tt.predict(X_te))))
```

**Nhiệm vụ với AI.** Ba bước theo đúng thứ tự. Bước một: **chưa** dùng công cụ AI, tự đọc đoạn mã rồi
viết và niêm phong danh sách lỗi dự đoán kèm lý do. Bước hai: đưa đoạn mã và hai câu kết luận cho một
công cụ AI tạo sinh, yêu cầu nó liệt kê mọi lỗi phương pháp và mọi lỗi phiên bản thư viện. Bước ba:
chạy thật để xác định danh sách đúng.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi
ứng với từng lỗi, cùng bản niêm phong của bước một. Chỉ dán tên cột, kiểu cột và số mức.

**Kiểm chứng.** Đoạn mã chứa **năm** lỗi thuộc hai loại. Ba lỗi làm chương trình dừng: nộp thông báo lỗi
nguyên văn, tên hàm hoặc tham số thay thế trong phiên bản thư viện mà chương khai báo, và dòng đã sửa.
Với thông báo lỗi có liệt kê một tập giá trị hợp lệ, thứ tự các giá trị trong tập **đổi theo từng lần
chạy** vì nó là `set` của Python; bài chấm theo tên lớp, tên tham số và giá trị sai, không theo thứ tự.
Hai lỗi còn lại không sinh thông báo nào mà làm kết luận sai. Với lỗi thứ nhất, nộp cặp số đo trên hai
tập cho cả hai mô hình và cho biết con số 1,0000 nói gì theo mục 5.3.3. Với lỗi thứ hai, nộp đủ ba con
số của mục 5.4.3 cộng bảng hệ số trước và sau, rồi viết thành lời phát biểu sai mà mô hình thứ hai đưa
ra về cột `tong_thoi_luong_phut`. Cuối cùng so ba danh sách: lỗi nào tự tìm được mà công cụ bỏ sót, lỗi
nào ngược lại, lỗi nào công cụ báo oan.

**Phản tư** (100 tới 200 từ). Công cụ AI giúp được gì và không giúp được gì khi soát một chương trình
học máy; vì sao tự viết danh sách trước khi hỏi công cụ có thể thay đổi hẳn chất lượng kết quả; rủi ro khi một mô
hình xếp hạng sinh viên theo điểm dự kiến được kết luận đạt yêu cầu bằng con số đo trên tập huấn luyện.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất lượng
kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi tiết ở
tài liệu hướng dẫn chấm đi kèm sách.
