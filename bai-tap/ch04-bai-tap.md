# Bài tập – Chương 4

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Trừ khi đề nêu nguồn khác, mọi bài làm trên `data/processed/du-lieu-sach.csv`, giải bằng bốn quy trình
của chương; từ tầng hai trở lên nộp mã và kết quả in ra.

### Củng cố

**Bài 4.1** (★ · MTS3 · ~10 phút · Kiến thức: mục 4.1.3, 4.2.2 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một cột chi phí vận chuyển có phân vị 75% bằng 0,0162 sau khi chuẩn hóa về khoảng $[0, 1]$; mô
hình sắp dùng là k lân cận gần nhất. Theo bước 3 Quy trình 4.1: A. giữ `MinMaxScaler`; B. bỏ bước
chuẩn hóa, vì cột đã nằm trong $[0, 1]$; C. đổi sang `RobustScaler`; D. khai `clip=True`.

(2) Một dịch vụ dự đoán nhận dòng mới có mức phân loại chưa từng thấy. `pd.get_dummies` không dùng
được, vì: A. nó trả cột kiểu `bool`; B. nó không lưu danh sách mức đã thấy, nên số cột của dòng mới
khác số cột lúc huấn luyện; C. nó chỉ chạy trên bảng nhỏ; D. nó không có tham số `drop`.

**Bài 4.2** (★ · MTS3 · ~10 phút · Kiến thức: mục 4.1.2, 4.1.3, 4.3.2, 4.3.3 · dạng: đúng hoặc sai, có giải thích)

Mỗi phát biểu đúng hay sai, kèm một câu giải thích có số. (1) Sau khi chuẩn hóa z,
một cột lệch mạnh về bên phải trở thành cột có phân bố hình chuông. (2) Rừng ngẫu nhiên và cây quyết
định không cần chuẩn hóa thang đo. (3) Khai `stratify=y` làm tỉ lệ lớp thiểu số của tập kiểm tra bằng
đúng tỉ lệ của cả bảng. (4) Rò rỉ dữ liệu luôn làm con số đánh giá cao hơn thật.

**Bài 4.3** (★ · MTS3 · ~15 phút · Kiến thức: mục 4.1.2 · dạng: tính toán)

Cột `so_ngay_cham_giao` của sáu đơn hàng nhận giá trị 0, 2, 3, 5, 8, 42; trung bình 10,0000, độ lệch
chuẩn trên cả sáu giá trị 14,5258, ba phân vị 25%, 50%, 75% là 2,2500, 4,0000, 7,2500. (a) Biến đổi cả
sáu giá trị bằng ba cách: về khoảng $[0, 1]$, chuẩn hóa z, và `RobustScaler` (trừ trung vị rồi
chia khoảng tứ phân vị). (b) Tính phân vị 75% sau mỗi cách, cho biết phép tự kiểm ở bước 3
Quy trình 4.1 có báo chọn sai bộ hay không, và cách nào giữ rõ nhất dấu hiệu đơn hàng 42 ngày.

**Bài 4.4** (★ · MTS3 · ~15 phút · Kiến thức: mục 4.1.4, 4.2.1, 4.2.4, 4.3.4 · dạng: ghép đôi)

Ghép mỗi triệu chứng với một nguyên nhân, kèm một phép kiểm phát hiện được nó.

Triệu chứng: (1) tập kiểm tra sau chuẩn hóa có hai ô lớn hơn 1, miền tới 1,1379; (2) tương quan giữa
cột mã và biến mục tiêu đổi dấu, từ +0,5428 thành −0,2172; (3) một dòng mới cho tổng theo nhóm cột một
nóng bằng 0 thay vì 1; (4) accuracy bằng 1,0000, ma trận nhầm lẫn không có ô sai nào.

Nguyên nhân: A. thư viện tự sắp thứ tự các mức theo bảng chữ cái; B. bộ chuẩn hóa khớp đúng trên tập
huấn luyện, hai giá trị lớn nhất lại rơi vào tập kiểm tra; C. một đặc trưng suy ra từ biến mục tiêu;
D. `handle_unknown="ignore"` gặp một mức chưa từng thấy.

**Bài 4.5** (★ · MTS3 · ~15 phút · Kiến thức: mục 4.1.2, 4.1.4 · dạng: đọc bảng tham số)

**Bảng cho Bài 4.5.** Sáu dãy tham số của ba bộ chuẩn hóa, khớp trên tập huấn luyện 2.409 dòng.

| Cột | (I) | (II) | (III) | (IV) | (V) | (VI) |
|---|---|---|---|---|---|---|
| `so_lan_dang_nhap` | 1,0 | 88,0 | 37,4259 | 15,1902 | 38,000 | 21,0 |
| `nop_tre_tb_gio` | 0,0 | 644,0 | 15,1895 | 21,2820 | 11,200 | 14,2 |

(a) Gán mỗi dãy từ (I) tới (VI) cho một thuộc tính trong `data_min_`, `data_max_`, `mean_`, `scale_`
của `StandardScaler`, `center_` và `scale_` của `RobustScaler`. (b) Tính giá trị mà
`nop_tre_tb_gio = 644` nhận được sau cả ba phép. (c) Một ô chứng minh ba bộ này **không** khớp trên cả
3.012 dòng: chỉ ra ô đó và giá trị đúng của nó khi khớp sai chỗ.

**Bài 4.6** (★ · MTS3 · ~10 phút · Kiến thức: mục 4.2.2, 4.3.1 · dạng: tính toán)

Sáu cột phân loại làm đặc trưng có số mức: `nganh` 7, `gioi_tinh` 3, `nhom_tuoi` 4,
`vua_lam_vua_hoc` 2, `hinh_thuc_hoc` 2, `thiet_bi_chinh` 4. (a) Tính số cột sau khi mã hóa một nóng cả
sáu cột; khi khai thêm `drop="first"`; và khi `nhom_tuoi` dùng mã hóa thứ tự. (b) Cộng tám cột số để
ra tổng số đặc trưng của phương án ở mục 4.4.1, rồi cho biết `drop="first"` làm mất bao nhiêu phần
trăm số cột.

**Bài 4.7** (★ · MTS3 · ~15 phút · Kiến thức: mục 4.3.3, 4.3.4 · dạng: giải thích có căn cứ số)

Ba sai sót sau đều không sinh thông báo lỗi. (1) Khớp `MinMaxScaler` trên cả bảng rồi mới chia tập.
(2) Giữ `diem_cuoi_ky` làm đặc trưng khi dự đoán `ket_qua`. (3) Chọn mô hình bằng chính con số đo trên
tập kiểm tra, qua 40 lượt thử. Với mỗi sai sót: xếp vào một trong hai họ rò rỉ ở mục 4.3.3 hoặc nêu rõ
nó ngoài hai họ đó; sắp cả ba theo mức nghiêm trọng tăng dần, mỗi mức kèm một con số của chương; nêu
một phép kiểm phát hiện được nó.

### Vận dụng

**Bài 4.8** (★★ · MTS3 · ~25 phút · Kiến thức: mục 4.1 · dạng: viết mã)

Đi trọn Quy trình 4.1 cho tám cột số với k lân cận gần nhất, so ba bộ chuẩn hóa thay vì một. (a) Chia
`test_size=0.2, random_state=42`, khớp lần lượt `MinMaxScaler`, `StandardScaler`, `RobustScaler`
**chỉ trên tập huấn luyện**, lập bảng bốn cột đo: phân vị 75% của `nop_tre_tb_gio` sau biến đổi, miền
giá trị của tập huấn luyện, số ô của tập kiểm tra ngoài miền đó, hệ số xác định của
`KNeighborsRegressor(n_neighbors=10)`. (b) Nêu bộ nào bị phép tự kiểm ở bước 3 báo chọn sai, kèm con
số. (c) Chỉ ra cột sinh ra hai ô vượt ngưỡng 1 ở phương án min–max, và vì sao không được sửa bằng
cách gọi lại `fit_transform` trên tập kiểm tra.

Tự kiểm: ba phân vị 75% là 0,0307, 0,2166, 0,6056; miền tập kiểm tra sau min–max tới 1,1379.

**Bài 4.9** (★★ · MTS3 · ~30 phút · Kiến thức: mục 4.2 · dạng: viết mã)

(a) Đi Quy trình 4.2 cho cột `tinh_thanh` tới bước 5, rồi chạy
`OneHotEncoder(sparse_output=False, min_frequency=...)` với ba ngưỡng 50, 80, 120; lập bảng gồm ngưỡng,
số cột sinh ra, `infrequent_categories_`. Ngưỡng 80 gộp đúng một mức mà số cột không đổi: giải thích
con số đó. (b) Dựng cột thứ bậc `muc_bai_tap` từ `diem_tb_bai_tap` bằng `pd.cut`, mốc cắt 5,0; 6,5;
8,0 và bốn nhãn "Yếu", "Trung bình", "Khá", "Tốt". Mã hóa hai lần theo bước 3 Quy trình 4.2, khai thứ
tự bằng tay rồi để thư viện tự sắp, in tương quan của cột mã với `diem_cuoi_ky`. (c) Nêu thứ tự thư
viện tự sắp, mức bị đặt sai chỗ, và kết luận mà một mô hình tuyến tính học trên cột mã sai sẽ rút ra.

Tự kiểm: ngưỡng 50 không gộp mức nào, ngưỡng 120 gộp 6 mức còn 11 cột; bốn mức có 559, 855, 934, 664
dòng; hai hệ số tương quan +0,5428 và −0,2172.

**Bài 4.10** (★★ · MTS3, MTS4 · ~25 phút · Kiến thức: mục 4.3.1, 4.3.2 · dạng: viết mã)

(a) Đi Quy trình 4.3 chia 3.012 dòng thành 70/15/15 bằng hai lần gọi `train_test_split`, phân tầng
theo `ket_qua` ở **cả hai lần**, `random_state=42`; in cho mỗi tập số dòng, tỉ lệ trên tổng, số dòng
và tỉ lệ lớp "Không đạt". (b) Làm lại lần chia thứ nhất mười lần với giá trị khởi tạo 0 tới 9, một lượt có phân
tầng và một lượt không, in độ rộng dao động của tỉ lệ lớp đó. (c) Vì sao bỏ `stratify` ở lần chia thứ
hai vẫn làm sai kết quả. (d) Nêu phép tự kiểm ở bước 5 Quy trình 4.3 và kết quả của nó.

Tự kiểm: ba tập có 2.108, 452, 452 dòng; tỉ lệ lớp "Không đạt" 17,93%, 17,92%, 17,92%; không phân tầng
thì dải dao động rộng 5,31 điểm phần trăm.

**Bài 4.11** (★★ · MTS3, MTS4 · ~30 phút · Kiến thức: mục 4.3, 4.4 · dạng: viết mã) **[HTTT]**

Mục 4.4.1 lưu bộ tiền xử lý cho bài toán hồi quy. Làm gói tương ứng cho bài toán phân loại `ket_qua`,
ghi vào thư mục tạm riêng để không đè file của chương. (a) Đi Quy trình 4.4 chọn tập đặc trưng, rồi
Quy trình 4.3 chia `test_size=0.2, random_state=42, stratify=y`. (b) Dựng `ColumnTransformer` gồm
`StandardScaler` cho tám cột số và `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` cho
sáu cột phân loại, khớp trên tập huấn luyện, rồi ghi bốn file Parquet, một file `.joblib` và bản ghi
các bước dạng JSON. (c) Viết hàm kiểm tra kết quả cuối cùng, in ba phép kiểm: bộ nạp lại cho kết quả trùng khít bộ
trong bộ nhớ; số đặc trưng sau biến đổi; trị tuyệt đối lớn nhất trong tám trung bình của tập kiểm tra.
(d) Đếm số dòng mức `"Khác"` của `gioi_tinh` trong tập huấn luyện và đối chiếu mốc cuối mục 4.2.3.

Tự kiểm: 2.409 và 603 dòng; 30 đặc trưng; trung bình lớn nhất theo trị tuyệt đối 0,0424; mức
`"Khác"` còn 42 dòng.

### Thực tế và liên ngành

**Bài 4.12** (★★★ · MTS3 · ~40 phút · Kiến thức: mục 4.3.4, 4.1.3 · dạng: bảo vệ phương án, Kế toán)

(a) Đi Quy trình 4.4 cho bài toán hồi quy `diem_cuoi_ky` trên cả 19 cột: bước 1 in tỉ lệ giá trị khác
nhau của từng cột; bước 3 lập bảng chéo giữa `ket_qua` và phép so `diem_cuoi_ky >= 4,0`; bước 4 đo hệ
số xác định trên cả hai tập ở ba phương án đặc trưng, gồm không có `ket_qua`, có thêm `ket_qua`, có
thêm `ma_sv`. (b) Cột `ket_qua` chỉ mang hai mức, thêm nó vào lại làm hệ số xác định trên tập kiểm tra
tăng 0,0600: giải thích bằng quan hệ giữa hai cột, và vì sao mức tăng đó là dấu hiệu lỗi.

(c) Phòng kế toán Công ty TNHH Minh An giao bảng 42.000 chứng từ, mục tiêu dự đoán
`tong_tien_sau_thue`. Các cột: `ma_chung_tu` 42.000 mức; `ma_khach_hang` 5.800 mức;
`tong_tien_truoc_thue` 0 tới 3,2 tỉ đồng; `thue_suat` nhận 0, 0,05 hoặc 0,08; `so_dong_hang` 1 tới
240; `da_thanh_toan` chỉ được ghi sau khi chứng từ đã tất toán. Đi Quy trình 4.4 cho bảng này, mỗi cột
bị bỏ kèm một dòng lý do có số; rồi đi Quy trình 4.1 chọn bộ chuẩn hóa cho các cột số còn lại.
(d) Trưởng phòng kế toán hỏi vì sao bỏ một cột có tương quan 1,00 với mục tiêu. Viết ba câu trả lời,
mỗi câu một luận điểm, dùng một con số của phần (a).

**Bài 4.13** (★★★ · MTS3 · ~40 phút · Kiến thức: mục 4.2.3, 4.2.4 · dạng: tình huống doanh nghiệp, Du lịch và Khách sạn)

(a) Mục 4.2.3 bỏ cột `tinh_thanh` 16 mức. Thử phương án thứ ba: gộp theo nghĩa nghiệp vụ bằng bảng tra
`data/raw/tinh-thanh.csv` thành cột `vung`, mức thiếu đặt tên `"Không rõ"`. In số mức và số dòng mỗi
vùng, đo hệ số xác định của hồi quy tuyến tính ở ba phương án: không có cột nơi học, có `tinh_thanh`,
có `vung`; kết luận một câu. (b) Nêu một điều mà cách gộp theo nghĩa làm được và `min_frequency` ở
Bài 4.9 không làm được.

(c) Công ty du lịch Hoa Sen giao bảng 56.000 lượt đặt tour, mục tiêu dự đoán khách có hủy tour hay
không. Bốn cột phân loại: `diem_den` 180 mức, 96 mức trong đó dưới 50 lượt; `hang_khach_san` năm mức
từ 1 tới 5 sao; `kenh_dat` 6 mức; `quoc_tich` 71 mức. Mỗi tháng mở thêm 3 tới 5 điểm đến mới. Với từng
cột, đi Quy trình 4.2 và nêu: loại biến, cách mã hóa, giá trị `handle_unknown`, số cột sinh ra; rồi
tính tổng số cột phân loại. (d) Hệ thống trả dự đoán cho một điểm đến mới mở: nêu dấu hiệu quan sát
được trên dòng dữ liệu sau biến đổi, và một phép kiểm đặt trước lời gọi `transform`.

**Bài 4.14** (★★★ · MTS3, MTS4 · ~90 phút · Kiến thức: mục 4.3.2, 4.3.3 · dạng: làm nhóm ba người, có trình bày, Công nghệ sinh học)

(a) Mục 4.3.3 đo rằng 6 trong 12 mã sinh viên còn trùng bị chia sang hai tập với `random_state=42`. Đo
lại với sáu giá trị khởi tạo 0 tới 5, rồi với `GroupShuffleSplit(test_size=0.2, random_state=42)` dùng
`groups=d["ma_sv"]`. Lập bảng kết quả và cho biết con số 6 có phải đặc tính của bộ dữ liệu hay không.

(b) Một phòng xét nghiệm của trung tâm công nghệ sinh học giao 18.600 bản ghi sinh hóa của 4.200 bệnh
nhân, mục tiêu phân loại nguy cơ. Số lần đo: 1.200 bệnh nhân đo 1 lần, 1.800 đo 3 lần, 1.200 đo 10
lần. Chia 80:20 theo dòng, chọn ngẫu nhiên độc lập từng dòng. Tính kỳ vọng số bệnh nhân có bản ghi ở
**cả hai** tập, theo từng nhóm rồi cộng lại; nêu cách chia đúng và một phép tự kiểm sau khi chia.

(c) Ba chỉ số sinh hóa có miền 0,2 tới 8,0; 30 tới 420; và 0,001 tới 0,049. Đi Quy trình 4.1 bước 1 và
bước 3: nêu tỉ số độ rộng miền, bộ chuẩn hóa chọn, và chi phí khi đưa bộ chuẩn hóa vào quy trình xét
nghiệm hằng ngày. (d) Ba phần trên chia cho ba người. Nộp mã nguồn chạy lại được, bảng kết quả, và bài
trình bày 7 phút trả lời một câu hỏi: nếu báo cáo kiểm tra kết quả cuối cùng ghi độ chính xác 96% thì con số đó
đáng tin ở mức nào.

Tự kiểm: sáu giá trị khởi tạo cho 1, 3, 7, 1, 8 và 4 mã bị chia hai tập; nhóm đo 10 lần góp khoảng 1.071 bệnh nhân.

### Thử thách có AI

**Bài 4.15** (★★★★ · MTS3, MTS4 · ~90 phút · Kiến thức: toàn chương · dạng: phát hiện lỗi trong mã do AI sinh và đặt lại đề bài cho AI)

**Bài toán.** Một trợ lý AI được yêu cầu "chuẩn bị dữ liệu và đo nhanh xem có dự đoán được `ket_qua`
hay không". Mã dưới đây là kết quả, chạy trên bảng đã nạp vào biến `d`. Mã chạy hết, không dừng, không
in cảnh báo, và in ra `accuracy = 1.0` cùng một ma trận nhầm lẫn không có ô sai nào. Người viết kết
luận rằng bài toán đã giải xong.

```python
COT_PL = ["nganh", "gioi_tinh", "nhom_tuoi", "vua_lam_vua_hoc",
          "hinh_thuc_hoc", "thiet_bi_chinh"]
X = pd.concat([d[COT_SO + ["diem_cuoi_ky"]],
               pd.get_dummies(d[COT_PL])], axis=1)
y = (d["ket_qua"] == "Không đạt").astype("int64")
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
cay = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
print("accuracy =", accuracy_score(y_te, cay.predict(X_te)))
print(confusion_matrix(y_te, cay.predict(X_te)))
```

**Nhiệm vụ với AI.** Hai vòng. Vòng một: đưa đoạn mã trên cho một công cụ AI tạo sinh, yêu cầu nó liệt
kê mọi lỗi phương pháp, không chỉ lỗi cú pháp. Vòng hai: chỉ đưa mô tả bộ dữ liệu, tên biến mục tiêu
và các điều kiện kiểm tra kết quả cuối cùng do sinh viên tự viết, rồi yêu cầu nó sinh lại chương trình từ đầu.

**Nhật ký prompt.** Nộp toàn văn các prompt của cả hai vòng, tên và phiên bản công cụ, ngày dùng,
trích đoạn phản hồi ứng với từng lỗi. Chỉ dán tên cột, kiểu cột và số mức, không dán dữ liệu thật.

**Kiểm chứng.** Tìm **bốn** lỗi, cả bốn thuộc loại chạy không báo lỗi mà kết quả sai. Mỗi lỗi nộp:
dòng mã sai, một phép kiểm phát hiện được nó, số đo trước và sau khi sửa, bản sửa theo quy trình của
chương. Riêng với `accuracy = 1.0`, nộp thêm bằng chứng cho biết cây đã học quy tắc nào: in đặc trưng
ở nút gốc và ngưỡng của nó, đổi ngưỡng về thang đo gốc, rồi đối chiếu định nghĩa cột `ket_qua`. So với
danh sách của vòng một: lỗi nào trợ lý AI tìm được, bỏ sót, báo oan. Hai bẫy phiên bản phải kiểm riêng
ở vòng hai: tham số bật mảng thường của `OneHotEncoder`, và kiểu dữ liệu `pd.get_dummies` trả về.

**Phản tư** (100 tới 200 từ). Trợ lý AI giúp được gì và không giúp được gì khi chuẩn bị dữ liệu; vì
sao một con số đánh giá cao là lý do để soát mã thay vì lý do kết thúc công việc; rủi ro khi một mô
hình dự đoán nguy cơ không đạt được kết luận đạt yêu cầu bằng một con số sai, xét từ phía sinh viên bị đánh giá.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất
lượng kiểm chứng phần trợ lý AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi
tiết ở tài liệu hướng dẫn chấm đi kèm sách.
