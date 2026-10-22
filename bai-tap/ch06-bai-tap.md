# Bài tập – Chương 6

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Mọi bài làm trên bảng 3.012 dòng của Chương 4, chia `test_size=0.2, random_state=42, stratify=y`, lớp
dương là "Không đạt"; từ tầng hai trở lên nộp mã nguồn và kết quả in ra. Mọi so sánh dạng "có đổi hay
không" phải in đủ **ba con số**: số ca đổi nhãn trên tổng, mức lệch lớn nhất, lệch của độ đo tổng hợp.
Câu hỏi "hai mô hình có khác nhau thật không" phải đo chính cái **hiệu** qua nhiều lần chia, đọc theo
dấu.

### Củng cố

**Bài 6.1** (★ · MTS4 · ~12 phút · Kiến thức: mục 6.2.3, 6.2.4, 6.4.2, 6.5.1 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một mô hình đạt accuracy 0,8441 và recall 0,1389 trên tập kiểm tra 603 dòng có 108 dòng lớp dương.
Kết luận hợp lý nhất: A. dùng được, vì accuracy vượt mốc luôn đoán lớp đa số; B. không dùng được cho
mục đích hỗ trợ sớm, vì recall thấp hơn cả mốc lấy ngẫu nhiên theo tỉ lệ lớp; C. lớp dương bị chọn
ngược; D. `max_iter` quá nhỏ.

(2) `RocCurveDisplay.from_estimator(mo, X_te, y_te, ax=o, color="black")` dừng với `TypeError`. Cách
viết đúng theo phiên bản thư viện sách khai: A. đổi `color` thành `c`;
B. `curve_kwargs={"color": "black"}`; C. quay lại `plot_roc_curve`; D. `from_predictions(y_te, ps,
color="black")`.

(3) Cần **một** con số để kiểm tra kết quả cuối cùng của hệ thống cảnh báo sớm, lớp dương 17,93%, một ca bỏ sót đắt hơn hẳn
một ca báo động sai. Độ đo nên dùng: A. accuracy; B. AUC; C. recall của lớp dương, báo kèm precision;
D. `weighted avg` của F1.

**Bài 6.2** (★ · MTS4 · ~12 phút · Kiến thức: mục 6.1.2, 6.2.3, 6.2.4, 6.3.2 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số. (1) Accuracy trên tập huấn luyện
bằng đúng 1,0000 luôn là dấu hiệu quá khớp. (2) Muốn phạt các trọng số mạnh hơn thì phải tăng `C`.
(3) Hạ ngưỡng quyết định trên một vectơ xác suất cố định không bao giờ làm recall giảm. (4) Một mô hình
đạt precision 1,0000 chắc chắn hữu ích hơn một mô hình đạt precision 0,7789 trên cùng tập kiểm tra.

**Bài 6.3** (★ · MTS4 · ~15 phút · Kiến thức: mục 6.3.1 · dạng: tính toán)

Mười hai sinh viên có tỉ lệ hoàn thành video 0,20 · 0,25 · 0,30 · 0,35 · 0,45 · 0,50 · 0,55 · 0,60 ·
0,70 · 0,75 · 0,85 · 0,90; bốn người mang nhãn "Không đạt" có tỉ lệ 0,20 · 0,25 · 0,30 · 0,45, tám
người còn lại mang nhãn "Đạt". (a) Tính chỉ số Gini của nút gốc. (b) Tính độ lợi thông tin của ba điểm
phân chia `<= 0,35`, `<= 0,45`, `<= 0,70` rồi cho biết thuật toán chọn điểm nào. (c) Nêu xác suất lớp
dương của hai lá ứng với điểm phân chia đã chọn và nhãn cây gán ở ngưỡng 0,5, kèm hai phép tự kiểm cho biết ba giá
trị độ lợi vừa tính không thể sai.

**Bài 6.4** (★ · MTS4 · ~15 phút · Kiến thức: mục 6.4.1, 6.4.2, 6.4.3 · dạng: tính toán)

Một trung tâm đào tạo trực tuyến khác chạy mô hình cảnh báo sớm trên 1.500 sinh viên và thu được ma
trận nhầm lẫn `[[1180, 95], [138, 87]]` theo quy ước `[[TN, FP], [FN, TP]]`. (a) Tính tỉ lệ lớp dương,
accuracy, precision, recall, $F_1$ từ bốn ô đó, rồi so accuracy ấy với accuracy của mốc luôn đoán lớp
đa số trên cùng 1.500 dòng. (b) Nêu khối lượng công việc theo bước 3 Quy trình 6.3, rồi áp bước 1 và 2
Quy trình 6.4 với giá 0,25 giờ cho một ca báo động sai và 45 giờ cho một ca bỏ sót để tính tổng thiệt
hại của mô hình và của mốc. (c) Cho biết hai phần trên dẫn tới hai kết luận khác nhau thế nào.

**Bài 6.5** (★ · MTS4 · ~15 phút · Kiến thức: mục 6.2.3, 6.4.1, 6.4.3 · dạng: đọc bảng)

Bảng dưới đây quét bảy mức ngưỡng trên **cùng một** vectơ xác suất của rừng ngẫu nhiên 200 cây, trên
603 dòng kiểm tra có 108 dòng lớp dương.

**Bảng cho Bài 6.5.** Bảy mức ngưỡng của rừng ngẫu nhiên.

| Ngưỡng | TP | FP | FN |
|---|---|---|---|
| 0,2 | 99 | 87 | 9 |
| 0,3 | 91 | 51 | 17 |
| 0,4 | 77 | 25 | 31 |
| 0,5 | 64 | 15 | 44 |
| 0,6 | 53 | 6 | 55 |
| 0,7 | 37 | 3 | 71 |
| 0,8 | 21 | 0 | 87 |

(a) Tính precision, recall, $F_1$ ở hai ngưỡng 0,4 và 0,6, rồi nêu ngưỡng cho $F_1$ lớn nhất trong
bảng. (b) Tính số ca báo động sai phải trả thêm cho mỗi ca bỏ sót bớt đi khi hạ ngưỡng từ 0,5 xuống 0,4
và từ 0,3 xuống 0,2, rồi so với tỉ giá tương ứng của Bảng 6.3. (c) Dòng ngưỡng 0,8 có FP bằng 0: tính
precision khi đó và nêu con số cho thấy vì sao nó không được báo cáo một mình. (d) Dòng ngưỡng 0,5 cho
FP bằng 15, còn Bảng 6.9 cho FP bằng 13 với đúng mô hình đó trên đúng tập kiểm tra đó: nêu đại lượng
phải đếm để tìm nguyên nhân.

**Bài 6.6** (★ · MTS4 · ~12 phút · Kiến thức: mục 6.1.1, 6.1.2, 6.1.3, 6.2.4 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, kèm một phép kiểm phát hiện được nó.

Dấu hiệu: (1) accuracy trên tập kiểm tra bằng 1,0000, hai ô ngoài đường chéo đều bằng 0; (2) accuracy
0,8209 còn recall bằng đúng 0, cột phải của ma trận nhầm lẫn toàn số 0; (3) `n_iter_` bằng đúng giá trị
`max_iter` đã khai, kèm `ConvergenceWarning`; (4) tỉ lệ lớp dương của tập kiểm tra chạy từ 15,75% tới
19,24% giữa các lần chạy.

Nguyên nhân: A. mô hình luôn trả lời lớp đa số; B. phép chia tập không truyền `stratify=y`; C. tập đặc
trưng còn một cột suy ra được từ nhãn; D. các cột số chưa chuẩn hóa z nên `lbfgs` chưa tới điểm tối ưu.

**Bài 6.7** (★ · MTS4 · ~15 phút · Kiến thức: mục 6.3.3, 6.5.2 · dạng: đọc hình)

Hình dưới đây là biểu đồ độ quan trọng hoán vị của **hồi quy logistic**, đo theo $F_1$ trên tập kiểm
tra, trung bình mười lần xáo trộn.

```
Mức F1 mất đi khi xáo trộn cột đó
(một ký tự # là 0,01; đoạn --- là một độ lệch chuẩn)
                        0    0,05   0,10   0,15   0,20
ty_le_hoan_thanh_video  |####################---     0,1952 ± 0,0300
diem_giua_ky            |################--          0,1568 ± 0,0233
diem_tb_bai_tap         |#########---                0,0871 ± 0,0261
hinh_thuc_hoc           |######-                     0,0567 ± 0,0090
so_bai_tap_nop          |######--                    0,0560 ± 0,0162
so_lan_dang_nhap        |#####-                      0,0481 ± 0,0140
tong_thoi_luong_phut    |#####-                      0,0472 ± 0,0087
vua_lam_vua_hoc         |###--                       0,0281 ± 0,0170
nop_tre_tb_gio          |##-                         0,0205 ± 0,0082
nhom_tuoi               |#-                          0,0140 ± 0,0084
gioi_tinh               |#-                          0,0122 ± 0,0072
so_lan_dien_dan         |#-                          0,0095 ± 0,0076
thiet_bi_chinh          |#-                          0,0091 ± 0,0060
nganh                  #|-                          -0,0062 ± 0,0109
```

(a) So với Bảng 6.8 của rừng ngẫu nhiên, chỉ ra cột đổi thứ hạng nhiều nhất, kèm cặp số chứng minh.
(b) Nêu mọi cột mà thanh độ lệch chuẩn phủ qua vạch 0, và kết luận đúng về những cột đó. (c) Giá trị
lớn nhất ở đây là 0,1952 còn ở Bảng 6.8 là 0,1274: cho biết hai con số đó có nói rằng hồi quy logistic
dựa vào `ty_le_hoan_thanh_video` nhiều hơn rừng ngẫu nhiên hay không, kèm lý do.

### Vận dụng

**Bài 6.8** (★★ · MTS4 · ~35 phút · Kiến thức: mục 6.2.3, 6.3.1, 6.4.1 · dạng: viết mã)

Mã 6.7 kiểm rằng `predict` trùng khít với `predict_proba >= 0,5` và in `True` cho hồi quy logistic.
(a) Chạy lại phép kiểm đó cho **rừng ngẫu nhiên 200 cây**, rồi lập bảng bốn ô và năm độ đo cho hai cách
gán nhãn `predict()` và `(predict_proba()[:, 1] >= 0.5)`. (b) In ba con số: số ca đổi nhãn trên tổng,
mức lệch lớn nhất giữa hai bảng độ đo kèm tên độ đo đó, lệch $F_1$. (c) Đếm số ca có xác suất lớp dương
**đúng bằng** 0,50, in nhãn thực và số phiếu trên 200 cây của từng ca. (d) Giải thích cơ chế làm hai
cách cho hai kết quả khác nhau, và lý do phép kiểm ấy luôn cho `True` với hồi quy logistic.

Tự kiểm: hồi quy logistic đổi 0/603 nhãn; rừng đổi 2/603, FP từ 13 lên 15, $F_1$ từ 0,6919 xuống
0,6845.

**Bài 6.9** (★★ · MTS4 · ~50 phút · Kiến thức: mục 6.1.2, 6.1.3, 6.4.4 · dạng: viết mã)

Đo xem sáu cột phân loại có đáng giữ trong tập đặc trưng hay không. (a) Dựng
`ColumnTransformer([("so", StandardScaler(), tám cột số)], remainder="drop")` rồi đi trọn Quy trình 6.1
cho `LogisticRegression(max_iter=1000)` và `RandomForestClassifier(n_estimators=200, random_state=42)`;
in số đặc trưng, hai dòng mốc của bước 3, ma trận nhầm lẫn và năm độ đo của mỗi mô hình. (b) In ba con
số của bản rút gọn so với bản 30 đặc trưng, cho từng mô hình. (c) Đo **hiệu** của recall, precision,
$F_1$ và AUC giữa hai bản trên năm lần chia với `random_state` 0 tới 4, mỗi lần khớp lại cả bộ tiền xử
lý; với từng độ đo cho biết hiệu có cùng dấu ở cả năm lần chia và có lần nào bằng 0 hay không. (d) Kết luận có bỏ sáu cột đó được hay không,
nêu độ đo nào cho kết luận được và độ đo nào không.

Tự kiểm: 8 đặc trưng; bản rút gọn của hồi quy logistic cho accuracy 0,9005 và recall 0,6204, ba con số
29/603 · 0,4870 · −0,0383; hiệu AUC âm cả 5/5 lần ở cả hai mô hình.

**Bài 6.10** (★★ · MTS4 · ~70 phút · Kiến thức: mục 6.2.3, 6.4.3, 6.4.4 · dạng: viết mã) **[HTTT]**

Phòng đào tạo xác định lại điều kiện triển khai: mỗi đợt chỉ gọi điện được cho **40** sinh viên, trong khi
mô hình của Ví dụ 6.3 gọi tên 95 người ở ngưỡng 0,5.

Xác suất của rừng là bội của 1/200 nên có ca bằng đúng ngưỡng thứ 40, và cách phá thế hòa đổi con số
đếm được. Cả bài dùng đúng một quy ước đã định sẵn: **`chon = np.argsort(-ps, kind="stable")[:40]`**, tức
sắp ổn định nên ca nào đứng trước trong tập kiểm tra thì được chọn trước. Bỏ `kind="stable"` đi là
`np.argsort` dùng thuật toán sắp **không ổn định** và kết quả đổi theo máy.

(a) Với mỗi mô hình, lấy 40 sinh viên có xác suất
lớp dương cao nhất rồi in số ca đúng trong 40, precision, recall và ngưỡng xác suất thấp nhất được
chọn. (b) Lặp lại trên mười lần chia phân tầng với `random_state` 0 tới 9, in hiệu số ca đúng của hồi
quy logistic trừ rừng ngẫu nhiên ở từng lần, rồi đếm số lần hiệu dương, âm, bằng 0. (c) Kết luận ở cùng
khối lượng công việc 40 cuộc gọi, so với kết luận của Bảng 6.11 ở ngưỡng 0,5, và nêu điều kiện áp dụng
mà phép so này buộc phải thêm vào kết luận của Ví dụ 6.3. (d) Nêu con số cho thấy vì sao giới hạn 40
cuộc gọi làm recall của cả hai mô hình giảm xuống dưới một nửa, và việc phòng đào tạo phải làm với phần
sinh viên không được gọi. (e) Với từng lần chia, in số ca của rừng có xác suất **đúng bằng** ngưỡng
thứ 40, rồi cho biết ở những lần chia nào con số của phần (b) sẽ đổi nếu dùng một cách phá thế hòa
khác, và vì sao một đề bài đòi đếm "k ca cao nhất" buộc phải quy định câu lệnh sắp xếp.

Tự kiểm: hồi quy logistic 35 ca đúng trong 40, rừng 37; hiệu ở phần (b) âm 5 lần, bằng 0 bốn lần, dương
1 lần, tổng −13. Hai con số này chỉ đúng với `kind="stable"`.

**Bài 6.11** (★★ · MTS4 · ~40 phút · Kiến thức: mục 6.2.4, 6.3.1, 6.4.4 · dạng: viết mã)

Mục 6.2.4 đo phép bỏ chuẩn hóa z với hồi quy logistic và được ba con số 10/603 · 0,1819 · −0,0226. Làm
lại đúng phép thử đó cho **rừng ngẫu nhiên 200 cây**. (a) In ba con số và hai ma trận nhầm lẫn. (b) In
thêm ba con số kiểm điều kiện: số giá trị khác nhau của mỗi cột số **trên tập huấn luyện** trước và sau chuẩn hóa z; thứ hạng
các dòng có giữ nguyên hay không; lệch xác suất lớn nhất quy ra số phiếu trên 200 cây. (c) Cho biết
điều kiện phân biệt của mục 6.2.4 dự đoán đúng kết quả này hay không, rồi dựng **hai ca thử** cho nó:
một mô hình của sách **có** tính chất mà điều kiện nêu và dự đoán vẫn đổi, một mô hình **không có**
tính chất đó mà dự đoán bất biến. Cho biết điều kiện cũ, "mô hình có chính quy hóa hay không", trượt
ở ca nào. (d) Hai ca đổi nhãn ở phần (a) nằm đúng chỗ mà Bài 6.8
vừa đo: chỉ ra chỗ đó và nêu vì sao $F_1$ lệch đúng 0,0000 trong khi hai sinh viên nhận hai kết luận
khác nhau.

Tự kiểm: đổi 2/603 nhãn · lệch xác suất lớn nhất 0,0200 · lệch $F_1$ +0,0000; hai ma trận nhầm lẫn
giống hệt nhau.

### Thực tế và liên ngành

**Bài 6.12** (★★★ · MTS4 · ~45 phút · Kiến thức: mục 6.1.3, 6.4.2, 6.4.3, 6.5.2 · dạng: bảo vệ phương án, Công nghệ sinh học)

(a) Khớp sáu mô hình: mốc luôn đoán "Đạt", hồi quy logistic với `C=1` và với `C=0.001`, cây quyết định
không giới hạn, cây sâu 5 tầng, rừng 200 cây. Lập bảng gồm bốn ô ma trận nhầm lẫn, năm độ đo và **ba
cột thứ hạng** theo accuracy, theo recall, theo precision; chỉ ra mô hình đổi thứ hạng nhiều nhất giữa
ba cách xếp, kèm con số chứng minh.

(b) Phòng kiểm nghiệm của Công ty Công nghệ sinh học Minh An sàng lọc 50.000 mẫu mỗi năm để tìm một
tác nhân có mặt trong 1,2% số mẫu. Một báo cáo kết luận: "mô hình đạt độ chính xác 98,8% nên đủ điều
kiện thay thế quy trình xét nghiệm hiện hành". Tính số mẫu dương thật trong một năm và accuracy của mô
hình luôn trả lời "âm", rồi viết **ba** câu hỏi, mỗi câu đủ bác bỏ kết luận trên và kèm một con số của
phần (a) làm căn cứ.

(c) Đi bước 1 Quy trình 6.1 và bước 1, bước 2 Quy trình 6.4 cho bài toán sàng lọc: chốt lớp dương, nêu
hai loại sai bằng ngôn ngữ của phòng kiểm nghiệm, chọn một độ đo chính kèm một độ đo phụ, nêu giả thiết
về giá hai loại sai. Cho biết mô hình `C=0.001` đứng hạng mấy theo precision và vì sao thứ hạng đó
không cứu được nó ở bài toán này.

(d) Bảo vệ phương án trong 5 phút trước hội đồng kỹ thuật, kèm một câu trả lời cho ý kiến "biểu đồ cột
accuracy nên cắt trục dọc từ 0,95 tới 1,00 cho dễ nhìn".

**Bài 6.13** (★★★ · MTS4 · ~50 phút · Kiến thức: mục 6.1.1, 6.3.3, 6.4.4 · dạng: tình huống nghiệp vụ và vấn đáp, Luật)

(a) Mã 6.1 giữ một cột suy thẳng ra nhãn nên accuracy 1,0000 làm lỗi lộ ra ngay. Dựng một ca khó phát
hiện hơn: thêm cột phân loại `canh_bao_giang_vien` hai mức "Có" và "Không", tạo bằng cách lấy nhãn thật
rồi đảo giá trị ở 15% số dòng chọn bằng `rng = np.random.default_rng(2026)` và
`vi_tri = rng.choice(len(bang), size=int(0.15 * len(bang)), replace=False)`. Đưa cột này vào nhóm cột
phân loại của bộ tiền xử lý, khớp lại hai mô hình của chương, in năm độ đo và ma trận nhầm lẫn của mỗi
mô hình cạnh bộ số của Bảng 6.9. In thêm **ba con số** của phép thêm cột, cho từng mô hình.

(b) Nêu lý do bảng độ đo ở phần (a) không tự báo động, kèm con số. Đo độ quan trọng hoán vị theo $F_1$
trên 15 cột gốc và cho biết đại lượng nào phát hiện được cột thêm vào, kèm tỉ số giữa giá trị của nó và
giá trị của cột đứng thứ hai.

(c) Trung tâm trợ giúp pháp lý Hoa Sen dùng một mô hình phân loại để xếp hồ sơ xin trợ giúp vào nhóm
"có dấu hiệu khai không trung thực" rồi chuyển nhóm đó sang thẩm tra bổ sung. Áp ba điều kiện của mục
6.4.4, mỗi điều kiện kèm một con số của chương làm căn cứ, và cho biết mô hình nào của chương thỏa yêu
cầu giải thích từng ca.

(d) Chuẩn bị trả lời vấn đáp 3 phút: dữ liệu của trung tâm có một cột ghi kết quả thẩm tra những năm
trước. Cho biết cột đó được dùng làm đặc trưng hay không, kèm phép kiểm duy nhất quyết định được điều
đó theo khối Chú ý của mục 6.1.1.

Tự kiểm: rừng ngẫu nhiên ở phần (a) cho accuracy 0,9552 · recall 0,8241 · AUC 0,9766, và ba con số
của nó là 58/603 · 0,5350 · +0,1764; độ quan trọng hoán vị của cột thêm vào là 0,4042.

**Bài 6.14** (★★★ · MTS4 · ~180 phút cho cả nhóm · Kiến thức: mục 6.2.3, 6.4.1, 6.4.4, 6.5.1 · dạng: làm nhóm ba người, có trình bày, Du lịch và Khách sạn)

(a) Áp bước 1 và 2 Quy trình 6.4 cho hai mô hình của Bảng 6.9 với cặp giá 150.000 đồng cho một ca báo
động sai và 1.200.000 đồng cho một ca bỏ sót: in tổng thiệt hại của mỗi mô hình ở ngưỡng 0,5, rồi tính
**tỉ lệ giá hòa vốn**, tức tỉ số giữa giá một ca bỏ sót và giá một ca báo động sai mà tại đó hai mô
hình thiệt hại bằng nhau. Nêu phép tự kiểm của bước 2 mà con số này thực hiện.

(b) Lặp lại phần (a) trên mười lần chia phân tầng với `random_state` 0 tới 9, in hiệu thiệt hại của hồi
quy logistic trừ rừng ngẫu nhiên ở từng lần cùng dải tỉ lệ giá hòa vốn. Vẽ biểu đồ cột so thiệt hại hai
mô hình theo ba quy ước của mục 6.5.2, thanh khoảng vẽ kèm là khoảng của **hiệu** thiệt hại. Nêu kết
luận đọc được từ hình mà bảng số che mất, và một kết luận sai mà người xem rút ra nếu thanh đó là dải
thiệt hại của riêng một mô hình.

(c) Khách sạn Thiên Đức dự báo khách đặt phòng sẽ hủy vào phút cuối để bán lại chỗ trống: một đêm
phòng trống mất 1.200.000 đồng, một lần báo hủy sai làm bán trùng chỗ và phải dàn xếp, mất 150.000
đồng. Dùng đúng bảy dòng của Bảng 6.3, tính tổng thiệt hại ở từng ngưỡng, nêu ngưỡng tốt nhất trong
bảng, rồi chỉ ra điều mà kết quả này nói về bản thân Bảng 6.3 và việc phải làm tiếp.

(d) Chia ba phần trên cho ba người. Nộp mã nguồn chạy lại được, một hình, bảng kết quả và bài trình bày
7 phút trả lời một câu hỏi: nếu khách sạn chỉ được chọn **một** ngưỡng cố định cho cả năm thì chọn
ngưỡng nào. Căn cứ phải là tổng thiệt hại theo ngưỡng, không phải cột $F_1$; nêu lý do loại cột $F_1$
dựa vào khối Chú ý của mục 6.4.3.

Tự kiểm: ở ngưỡng 0,5 hai mô hình thiệt hại 43,95 và 54,75 triệu đồng; tỉ lệ giá hòa vốn của lần chia
chính là 0,80; hiệu thiệt hại ở phần (b) âm cả 10/10 lần, từ −13,20 tới −3,90 triệu đồng.

### Thử thách có AI

**Bài 6.15** (★★★★ · MTS4 · ~90 phút · Kiến thức: toàn chương · dạng: niêm phong dự đoán, hỏi AI, chạy thật, rồi phát hiện lỗi)

**Bài toán.** Một nhóm nhờ trợ lý AI tạo sinh viết chương trình chọn mô hình phân loại cho hệ thống
cảnh báo sớm, và nhận về đoạn mã dưới đây cùng báo cáo ba câu: "rừng ngẫu nhiên đạt accuracy 1,0000 nên
không cần dòng mốc"; "hồi quy logistic được đặt `C=0.001` để chống quá khớp mà accuracy vẫn trên 0,84";
"ngưỡng tốt nhất là 0,29 với F1 0,7928, đây là hiệu năng của mô hình khi triển khai". Phần nhập thư
viện và các dòng nạp `bang`, `COT_SO`, `COT_PL` giữ nguyên như các chương trình của chương, không chứa
lỗi nào.

```python
def tien_xu_ly():
    return ColumnTransformer([
        ("so", StandardScaler(), COT_SO + ["diem_cuoi_ky"]),
        ("pl", OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False), COT_PL)])

X = bang.drop(columns=["ma_sv", "ngay_dang_ky", "tinh_thanh",
                       "ket_qua"])
y = (bang["ket_qua"] == "Không đạt").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42,
                                          stratify=y)
rf = Pipeline([("tien", tien_xu_ly()),
               ("mo_hinh", RandomForestClassifier(
                   n_estimators=200,
                   random_state=42))]).fit(X_tr, y_tr)
print("accuracy rừng %.4f" % accuracy_score(y_te, rf.predict(X_te)))
lg = Pipeline([("tien", tien_xu_ly()),
               ("mo_hinh", LogisticRegression(
                   penalty="l1", C=0.001,
                   max_iter=1000))]).fit(X_tr, y_tr)
ps = lg.predict_proba(X_te)[:, 1]
tot = max(np.arange(0.05, 0.95, 0.01),
          key=lambda ng: f1_score(y_te, (ps >= ng).astype(int)))
print("ngưỡng tốt nhất %.2f | F1 %.4f"
      % (tot, f1_score(y_te, (ps >= tot).astype(int))))
fig, o = plt.subplots()
RocCurveDisplay.from_estimator(rf, X_te, y_te, ax=o, color="black")
print("recall %.4f"
      % recall_score(bang.loc[X_te.index, "ket_qua"],
                     np.where(rf.predict(X_te) == 1,
                              "Không đạt", "Đạt")))
```

**Nhiệm vụ với AI.** Ba bước theo đúng thứ tự. Bước một: **chưa** dùng công cụ AI, tự đọc đoạn mã và
ba câu báo cáo, rồi viết và niêm phong danh sách lỗi dự đoán kèm con số dự đoán cho từng độ đo sẽ nhận
được sau khi sửa. Bước hai: đưa đoạn mã và ba câu báo cáo cho một công cụ AI tạo sinh, yêu cầu nó liệt
kê mọi lỗi phương pháp và mọi lỗi phiên bản thư viện. Bước ba: chạy thật, sửa lần lượt từng lỗi, rồi
so ba kết quả.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi
ứng với từng lỗi, cùng bản niêm phong của bước một. Chỉ dán tên cột, kiểu cột và số mức.

**Kiểm chứng.** Đoạn mã và báo cáo chứa **sáu** lỗi thuộc hai loại. Ba lỗi làm chương trình dừng: nộp
thông báo lỗi nguyên văn, tên hàm hoặc tham số thay thế trong phiên bản thư viện sách khai, và dòng đã
sửa; một trong ba lỗi còn in một cảnh báo trước khi dừng, nộp cả cảnh báo đó. Ba lỗi còn lại không làm
chương trình dừng mà làm kết luận sai, và chúng **chồng lên nhau**: phải sửa lỗi thứ nhất mới đo được
hậu quả của hai lỗi sau. Với từng lỗi im lặng, nộp bộ số trước và sau khi sửa, dòng mốc tương ứng của
mục 6.1.3, và câu kết luận sai mà báo cáo đưa ra. Với lỗi chọn ngưỡng, phần sửa phải **khai giao thức
chia tập kiểm định của chính mình** (tỉ lệ, `random_state`, có phân tầng hay không) rồi nộp con số đo
được kèm phần lạm phát so với cách quét trên tập kiểm tra; con số ấy đổi theo giao thức nên bài làm
được chấm theo giao thức và theo phần lạm phát, không theo một giá trị cho trước. Với mỗi lỗi dừng,
phải chạy thử **cả dòng đã sửa** chứ không chỉ dòng sai: một trong ba lỗi dừng có cách sửa hiển nhiên
mà vẫn cho lại đúng thông báo cũ. Cuối cùng so ba danh sách: lỗi nào tự tìm được mà
công cụ bỏ sót, lỗi nào ngược lại, lỗi nào công cụ báo oan.

**Phản tư** (100 tới 200 từ). Công cụ AI giúp được gì và không giúp được gì khi soát một chương trình
phân loại; vì sao niêm phong dự đoán trước khi hỏi công cụ có thể thay đổi hẳn chất lượng kết quả; rủi ro khi một
danh sách sinh viên có nguy cơ được kết luận đạt yêu cầu bằng một con số accuracy, kèm trách nhiệm với 34 ca bỏ
sót và 21 ca báo động sai của mô hình được chọn.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất
lượng kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi
tiết ở tài liệu hướng dẫn chấm đi kèm sách.
