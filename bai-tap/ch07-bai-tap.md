# Bài tập – Chương 7

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Mọi bài làm trên bảng 3.012 dòng của Chương 4, chia `test_size=0.2, random_state=42, stratify=y`, lớp dương là “Không đạt”; từ tầng hai trở lên nộp mã nguồn và kết quả in ra. Mô hình mặc định là `LogisticRegression(max_iter=1000)` và `RandomForestClassifier(n_estimators=200, random_state=42)`; mọi phép xác thực chéo khai đủ `n_splits`, `shuffle`, `random_state`. Câu hỏi “thay đổi này có ảnh hưởng không” phải in đủ **ba con số**: số ca đổi nhãn trên tổng, mức lệch lớn nhất, lệch của độ đo tổng hợp. Câu hỏi “hai cấu hình có khác nhau thật không” cần đo chênh lệch ghép cặp theo từng phần, đọc độ lớn và độ bất định; số lần cùng dấu chỉ mô tả tính ổn định; không so chênh lệch trung bình với độ lệch chuẩn giữa các phần, và không xem hai khoảng một độ lệch chuẩn có chồng nhau hay không.

### Củng cố

**Bài 7.1** (★ · MTS5 · \~10 phút · Kiến thức: mục 7.3.2, 7.3.3, 7.4.3 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Dòng `KFold(n_splits=5, shuffle=False, random_state=42)` làm chương trình dừng ngay. Nguyên nhân: A. `n_splits` phải là số lẻ lớn hơn 5; B. `random_state` chỉ có nghĩa khi `shuffle=True` nên thư viện từ chối cặp tham số đó; C. `KFold` không có tham số `random_state`; D. bài toán phân loại bắt buộc dùng `StratifiedKFold`.

(2) Một báo cáo viết: “bơm SMOTE rồi xác thực chéo cho F1 0,8898, cao hơn hẳn 0,7089 của cách đặt SMOTE làm bước giữa của `imblearn.pipeline.Pipeline`, nên chọn cách thứ nhất”. Con số đủ bác bỏ: A. cột AUC; B. độ lệch chuẩn giữa các phần; C. F1 của chính hai mô hình ấy trên 603 dòng kiểm tra thật, bằng nhau và bằng 0,7097; D. số dòng sinh thêm, 1.545.

(3) Hai cấu hình cho F1 xác thực chéo 0,7089 với độ lệch chuẩn giữa các phần 0,0257, và 0,6837 với 0,0137. Kết luận hợp lý nhất: A. hai khoảng một độ lệch chuẩn chồng nhau 0,0142 nên chưa phân biệt được; B. cần phân tích hiệu F1 ghép cặp, độ lớn và độ bất định; hai trung bình và hai độ lệch chuẩn riêng lẻ chưa đủ để kết luận; C. cấu hình thứ nhất tốt hơn vì trung bình cao hơn 0,0252; D. cấu hình thứ hai tốt hơn vì độ lệch chuẩn nhỏ hơn.

**Bài 7.2** (★ · MTS5 · \~12 phút · Kiến thức: mục 7.1.4, 7.2.4, 7.4.3, 7.5.1 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số. (1) Bỏ bốn cột có độ quan trọng hoán vị dưới 0,01 làm rừng ngẫu nhiên tốt lên. (2) Lưới càng nhiều ô thì con số kiểm định của ô vô địch càng bị thổi lên. (3) Đổi ngưỡng quyết định không đổi AUC. (4) Mọi dòng SMOTE sinh thêm đều có tổng 22 ô một nóng đúng bằng 6,0000, nên bảng sau khi bơm là bảng hợp lệ.

**Bài 7.3** (★ · MTS5 · \~15 phút · Kiến thức: mục 7.3.2, 7.3.3 · dạng: tính toán)

Hai cách xử lý mất cân bằng đi qua **cùng một** phép `StratifiedKFold(5, shuffle=True, random_state=42)` trên 2.409 dòng huấn luyện, `scoring="f1"`, cùng một mô hình.

**Bảng cho Bài 7.3.** F1 từng phần của hai cách xử lý mất cân bằng.

  ----------------------------------------------------------------------
  Cách                      phần 1   phần 2   phần 3   phần 4   phần 5
  ------------------------- -------- -------- -------- -------- --------
  không làm gì              0,6750   0,6832   0,6923   0,7013   0,6667

  SMOTE đặt làm bước giữa   0,6798   0,7177   0,7222   0,7400   0,6849
  ----------------------------------------------------------------------

(a) Tính trung bình và độ lệch chuẩn giữa các phần của từng cách với `ddof=1`, viết hai khoảng “trung bình cộng trừ một độ lệch chuẩn”, rồi tính đoạn chồng nhau và độ rộng của nó. (b) Tính năm hiệu ghép cặp theo từng phần, đếm số phần hiệu dương, nêu kết luận đúng. (c) Một nhóm viết phép tự kiểm “hai khoảng một độ lệch chuẩn chồng nhau thì hai cấu hình không phân biệt được”. Dựng phản ví dụ từ chính bảng này, rồi viết lại phép kiểm cho đúng.

**Bài 7.4** (★ · MTS5 · \~15 phút · Kiến thức: mục 7.4.3, 7.3.2 · dạng: tính toán)

Tập huấn luyện có 2.409 dòng gồm 1.977 dòng “Đạt” và 432 dòng “Không đạt”; cả bảng có 3.012 dòng gồm 2.472 và 540.

(a) `SMOTE(random_state=42)` để `sampling_strategy` mặc định bơm lớp thiểu số lên bằng lớp đa số: tính số dòng sau khi bơm và số dòng sinh thêm. Làm lại với `sampling_strategy=0.5`, biết thư viện lấy phần nguyên của tích. (b) Một nhóm bơm cả 3.012 dòng **trước** khi chia 80 trên 20. Tính tổng số dòng sau khi bơm, số dòng sinh thêm, số dòng của tập gọi là “kiểm tra”, và số dòng sinh thêm **kỳ vọng** rơi vào tập đó nếu phép chia rải đều; con số đo được là 411, cho biết hai con số lệch theo chiều nào và vì sao.
(b) SMOTE mặc định dựng bộ láng giềng với `n_neighbors` bằng `k_neighbors` cộng 1, tức 6, nên cần ít nhất 6 dòng lớp thiểu số **trong phần dùng để khớp**. Với `StratifiedKFold(5)`, lập bảng số dòng lớp thiểu số của phần dùng để khớp khi cả bảng có 6, 7 và 8 dòng lớp thiểu số, rồi nêu số dòng tối thiểu để phép xác thực chéo đó chạy được.

**Bài 7.5** (★ · MTS5 · \~12 phút · Kiến thức: mục 7.4.3 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, kèm một phép kiểm phát hiện được nó trước khi đọc tới con số cuối.

Dấu hiệu: (1) `TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough'`; (2) `ValueError: could not convert string to float: 'Ngôn ngữ Anh'`; (3) `AttributeError: This 'Pipeline' has no attribute 'predict'` trong khi `fit` đã chạy sạch; (4) F1 xác thực chéo 0,8898, trong khi chính mô hình thu được cho đúng 0,7097 trên 603 dòng kiểm tra, y hệt bản đặt đúng chỗ; (5) tập gọi là “kiểm tra” có 989 dòng và tỉ lệ lớp dương 50%.

Nguyên nhân: A. `Pipeline` nhập từ `sklearn.pipeline` nhưng có một bước lấy mẫu lại; B. bước lấy mẫu lại đặt trước bộ tiền xử lý nên nhận cả cột chữ; C. chuỗi xử lý không có bước mô hình, bước cuối là bộ lấy mẫu lại; D. gọi `fit_resample` rồi đưa bảng đã bơm vào `cross_val_score`; E. bơm cả bảng trước khi chia tập.

**Bài 7.6** (★ · MTS5 · \~15 phút · Kiến thức: mục 7.2.2, 7.2.4 · dạng: đọc bảng)

Một nhóm dò lưới hai trục cho rừng ngẫu nhiên 200 cây theo Quy trình 7.1, tập kiểm định tách bằng `train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)` cho 1.927 và 482 dòng, mỗi ô là F1 trên 482 dòng kiểm định. Khi hai ô bằng nhau tới bốn chữ số, xếp ô có `max_depth` nhỏ hơn lên trước, rồi tới ô có `min_samples_leaf` nhỏ hơn.

**Bảng cho Bài 7.6.** F1 kiểm định, cột theo `min_samples_leaf` bằng 1, 5, 10 và 20.

  ----------------------------------------------------
  `max_depth`      1        5        10       20
  ---------------- -------- -------- -------- --------
  4                0,6202   0,6202   0,6202   0,6202

  6                0,7101   0,6809   0,6812   0,6515

  8                0,7222   0,7133   0,6667   0,6617

  10               0,7222   0,7222   0,7042   0,6716

  12               0,7067   0,7211   0,7042   0,6617

  không giới hạn   0,7152   0,7260   0,7042   0,6617
  ----------------------------------------------------

(a) Nêu ô vô địch và năm ô dẫn đầu theo đúng quy tắc phá thế hòa đã quy định. (b) Áp bước 4 Quy trình 7.1 cho ô vô địch, xét **từng trục**: trục nào có đỉnh nằm trong, trục nào có đỉnh dính biên, và với trục dính biên thì nới dãy giá trị được hay không. Giải thích thêm vì sao bốn ô của hàng `max_depth=4` bằng đúng nhau. (c) Đổi giá trị khởi tạo của phép chia kiểm định sang 0 rồi 7 thì ô vô địch là `max_depth=10, min_samples_leaf=1` với 0,6621, và `max_depth` không giới hạn với `min_samples_leaf=5` cho 0,7848. Cho biết ba con số 0,6621, 0,7260 và 0,7848 có so được với nhau hay không. (d) Khớp lại ô vô địch của giá trị khởi tạo 42 trên cả 2.409 dòng rồi đo một lần cho F1 kiểm tra 0,6629, trong khi rừng để mặc định cho 0,6919; ba con số của phép so là 9 trên 603 nhãn, 0,1382, âm 0,0290. Nêu kết luận đúng, và vì sao không được quay lại chọn một ô khác.

**Bài 7.7** (★ · MTS5 · \~12 phút · Kiến thức: mục 7.2.4 · dạng: đọc hình)

Hình dưới đây đo mức lạm phát của con số kiểm định, tức hiệu “điểm kiểm định của ô vô địch trừ điểm kiểm tra của **cùng** mô hình đó”, trên cùng tập kiểm định 482 dòng. Mỗi mức $k$ là số ứng viên đem ra so, lấy trung bình của 2.000 lần lấy mẫu.

>                      -0,0025    0         +0,005    +0,010    +0,015
>     Nhóm 1: chọn trong 42 ô của một lưới hai trục
>       k = 1 ứng viên          ##|  -0,0012
>       k = 5 ứng viên        ####|  -0,0020
>       k = 10 ứng viên      #####|  -0,0025
>       k = 20 ứng viên       ####|  -0,0022
>       k = 42 ứng viên           |####  +0,0021
>     Nhóm 2: chọn trong 100 bộ 6 đặc trưng lấy ngẫu nhiên từ 30 đặc trưng
>       k = 10 ứng viên         ##|  -0,0008
>       k = 20 ứng viên          #|  -0,0006
>       k = 50 ứng viên           |##############  +0,0069
>       k = 100 ứng viên          |############################  +0,0139

Một ký tự `#` là 0,0005 điểm, vạch `|` là mốc 0. Độ lệch chuẩn điểm kiểm định **giữa các ứng viên**: nhóm 1 là 0,0319, nhóm 2 là 0,1772. Độ lệch chuẩn của chênh lệch “kiểm định trừ kiểm tra” cho một cấu hình lấy bất kỳ: 0,0147.

(a) Nêu mức $k$ nhỏ nhất của mỗi nhóm mà tại đó lạm phát vượt mốc 0, và mức $k$ nhỏ nhất mà tại đó lạm phát vượt 0,0147. (b) Hai nhóm đều có một mức $k = 20$ nhưng cho hai giá trị khác nhau. Nêu đại lượng in kèm hình giải thích được khoảng cách đó, và nói rõ cơ chế. (c) Một nhóm kết luận “lạm phát tăng theo số lần gọi `fit`”. Bác bỏ bằng đúng hai con số của hình, rồi viết lại kết luận. Cho biết thêm một giá trị lạm phát âm có nghĩa gì.

### Vận dụng

**Bài 7.8** (★★ · MTS5 · \~60 phút · Kiến thức: mục 7.4.3, 7.3.2 · dạng: viết mã)

Đi trọn Quy trình 7.4 rồi dựng lại hai cách đặt sai để đo mức lạm phát của từng cách. Mọi phép xác thực chéo dùng `StratifiedKFold(5, shuffle=True, random_state=42)` trên 2.409 dòng huấn luyện với `scoring="f1"`; bộ lấy mẫu lại là `SMOTE(random_state=42)`, `sampling_strategy` mặc định.

(a) Dựng `imblearn.pipeline.Pipeline` ba bước theo bước 1 và 2 Quy trình 7.4; in năm con số F1 kèm trung bình và độ lệch chuẩn `ddof=1`, rồi khớp lại trên cả tập huấn luyện và đo một lần trên 603 dòng kiểm tra.

(b) Bản sai thứ nhất: gọi `fit_resample` trên tập huấn luyện đã biến đổi rồi đưa bảng đã bơm vào `cross_val_score` với `Pipeline` của `sklearn`. In năm con số F1, trung bình, mức lạm phát so với phần (a), rồi in **ba con số** của phép so hai mô hình thu được trên 603 dòng kiểm tra; cho biết ba con số ấy nói gì về loại hỏng của cách làm này. Bản sai thứ hai: bơm cả 3.012 dòng rồi mới chia 80 trên 20, in tổng số dòng, số dòng của tập gọi là “kiểm tra” và số dòng sinh thêm trong đó, F1 báo cáo trên tập ấy, F1 của chính mô hình đó trên 603 dòng kiểm tra thật, và phép tự kiểm của bước 5 Quy trình 7.4 mà nó vi phạm.

(c) Đếm ô một nóng của các dòng sinh thêm, coi một ô là **nằm giữa** khi nó lệch khỏi cả 0 và 1 quá 0,001: in số dòng có ít nhất một ô nằm giữa, tỉ lệ phần trăm, giá trị nhỏ nhất và lớn nhất của tổng 22 ô trên mỗi dòng sinh thêm. Cho biết vì sao phép kiểm theo tổng không phát hiện được lỗi này, rồi in lại số dòng khi đổi sai số sang 0,000000001.

(d) In ba con số của bản ở phần (a) so với bản không lấy mẫu lại. Xác thực chéo cho hiệu F1 dương ở cả năm phần theo Bảng 7.11, còn tập kiểm tra cho F1 thấp hơn. Viết câu kết luận được phép viết, nêu rõ con số nào dùng để chọn và con số nào chỉ để báo cáo.

Tự kiểm: (a) 0,7089 với sd 0,0257, F1 kiểm tra 0,7097; (b) 0,8898, lạm phát `+0,1809`, ba con số 0 trên 603 nhãn, 0,0000, `+0,0000`, rồi 4.944 dòng với 411 trên 989 dòng sinh ra, 0,8725 so với 0,7050; (c) 1.365 trên 1.545 dòng, tức 88,3%, tổng luôn bằng 6,0000; (d) 76 trên 603 nhãn, 0,4797, âm 0,0194.

**Bài 7.9** (★★ · MTS5 · \~50 phút · Kiến thức: mục 7.4.4, 7.4.2 · dạng: viết mã)

Chốt điểm làm việc cho hồi quy logistic theo Quy trình 7.5, bằng ba cách chốt khác nhau.

(a) Dùng `TunedThresholdClassifierCV(ong, scoring=..., cv=StratifiedKFold(5, shuffle=True, random_state=hat), refit=True, random_state=42)` với `hat` lần lượt 42, 0, 1, 2, 7; chạy hai lần, một lần `scoring="f1"`, một lần `scoring=make_scorer(fbeta_score, beta=2)`. Mỗi giá trị khởi tạo in ngưỡng chọn được, `best_score_`, rồi F1, recall, precision, TP, FP, FN trên 603 dòng kiểm tra.

(b) Áp bước 3 Quy trình 7.5: in **hai** dãy cho mỗi độ đo, dãy ngưỡng và dãy F1 kiểm tra, kèm độ rộng từng dãy. Cách chọn theo $F_{2}$ cho dãy ngưỡng hẹp hơn mà dãy F1 kiểm tra lại rộng hơn; giải thích hiện tượng đó rồi cho biết bước 3 kết luận gì cho từng độ đo.

(c) Nguồn lực hữu hạn: phòng đào tạo gọi được $k$ sinh viên mỗi đợt, $k$ lần lượt 40, 80, 120, 150. Lấy $k$ ca xác suất cao nhất bằng đúng câu lệnh `chon = np.argsort(-ps, kind="stable")[:k]`, tức sắp ổn định nên ca nào đứng trước trong tập kiểm tra thì được chọn trước. Mỗi $k$ in ngưỡng xác suất thấp nhất được chọn, số ca đúng trong $k$, recall, precision, F1, và số ca có xác suất **đúng bằng** ngưỡng ấy. Cho biết vì sao một đề bài đòi lấy “$k$ ca cao nhất” buộc phải quy định câu lệnh sắp xếp.

(d) In ba con số của phép so bản $k = 120$ với bản để ngưỡng 0,2727, rồi cho biết mức lệch xác suất của phép so này phải bằng mấy và vì sao, theo phép tự kiểm của bước 4 Quy trình 7.5.

Tự kiểm: `scoring="f1"` cho năm ngưỡng 0,2727 · 0,2525 · 0,2727 · 0,2929 · 0,3131; $F_{2}$ cho 0,1616 · 0,1717 · 0,1616 · 0,1818 · 0,1717; $k = 120$ đúng 79 ca với ngưỡng 0,3605; (d) cho 28 trên 603 nhãn, 0,0000, âm 0,0336.

**Bài 7.10** (★★ · MTS5 · \~60 phút · Kiến thức: mục 7.5.1, 7.5.2, 7.4.4 · dạng: viết mã) **\[HTTT\]**

Chốt cấu hình cuối cùng cho hệ thống cảnh báo sớm của phòng đào tạo theo Quy trình 7.6, rồi bảo vệ con số thu được.

(a) Khớp cấu hình B, tức hồi quy logistic trong chuỗi xử lý của Chương 4 với ngưỡng chọn bằng `TunedThresholdClassifierCV(..., scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True, random_state=42)`, rồi đo đúng một lần trên 603 dòng kiểm tra: ngưỡng, năm độ đo, ma trận nhầm lẫn. In ba con số của phép so với mô hình Chương 6, tức cùng chuỗi xử lý ấy để ngưỡng 0,5.

(b) Lặp lại phép so trên mười lần chia phân tầng với `random_state` 0 tới 9, mỗi lần khớp lại cả bộ tiền xử lý. In hiệu F1, hiệu recall và hiệu precision của từng lần, số lần hiệu dương của từng độ đo, và dải mười ngưỡng chọn được.

(c) Dựng đường cong học của cấu hình B và của cấu hình A, tức cùng mô hình nhưng để ngưỡng 0,5, bằng `learning_curve(..., cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1", train_sizes=np.linspace(0.1, 1.0, 6))`. In hai đường và khoảng cách giữa chúng, rồi trả lời: có nên đi thu thêm dòng dữ liệu hay không, và vì sao khoảng cách của cấu hình B rộng hơn của cấu hình A ở cả sáu mốc.

(d) Tính tổng tổn thất hai bản theo giá 0,25 giờ cho một ca báo động sai và 45 giờ cho một ca bỏ sót, rồi viết đoạn kết luận của bước 5 Quy trình 7.6. Đoạn kết luận phải nêu **cả cái mất**, quy ra số cuộc gọi thêm mỗi đợt và mức precision đã đánh đổi.

(e) Bước 1 Quy trình 7.6 cho nhóm ứng viên gồm B, C, E, F. Cấu hình F có F1 xác thực chéo cao nhất bảng, 0,7220, còn hiệu F1 từng phần của B trừ F dương 2 trên 5 phần với trung bình âm 0,0105. Cho biết vì sao không chọn F, viết ba lý do chọn B theo bước 2 mà không lý do nào nhắc tới một con số độ đo, rồi bác bỏ đề nghị “F1 kiểm tra thấp hơn Chương 6 đúng 0,0025 nên quay lại ngưỡng 0,5” bằng bước 4 Quy trình 7.6 và ba đại lượng đo được ở phần (b) và (d).

Tự kiểm: (a) ngưỡng 0,2727, F1 0,7266, recall 0,8611, ma trận `[[440, 55], [15, 93]]`, ba con số 53 trên 603 nhãn, 0,0000, âm 0,0025; (b) hiệu F1 dương 10 trên 10 với trung bình `+0,0364`, hiệu precision dương 0 trên 10; (c) 688,75 giờ thay vì 1.535,25 giờ.

**Bài 7.11** (★★ · MTS5 · \~45 phút · Kiến thức: mục 7.3.4, 7.4.2, 7.2.1 · dạng: viết mã)

Tinh chỉnh hồi quy logistic theo Quy trình 7.3 trên lưới hai trục `C` thuộc {0,01; 0,1; 1; 10; 100} và `class_weight` thuộc {`None`, `"balanced"`}, với `cv=StratifiedKFold(5, shuffle=True, random_state=42)`, `scoring="f1"`, `refit=True`.

(a) In `best_params_`, `best_score_`, số lần khớp tính theo bước 2, và bảng `cv_results_` xếp theo `rank_test_score` gồm `mean_test_score`, `std_test_score` và năm cột `split0_test_score` tới `split4_test_score`. Cột `param_mo_hinh__class_weight` in ra giá trị thiếu ở năm ô có `class_weight` bằng `None`: nêu nguyên nhân và viết câu lệnh lọc đúng năm ô ấy.

(b) Áp bước 4: so hai ô dẫn đầu bằng **hiệu theo từng phần** lấy từ các cột `split*`; in năm hiệu, số phần hiệu dương, trung bình hiệu. So thêm trung bình chênh lệch hai ô với `std_test_score` của chúng rồi cho biết vì sao phép so thứ hai không được dùng để kết luận. Kết luận cuối phải nói rõ chọn ô nào và theo tiêu chí gì.

(c) Áp bước 5: đo đúng một lần trên 603 dòng kiểm tra, in năm độ đo và ma trận nhầm lẫn, rồi so con số F1 ấy với `best_score_` và giải thích chiều lệch.

(d) Đổi `random_state` của `cv` sang 0, 1, 7 và 2026, chạy lại trọn lưới: in cấu hình chọn được và F1 kiểm tra của từng giá trị khởi tạo, rồi kết luận về độ bền. Một giá trị khởi tạo cho cấu hình khác hẳn bốn giá trị còn lại; nêu giá trị đó và cho biết nó đổi kết luận của phần (b) hay không.

Tự kiểm: `best_params_` là `C` bằng 0,1 với `class_weight="balanced"`, `best_score_` 0,6998, 51 lần khớp; hiệu từng phần hai ô dẫn đầu dương 4 trên 5; F1 kiểm tra 0,7092 với ma trận `[[421, 74], [8, 100]]`; giá trị khởi tạo 0 cho `C` bằng 1 với `class_weight=None`.

### Thực tế và liên ngành

**Bài 7.12** (★★★ · MTS5 · \~45 phút · Kiến thức: mục 7.4.1, 7.4.3, 7.3.2 · dạng: bảo vệ phương án, Thiết kế công nghiệp)

Công ty Thiết kế và Chế tạo Bảo Lâm kiểm tra khuyết tật bề mặt trên chi tiết nhựa đúc. Tỉ lệ chi tiết lỗi là 1,2%, và một chi tiết lỗi lọt ra khỏi xưởng đắt hơn hẳn một chi tiết tốt bị giữ lại để soi.

(a) Dựng một bảng huấn luyện có tỉ lệ lớp dương xấp xỉ 1,2%: giữ nguyên cả 1.977 dòng lớp âm của tập huấn luyện, lấy 24 dòng lớp dương bằng `np.random.default_rng(42)` với `rng.choice(..., replace=False)`, giữ nguyên 603 dòng kiểm tra. Khớp hồi quy logistic rồi in năm độ đo và ma trận nhầm lẫn ở ngưỡng 0,5, kèm ngưỡng lớn nhất trong các ngưỡng thử vẫn đạt recall ít nhất 0,60 và precision tại ngưỡng đó.

(b) Bơm `SMOTE(random_state=42)` đúng chỗ theo Quy trình 7.4 trên bảng thưa ấy, một lần với `StratifiedKFold(5, shuffle=True, random_state=42)` và một lần với `StratifiedKFold(10, shuffle=True, random_state=42)`, `scoring="f1"`. In các phần của cả bản có SMOTE và bản không làm gì, hiệu theo từng phần và số phần hiệu dương. Nêu kết luận, rồi nêu con số của bản 10 phần cho thấy vì sao không nên chia 10 phần với bảng này.

(c) Xưởng kiểm 50.000 chi tiết mỗi năm. Tính số chi tiết lỗi một năm và accuracy của mô hình luôn trả lời “không lỗi”. Một báo cáo kết luận “mô hình đạt độ chính xác 98,8% nên thay được công đoạn soi bằng mắt”. Viết **ba** câu hỏi, mỗi câu đủ bác bỏ kết luận ấy và kèm một con số của phần (a) hoặc (b).

(d) Phát biểu “dữ liệu mất cân bằng thì phải lấy mẫu lại” cần điều kiện áp dụng. Dựng **hai ca** từ số liệu đã đo: một ca mất cân bằng nặng mà lấy mẫu lại **không** giúp, một ca mất cân bằng nhẹ hơn mà lấy mẫu lại **có** giúp. Mỗi ca nêu tỉ lệ lớp dương, số dòng lớp dương tuyệt đối, hiệu F1 theo từng phần và số phần hiệu dương. Viết lại phát biểu cho đúng phạm vi, rồi bảo vệ phương án trong 5 phút trước hội đồng kỹ thuật của xưởng, kèm một câu trả lời cho ý kiến “tăng số chi tiết lỗi bằng SMOTE là cách rẻ nhất, vì không phải đi thu thêm mẫu lỗi”.

Tự kiểm: bảng thưa có 2.001 dòng với 24 dòng lớp dương; ngưỡng 0,5 cho recall 0,1389, precision 1,0000, AUC 0,9375; ngưỡng lớn nhất vẫn đạt recall ít nhất 0,60 là khoảng 0,0609; bản 5 phần cho SMOTE 0,1326 và bản không làm gì 0,1833 với hiệu dương 2 trên 5.

**Bài 7.13** (★★★ · MTS5 · \~50 phút · Kiến thức: mục 7.2.4, 7.3.2, 7.3.3 · dạng: tình huống nghiệp vụ và vấn đáp, Ngôn ngữ Anh)

Trung tâm Ngoại ngữ Sao Mai chỉ có hồ sơ của 180 học viên và muốn dự báo học viên nào cần kèm thêm trước kỳ thi nói. Cả bài làm trên bảng nhỏ `nho = bang.sample(n=180, random_state=2026)`, chia `test_size=0.2, random_state=42, stratify`, mô hình là hồi quy logistic trong chuỗi xử lý của Chương 4.

(a) In số dòng và số ca lớp dương của bảng nhỏ, của tập huấn luyện và của tập kiểm tra. Chọn `C` trong {0,01; 0,1; 1; 10; 100} trên **một** tập kiểm định tách bằng `train_test_split(..., test_size=0.25, random_state=hat, stratify=...)` với `hat` lần lượt 42, 0, 1, 2, 7: mỗi giá trị khởi tạo in năm giá trị F1 kiểm định, giá trị `C` chọn được, và F1 trên tập kiểm tra sau khi khớp lại trên cả tập huấn luyện.

(b) Làm lại phép chọn `C` bằng `StratifiedKFold(5, shuffle=True, random_state=42)` trên tập huấn luyện, in năm phần của từng giá trị `C` kèm trung bình và độ lệch chuẩn. Cho biết `C` chọn được có đổi so với phần (a) hay không, và cách nào đáng tin hơn, kèm con số. Với `C` bằng 0,01, F1 bằng đúng 0,0000 ở bốn trong năm phần: nêu nguyên nhân và một phép kiểm phát hiện được nó **trước** khi đọc tới cột F1.

(c) In số ca lớp dương của từng phần dùng để đo khi chia 5 phần và khi chia 10 phần. Dùng ngưỡng của mục 7.3.2 về cỡ lớp thiểu số để kết luận nên chia mấy phần, kèm con số của chính bảng nhỏ này.

(d) Chuẩn bị trả lời vấn đáp 3 phút. Trung tâm muốn in lên tài liệu tuyển sinh câu “mô hình đạt F1 0,7778”. Cho biết con số ấy được phép công bố hay không, nó là con số của phép đo nào, và con số nào được phép công bố thay nó.

Tự kiểm: bảng nhỏ có 39 ca lớp dương; tập huấn luyện 144 dòng với 31 ca dương, tập kiểm tra 36 dòng với 8 ca dương; năm giá trị khởi tạo của phần (a) cho `C` bằng 100 · 0,1 · 100 · 0,1 · 10; phần (b) chọn `C` bằng 0,1 với F1 trung bình 0,6983.

**Bài 7.14** (★★★ · MTS5 · \~180 phút cho cả nhóm · Kiến thức: mục 7.4.4, 7.5.1, 7.5.2 · dạng: làm nhóm ba người, có trình bày, Quản trị kinh doanh)

Chuỗi cửa hàng tiện lợi Hương Việt dùng một mô hình để dự báo khách hàng thành viên sắp rời đi, rồi gửi voucher giữ khách. Một voucher gửi cho người vốn không rời đi mất 200.000 đồng; một khách rời đi mà không giữ được mất 900.000 đồng.

(a) Lấy vectơ xác suất của mô hình chốt ở Mã 7.16 trên 603 dòng kiểm tra rồi lập bảng bảy ngưỡng 0,10 · 0,15 · 0,20 · 0,2727 · 0,35 · 0,45 · 0,60, mỗi dòng in TP, FP, FN, recall, precision, F1 và số ca được gọi tên.

(b) Thêm cột tổng chi phí theo cặp giá trên, đơn vị triệu đồng. Nêu ngưỡng rẻ nhất, rồi tính **tỉ lệ giá hòa vốn** giữa ngưỡng rẻ nhất và ngưỡng rẻ thứ hai, tức tỉ số giữa giá một ca bỏ sót và giá một ca báo động sai mà tại đó hai ngưỡng tốn như nhau; so tỉ lệ ấy với tỉ lệ giá thật. Quét thêm lưới ngưỡng bước 0,01 để biết điểm thấp nhất có nằm trong bảy dòng hay không, rồi so nó với ngưỡng lý thuyết của khung tổn thất tuyến tính.

(c) Lặp lại phép so hai ngưỡng rẻ nhất trên mười lần chia phân tầng với `random_state` 0 tới 9, mỗi lần khớp lại cả chuỗi xử lý: in chi phí từng ngưỡng ở từng lần, hiệu chi phí, số lần hiệu âm, dải hiệu. Vẽ biểu đồ cột so chi phí theo bảy ngưỡng theo ba quy ước của mục 6.5.2, thanh khoảng là khoảng của **hiệu** chi phí giữa hai ngưỡng rẻ nhất. Nêu kết luận đọc được từ hiệu mà bảng của một lần chia che mất, và một kết luận sai nếu thanh khoảng là dải chi phí của riêng một ngưỡng.

(d) Chia ba phần trên cho ba người. Nộp mã nguồn chạy lại được, một hình, bảng kết quả và bài trình bày 7 phút trả lời một câu hỏi: nếu mỗi đợt chuỗi chỉ in được **120** voucher thì chọn một ngưỡng cố định hay chọn “120 ca xác suất cao nhất”, lấy bằng đúng `np.argsort(-ps, kind="stable")[:120]`? Nêu lý do phải quy định câu lệnh ấy trong đề bài. Căn cứ phải là tổng chi phí và số ca đúng trong 120, không phải cột F1.

Tự kiểm: ngưỡng 0,20 cho TP 98, FP 67, FN 10; chi phí bảy dòng chạy từ 22,40 tới 42,60 triệu đồng; phần (c) cho hiệu âm 8 trên 10 lần; phép lấy 120 ca cao nhất cho 79 ca đúng với ngưỡng thấp nhất được chọn 0,3605.

### Thử thách có AI

**Bài 7.15** (★★★★ · MTS5 · \~100 phút · Kiến thức: toàn chương · dạng: niêm phong dự đoán, hỏi AI, chạy thật, rồi phát hiện lỗi)

**Bài toán.** Một nhóm nhờ trợ lý AI tạo sinh viết chương trình cải thiện mô hình cảnh báo sớm trên dữ liệu mất cân bằng, và nhận về đoạn mã dưới đây cùng báo cáo bốn câu: “SMOTE đẩy F1 xác thực chéo lên 0,8898, cao hơn hẳn mức 0,68 của bản không can thiệp”; “nếu `Pipeline` báo lỗi về các bước trung gian thì chỉ cần gọi `fit_resample` trước rồi đưa bảng đã bơm vào `Pipeline` thường”; “quét `random_state` trong lưới giúp tìm được cấu hình rừng tốt nhất, `best_score_` 0,6832”; “ngưỡng tốt nhất là 0,49 với F1 0,7379, đây là hiệu năng của mô hình khi triển khai”. Phần nhập thư viện và các dòng nạp `bang`, `tien` giữ nguyên như các chương trình của chương, không chứa lỗi nào.

>     X = bang.drop(columns=["ma_sv", "ngay_dang_ky", "tinh_thanh",
>                            "diem_cuoi_ky", "ket_qua"])
>     y = (bang["ket_qua"] == "Không đạt").astype(int)
>     X_tr, X_te, y_tr, y_te = train_test_split(
>         X, y, test_size=0.2, random_state=42, stratify=y)
>     cv = KFold(n_splits=5, shuffle=False, random_state=42)
>     lg = Pipeline([("tien", clone(tien)),
>                    ("mo_hinh", LogisticRegression(max_iter=1000))])
>     ong = Pipeline([("tien", clone(tien)),
>                     ("lay_mau", SMOTE(ratio=1.0, random_state=42)),
>                     ("mo_hinh", LogisticRegression(max_iter=1000))])
>     print("F1 có SMOTE %.4f"
>           % cross_val_score(ong, X_tr, y_tr, cv=cv,
>                             scoring="f1").mean())
>     bo = clone(tien).fit(X_tr)
>     A2, a2 = SMOTE(random_state=42).fit_resample(bo.transform(X_tr),
>                                                  y_tr)
>     d = cross_val_score(
>         Pipeline([("mo_hinh", LogisticRegression(max_iter=1000))]),
>         A2, a2, cv=cv, scoring="f1")
>     print("F1 sau khi bơm %.4f" % d.mean())
>     luoi = {"mo_hinh__max_depth": [8, None],
>             "mo_hinh__random_state": [0, 1, 7, 42, 2026]}
>     do = GridSearchCV(
>         Pipeline([
>             ("tien", clone(tien)),
>             ("mo_hinh", RandomForestClassifier(n_estimators=200))]),
>         luoi, cv=cv, scoring="f1").fit(X_tr, y_tr)
>     print("best_score_ %.4f %s" % (do.best_score_, do.best_params_))
>     ps = clone(lg).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
>     tot = max(np.arange(0.05, 0.95, 0.01),
>               key=lambda t: f1_score(y_te, (ps >= t).astype(int)))
>     print("ngưỡng tốt nhất %.2f | F1 %.4f"
>           % (tot, f1_score(y_te, (ps >= tot).astype(int))))

**Nhiệm vụ với AI.** Ba bước theo đúng thứ tự. Bước một: **chưa** dùng công cụ AI, tự đọc đoạn mã và bốn câu báo cáo, rồi viết và niêm phong danh sách lỗi dự đoán kèm con số dự đoán cho từng độ đo. Bước hai: đưa đoạn mã và bốn câu báo cáo cho một công cụ AI tạo sinh, yêu cầu nó liệt kê mọi lỗi phương pháp và mọi lỗi phiên bản thư viện. Bước ba: chạy thật, sửa lần lượt từng lỗi theo đúng thứ tự chương trình gặp, rồi so ba kết quả.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi ứng với từng lỗi, cùng bản niêm phong của bước một. Chỉ dán tên cột, kiểu cột và số mức.

**Kiểm chứng.** Đoạn mã và báo cáo chứa **sáu** lỗi. Ba lỗi làm chương trình dừng: nộp thông báo nguyên văn, tên hàm hoặc tham số thay thế trong phiên bản thư viện sách khai, và dòng đã sửa; với mỗi lỗi dừng phải chạy thử **cả dòng đã sửa**, vì hai trong ba lỗi có cách sửa hiển nhiên mà vẫn dừng, một lần với đúng thông báo cũ và một lần với một thông báo mới. Ba lỗi còn lại không làm chương trình dừng mà làm kết luận sai, và chúng **chồng lên nhau**: phải sửa ba lỗi dừng mới đo được hậu quả của chúng; mỗi lỗi im lặng nộp bộ số trước và sau khi sửa kèm mức lạm phát.

Ba con số sau được chấm theo giá trị, vì giao thức sinh ra chúng đã quy định: F1 xác thực chéo của bản SMOTE đặt đúng chỗ, F1 của bản bơm trước rồi đưa vào `Pipeline` thường, và F1 trên 603 dòng kiểm tra của bản chọn ngưỡng bằng xác thực chéo. Giao thức: mọi phép xác thực chéo của bản sửa dùng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trên 2.409 dòng huấn luyện với `scoring="f1"`; bộ lấy mẫu lại là `SMOTE(random_state=42)` đặt làm bước giữa của `imblearn.pipeline.Pipeline`; ngưỡng chọn bằng `TunedThresholdClassifierCV(ong, scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True, random_state=42)`.

Với lỗi quét `random_state` trong lưới, phần sửa phải in năm giá trị `mean_test_score` của năm giá trị khởi tạo ứng với **cùng** một `max_depth`, rồi tính mức “lợi giả”, tức hiệu giữa giá trị cao nhất và trung bình của năm giá trị ấy. Cuối cùng so ba danh sách: lỗi nào tự tìm được mà công cụ bỏ sót, lỗi nào ngược lại, lỗi nào công cụ báo oan.

**Phản tư** (100 tới 200 từ). Công cụ AI giúp được gì và không giúp được gì khi soát một chương trình có bước lấy mẫu lại; vì sao một con số cao hơn lại là dấu hiệu đáng ngờ; rủi ro khi một hệ thống gọi tên sinh viên được kết luận đạt yêu cầu bằng con số 0,8898, kèm trách nhiệm với 1.545 dòng dữ liệu sinh ra không ứng với một con người nào.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất lượng kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi tiết ở tài liệu hướng dẫn chấm đi kèm sách.
