# Bài tập – Chương 8

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Mọi bài làm trên bảng 3.012 dòng của Chương 4, chia `test_size=0.2, random_state=42, stratify=y`, lớp dương là “Không đạt”. Mô hình xuất phát là `models/ch08-mo-hinh-trien-khai.joblib` với ngưỡng 0,2727104755205902 đọc từ `models/ch08-ho-so-trien-khai.json`; dữ liệu mới luôn đưa vào dưới dạng `DataFrame`. Từ tầng hai trở lên nộp mã nguồn và kết quả in ra.

Câu hỏi “thay đổi này có ảnh hưởng không” phải in đủ **ba con số**: số ca đổi nhãn trên tổng số ca, mức lệch xác suất lớn nhất, lệch của độ đo tổng hợp. Chương này có nhiều ca mà cả ba con số đều bằng 0 trong khi dữ liệu vào đã sai, nên bài nào kiểm hợp đồng dữ liệu vào còn phải in **ba phép đếm** của Quy trình 8.2: phần lệch giữa hai tập tên cột, số ô ngoài miền theo từng cột số, số dòng mang mức phân loại chưa có trong `categories_` theo từng cột phân loại.

### Củng cố

**Bài 8.1** (★ · MTS6 · \~12 phút · Kiến thức: mục 8.1.1, 8.1.2, 8.1.3 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một nhóm muốn xác nhận số dòng đã dùng để huấn luyện mô hình. Cách phù hợp nhất: A. nhân n_features_in\_ với số nhánh; B. lấy len(statistics\_) của bộ điền; C. đọc hồ sơ huấn luyện và đối chiếu trạng thái như n_samples_seen\_ nếu có, sau khi xét cách dùng trọng số và giá trị thiếu; D. lấy n_iter\_ của LogisticRegression.

(2) Lưu `Pipeline` bên trong thay vì lưu đối tượng có ngưỡng làm đổi 53 trên 603 nhãn, lệch xác suất 0,0000 và lệch F1 **dương** 0,0025. Con số nên dùng làm phép kiểm cho lỗi này: A. lệch F1, vì nó là độ đo tổng hợp của bài toán; B. lệch xác suất, vì nó bằng 0 nên chứng tỏ mô hình không đổi; C. số sinh viên bị bỏ sót, vì nó đi từ 15 lên 34; D. dung lượng file, vì bản thiếu ngưỡng nhỏ hơn 2,1 KB.

(3) Chương trình nạp một file mô hình nhận từ đối tác và in ra `InconsistentVersionWarning`. Hậu quả nếu không xử lý gì: A. `joblib.load` trả về `None`; B. không có bảo đảm tương thích; tệp có thể nạp kèm cảnh báo nhưng vẫn cần dừng để xác minh đúng môi trường trước khi sử dụng; C. mọi lời gọi `predict` sau đó dừng với `ValueError`; D. thư viện tự chuyển đối tượng sang phiên bản đang cài.

**Bài 8.2** (★ · MTS6 · \~15 phút · Kiến thức: mục 8.1.3, 8.2.2, 8.3.2, 8.3.3 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số; phát biểu chỉ đúng trong một phạm vi thì nêu rõ điều kiện áp dụng.

(1) Đảo thứ tự các cột của bảng dữ liệu mới không đổi kết quả dự đoán.

(2) Khai `handle_unknown="ignore"` làm mức phân loại lạ được xử lý an toàn, vì lệnh gọi không báo lỗi và độ đo không đổi.

(3) Mã băm SHA-256 của file mô hình trùng với mã băm do bên gửi công bố thì nạp file ấy là an toàn.

(4) Một bản ghi khuyết ô `diem_giua_ky` và ô `thiet_bi_chinh` làm chương trình dự đoán dừng với `ValueError: Input X contains NaN.`

**Bài 8.3** (★ · MTS6 · \~15 phút · Kiến thức: mục 8.3.1, 8.3.2 · dạng: tính toán)

Mã 8.13 đếm được 92 giá trị phải khớp giữa chương trình huấn luyện và chương trình dự đoán khi hai chương trình gọi rời từng bước, thuộc 6 hạng mục. Tính lại con số ấy cho hai cấu hình dưới đây, lập bảng 6 hạng mục cho từng cấu hình, và nêu số đặc trưng ra.

(a) Giữ 8 cột số, thêm `tinh_thanh` vào nhánh phân loại thành 7 cột phân loại; sau bước làm sạch của Chương 3, `tinh_thanh` có 16 mức.

(b) Giữ nguyên 14 cột vào nhưng khai `OneHotEncoder(handle_unknown="ignore", drop="first")`.

(c) Bước 4 Quy trình 8.3 có phép tự kiểm “số đặc trưng ra bằng số cột số cộng tổng số mức của các cột phân loại”. Cho biết phép tự kiểm ấy đạt hay trượt ở từng cấu hình, rồi viết lại nó thành một công thức đúng cho cả hai.

**Bài 8.4** (★ · MTS6 · \~15 phút · Kiến thức: mục 8.2.1, 8.2.2 · dạng: đọc bảng)

Năm lô dưới đây dựng từ chính 603 dòng kiểm tra, mỗi lô đổi đúng một thứ. Lô mốc so sánh cho F1 0,7266 với 148 dòng vượt ngưỡng; ba cột cuối đếm so với lô mốc.

**Bảng cho Bài 8.4.** Năm lô dữ liệu mới, ba con số và ba phép đếm.

  ------------------------------------------------------------------------------------------------------------------------------------
  Lô                                                 Đổi nhãn   Lệch xác suất   Lệch F1       Cột thiếu   Ô ngoài miền   Dòng mức lạ
  -------------------------------------------------- ---------- --------------- ------------- ----------- -------------- -------------
  1\. `nop_tre_tb_gio` bằng 99999 ở 20 dòng đầu      16/603     0,9999          −0,0427       0           20             0

  2\. `diem_tb_bai_tap` nhân 10 cho cả cột           144/603    0,9943          −0,6551       0           599            0

  3\. `nhom_tuoi` viết `23–30` bằng gạch ngang dài   10/603     0,0453          −0,0030       0           0              577

  4\. `vua_lam_vua_hoc` đổi sang `Yes` và `No`       40/603     0,1515          −0,0336       0           0              603

  5\. `hinh_thuc_hoc` bằng `Từ xa` ở 15 dòng cuối    2/603      0,1605          **+0,0057**   0           0              15
  ------------------------------------------------------------------------------------------------------------------------------------

(a) Xếp năm lô thành ba nhóm theo **dấu hiệu để lại**, theo đúng cách chia của Bảng 8.3, và cho biết những nhóm nào không có lô nào.

(b) Lô 5 cho lệch F1 dương. Nêu lý do con số dương ấy không phải bằng chứng rằng lô 5 vô hại, và chỉ ra con số nào trong bảng mới là bằng chứng. Lô 3 và lô 4 đổi mức của cả cột mà chỉ đổi 10 và 40 nhãn: giải thích chênh lệch giữa số dòng mang mức lạ và số nhãn đổi.

(c) Hai cột số của chính lô mốc có một ô vượt miền quan sát trên tập huấn luyện: `tong_thoi_luong_phut` đạt 2198 trong khi miền là từ 20 tới 2151, `nop_tre_tb_gio` đạt 644 trong khi miền là từ 0 tới 420. Nêu bước 3 Quy trình 8.2 đòi gì ở đây, rồi dùng ba dấu hiệu của mục 3.2.1 để phân biệt hai ô ấy với 20 ô của lô 1.

**Bài 8.5** (★ · MTS6 · \~12 phút · Kiến thức: mục 8.2.2, 8.2.3, 8.3.3 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, rồi nêu bước nào của Quy trình 8.2 phát hiện được nó trước khi chương trình chạy tới lệnh `predict`. Nếu không bước nào của quy trình phát hiện được một dấu hiệu thì nói rõ như vậy và nêu phép kiểm phải thêm, kèm chỗ đặt nó trong quy trình.

Dấu hiệu: (1) `ValueError: columns are missing: {'diem_giua_ky'}`; (2) `ValueError: Specifying the columns using strings is only supported for dataframes.`; (3) `ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: '6,7'`; (4) `ValueError: Found array with 0 sample(s) (shape=(0, 8)) while a minimum of 1 is required by SimpleImputer.`; (5) `AttributeError: 'Pipeline' object has no attribute 'best_threshold_'`; (6) không có thông báo nào, 0 trên 150 nhãn đổi, lệch F1 đúng 0,0000, lệch xác suất lớn nhất 0,0518.

Nguyên nhân: A. lô mới đọc xong bị chuyển sang mảng bằng `.to_numpy()`; B. cột `diem_giua_ky` bị đổi tên thành `diemgiuaky` khi xuất file; C. bộ lọc điều kiện ở đầu chương trình không giữ lại dòng nào; D. 45 dòng mang một mức phân loại chưa có trong `categories_`; E. chương trình nạp `ch08-pipeline-khong-nguong.joblib` thay cho đối tượng có ngưỡng; F. cột `diem_tb_bai_tap` dùng dấu phẩy thập phân nên cả cột bị đọc thành chuỗi.

**Bài 8.6** (★ · MTS6 · \~20 phút · Kiến thức: mục 8.1.2, 8.2.3, 8.3.4 · dạng: đọc hồ sơ và đọc mã)

Một nhóm nộp hai sản phẩm kèm `ch08-mo-hinh-trien-khai.joblib`: một hồ sơ mô hình và một chương trình dự đoán. Hồ sơ dưới đây đã lược các khóa không có lỗi; khóa `cot_vao_theo_thu_tu` của nó là 14 tên cột xếp theo đúng thứ tự của `bo-tien-xu-ly.joblib`, tức tám cột số trước, sáu cột phân loại sau.

>     {
>       "so_dong_huan_luyen": 3012,
>       "cach_chia":
>         "train_test_split(test_size=0.2, random_state=0, stratify=y)",
>       "nguong_quyet_dinh": 0.2727,
>       "do_do_tren_tap_kiem_tra": {"f1": 0.7266, "recall": 0.6852}
>     }

Chương trình dự đoán, phần nhập thư viện và phần dựng đường dẫn giữ nguyên như các chương trình của chương, không chứa lỗi nào:

>     mo_hinh = joblib.load(THU_MUC_MO_HINH
>                           / "ch08-mo-hinh-trien-khai.joblib")
>     ong = mo_hinh.estimator.estimator
>     lo = pd.read_csv(VAO / "lo-moi.csv")
>     tien = clone(ong.named_steps["tien"]).fit(lo)
>     A = tien.transform(lo)
>     ps = ong.named_steps["mo_hinh"].predict_proba(A.to_numpy())[:, 1]
>     nhan = (ps > 0.2727).astype(int)
>     pd.DataFrame({"nhan": nhan}).to_csv(RA / "ket-qua.csv", index=False)

(a) Hồ sơ có **năm** chỗ sai. Nêu từng chỗ, xếp nó vào một trong bốn loại khóa của bước 3 Quy trình 8.1 và cho biết phép kiểm tương ứng. Chỉ ra chỗ sai **không** phát hiện được bằng bất kỳ phép so nào với đối tượng mô hình, và nói rõ phải đối chiếu với cái gì.

(b) Chương trình có **bốn** lỗi. Nêu từng lỗi, bước nào của Quy trình 8.2 nó vi phạm, và dấu hiệu nó để lại: dừng chương trình, chỉ sinh cảnh báo, hay không để lại gì.

(c) Một trong bốn lỗi không đổi một nhãn nào trên 603 dòng kiểm tra. Nêu lỗi đó và nêu phép đếm phát hiện được nó, nói rõ phép đếm ấy đếm cái gì.

### Vận dụng

**Bài 8.7** (★★ · MTS1, MTS6 · \~55 phút · Kiến thức: mục 8.1.1, 8.1.2, 8.1.3 · dạng: viết mã)

Đi trọn Quy trình 8.1 cho mô hình cuối Chương 7, rồi đo hậu quả của việc lưu sai tầng đối tượng trên nhiều lần chia.

(a) Nạp `ch07-mo-hinh-chot.joblib`, xếp các đại lượng mà `fit` đã ghi vào đối tượng theo bốn nhóm của Bảng 8.1. Đi bước 1 và bước 2, in các con số của phép tự kiểm từng bước, trong đó có mức dung lượng suy từ số tham số đã học.

(b) Đi bước 3: ghi hồ sơ mô hình thành JSON, in kèm **nguồn của từng khóa** theo bốn loại của bước 3: đọc từ đối tượng, đọc từ dữ liệu trong cùng lần chạy, gọi hàm đo trong cùng lần chạy, hay mô tả bằng lời. Không gõ tay giá trị nào thuộc ba loại đầu.

(c) Đi bước 4 và bước 5: ghi mã băm SHA-256 ra file cùng tên rồi tính lại để so; sao file sang một **bản sao**, đảo một bit ở giữa bản sao rồi in dung lượng, mã băm và kết quả lệnh nạp của cả hai file. Nạp lại trong một tiến trình mới, so tên cột **từng phần tử theo thứ tự**, so ngưỡng, so nhãn từng dòng. Thêm một phép so cố tình làm sai: so hai danh sách tên cột bằng `len`, một bên lấy từ `bo-tien-xu-ly.joblib`.

(d) Lưu sai tầng, tức lưu `mo_hinh.estimator_` thay cho `mo_hinh`. Lặp phép so trên **sáu** lần chia phân tầng với `random_state` lần lượt 42, 0, 1, 2, 3, 4; mỗi lần khớp lại cả chuỗi xử lý và chọn lại ngưỡng bằng `TunedThresholdClassifierCV(ong, scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True, random_state=42)`. In bảng sáu dòng gồm ngưỡng, ba con số, số sinh viên bị bỏ sót trước và sau; đếm số lần hiệu F1 dương, số lần hiệu F1 âm, số lần lệch xác suất khác 0, số lần số bị bỏ sót tăng. Kết luận: trong ba con số, con số nào dùng được làm phép kiểm cho lỗi này và con số nào không, lý do đọc từ chính bảng.

Tự kiểm: (a) ma trận `[[440, 55], [15, 93]]`, và file ghi lại ở bước 2 nặng 9,7 KB; (c) mã băm tham chiếu của lần chạy tác giả bắt đầu bằng 962f51ff92158a65; không yêu cầu khớp chuỗi này khi khác môi trường hoặc cách tuần tự hóa; bản sao đảo một bit cho một mã băm khác hẳn mà **dung lượng không đổi**, nên phép so dung lượng không phân biệt được hai file. Giá trị cụ thể của mã băm thứ hai phụ thuộc vào bit nào bị đảo, nên chỉ cần nêu hai điều: hai mã băm khác nhau và hai dung lượng bằng nhau. (d) lần chia 42 cho lệch F1 `+0,0025`, năm lần còn lại cho hiệu âm, lệch xác suất bằng 0,0000 ở cả sáu lần.

**Bài 8.8** (★★ · MTS6 · \~55 phút · Kiến thức: mục 8.2.1, 8.2.2, 8.2.3 · dạng: viết mã)

Phòng đào tạo gửi một lô 40 sinh viên mới. Dựng lại lô ấy rồi đi trọn Quy trình 8.2. Cách dựng, theo đúng thứ tự: lấy `lo = X_te.iloc[:40].copy()` làm lô mốc so sánh; đặt `thiet_bi_chinh` bằng `"May tinh"` ở 5 dòng đầu; đặt `tong_thoi_luong_phut` bằng 99999 ở dòng chỉ số 5; đặt `diem_giua_ky` bằng `np.nan` ở ba dòng chỉ số 10, 11, 12; thêm cột `tinh_thanh` bằng `"Hà Nội"` cho cả lô.

(a) Đi bước 1 tới bước 4, in đủ các con số của phép tự kiểm từng bước, kèm giá trị vi phạm của mỗi ô ngoài miền và giá trị lạ của mỗi mức lạ. Mỗi con số khác 0 kèm một dòng quyết định xử lý.

(b) Đi bước 5 tới bước 7: gọi `predict_proba` một lần cho cả lô, áp ngưỡng, ghi file có cột xác suất và cột nhãn xếp giảm dần theo xác suất. In **ba con số** của lô có lỗi so với lô mốc và tỉ lệ dòng vượt ngưỡng của cả hai lô; kiểm lại file đã ghi theo phép tự kiểm của bước 6.

(c) Ba con số của phần (b) không phát hiện được lỗi nào. In ba con số của **từng lỗi một**, mỗi lần chỉ đổi một thứ so với lô mốc, rồi lập bảng bốn dòng. Trả lời: lỗi nào đổi xác suất nhiều nhất, và vì sao không lỗi nào đổi nhãn.

(d) Giữ nguyên giá trị 99999 nhưng chuyển nó sang dòng có xác suất cao nhất của lô mốc. In chỉ số dòng ấy, xác suất trước và sau, ba con số, recall và F1 của cả lô trước và sau. Viết hai câu kết luận: một câu về điều kiện làm cho cùng một lỗi dữ liệu trở nên đo được, một câu về phép kiểm nên dùng khi điều kiện ấy không thỏa mãn.

Tự kiểm: (a) 1 ô ngoài miền, 5 dòng mức lạ, 3 ô thiếu, 1 cột thừa; (b) lô mốc có 7 dòng vượt ngưỡng với F1 0,9231, ba con số là 0 trên 40 nhãn, 0,0493, `+0,0000`; (d) dòng chỉ số 22 đi từ 0,9587 về 0,0000, recall 1,0000 xuống 0,8333.

**Bài 8.9** (★★ · MTS3, MTS6 · \~60 phút · Kiến thức: mục 8.3.1, 8.3.2, 8.3.3 · dạng: viết mã) **\[HTTT\]**

Mục 4.4.1 đã loại `tinh_thanh` khỏi danh sách đặc trưng. Một nhóm của phòng đào tạo đề nghị đưa cột ấy trở lại, vì “nơi học ảnh hưởng tới điều kiện đường truyền”. Dựng mô hình triển khai 15 cột theo Quy trình 8.3 rồi tính chi phí của đề nghị đó.

(a) Đi bước 1 tới bước 5, in các con số của phép tự kiểm từng bước. Riêng bước 2, làm trên bản dữ liệu còn 313 ô thiếu ở 8 cột số: in độ lệch chuẩn từng cột sau biến đổi với thứ tự đúng và với thứ tự đảo, **nêu rõ** `ddof` **đang dùng**, và cho biết phép so ấy mất hiệu lực ở những cột nào.

(b) Chọn ngưỡng bằng `TunedThresholdClassifierCV(ong, scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True, random_state=42)`, in ngưỡng và `best_score_`, rồi đo đúng một lần trên 603 dòng kiểm tra: F1, recall, precision, ma trận nhầm lẫn, số dòng vượt ngưỡng.

(c) Đi bước 6: in **ba con số** của mô hình 15 cột so với mô hình 14 cột, kèm recall và số sinh viên bị bỏ sót của cả hai; đếm lại số giá trị phải khớp theo Mã 8.13 cho cấu hình 15 cột. Viết đoạn kết luận trả lời đề nghị của nhóm, nêu **cả cái mất**: độ đo đổi bao nhiêu, con số nghiệp vụ có đổi hay không, và số hạng mục phải bảo trì thêm.

(d) Gắn ngưỡng vào đối tượng theo Mã 8.15, lưu một file, nạp lại trong một tiến trình mới rồi dự đoán cho một bản ghi dựng bằng `pd.DataFrame([tu_dien])` với 15 khóa. In dung lượng file, ngưỡng sau khi nạp lại, kết quả so nhãn từng dòng với bản trong bộ nhớ, rồi dự đoán lại cùng bản ghi sau khi bỏ khóa `tinh_thanh` khỏi từ điển và cho biết điều gì xảy ra.

Tự kiểm: (a) 15 cột vào, 46 đặc trưng ra, `tinh_thanh` 16 mức; (b) ngưỡng 0,2525, `best_score_` 0,7142, F1 0,7209 với ma trận `[[438, 57], [15, 93]]`; (c) ba con số là 8 trên 603 nhãn, 0,1601, `−0,0056`, số bị bỏ sót giữ nguyên 15, và 125 giá trị phải khớp.

**Bài 8.10** (★★ · MTS3, MTS6 · \~50 phút · Kiến thức: mục 8.3.4, 8.2.2 · dạng: viết mã)

Khớp lại bộ tiền xử lý trong chương trình dự đoán là lỗi triển khai hay gặp nhất. Đo nó trên bốn cỡ lô, rồi dựng một phép kiểm chạy được khi chưa có nhãn thật.

(a) Với cỡ lô lần lượt 50, 100, 250 và 500 dòng đầu của tập kiểm tra: tính bản đúng bằng `tien.transform(lo)`, bản sai bằng `clone(tien).fit(lo).transform(lo)`. So danh sách tên đặc trưng của hai bản **trước** khi gọi mô hình; lô nào lệch thì in số đặc trưng của cả hai bản cùng tên đặc trưng bị mất rồi dừng lô đó, lô nào khớp thì in ba con số kèm recall trước và sau.

(b) Hai lô nhỏ dừng vì cùng một lý do. Nêu lý do, rồi trả lời: điều kiện phân biệt lô dừng với lô chạy sạch có phải là cỡ lô hay không. Dựng **hai ca** để bảo vệ câu trả lời (một lô nhỏ chạy sạch và một lô lớn vẫn dừng) bằng cách chọn dòng theo một cột phân loại thay vì lấy theo thứ tự.

(c) Dựng phép kiểm không cần nhãn thật: so `mean_` và `scale_` của bộ tiền xử lý đang dùng với giá trị trong file mô hình, **từng phần tử**. In mức lệch tương đối lớn nhất của từng đại lượng cho hai lô còn chạy được ở phần (a), rồi cho biết phép kiểm này phát hiện được lô nào mà ba con số không phát hiện được.

(d) Mục 8.3.4 đo mức lạm phát khi bước tiền xử lý nằm ngoài xác thực chéo: âm 0,0013 cho bước điền cộng chuẩn hóa và 0,0048 cho bản thêm chọn 5 đặc trưng, cả hai nhỏ hơn độ lệch chuẩn giữa các phần. Dùng hai con số ấy và kết quả phần (a) để viết hai câu: một câu nói rõ lý do dùng pipeline ở bài toán này **không** phải mức lạm phát của con số báo cáo, một câu nêu điều kiện mà dưới nó lạm phát trở nên lớn hơn độ lệch chuẩn giữa các phần, kèm con số của mục 7.4.3.

Tự kiểm: (a) lô 50 và lô 100 đều cho 29 đặc trưng thay vì 30 và mất `pl__nhom_tuoi_Trên 40`; lô 250 cho 3 trên 250 nhãn, 0,1377, `−0,0319`; lô 500 cho 10 trên 500 nhãn, 0,1894, `−0,0208`; (c) lô 250 cho mức lệch tương đối lớn nhất 0,1753 ở `mean_` và 1,1986 ở `scale_`.

Năm bài 8.15 tới 8.19 dưới đây thuộc tầng Vận dụng, giữ số riêng để khớp các câu dẫn trong thân chương.

**Bài 8.15** (★★ · MTS6 · \~30 phút · Kiến thức: mục 8.1.1, 8.1.2 · dạng: viết mã)

Bảng 8.2 đo dung lượng của năm đối tượng ở hai mức nén, **trên chuỗi đối tượng của Chương 7**, tức `ch07-mo-hinh-chot.joblib` và các tầng nằm trong nó, không phải mô hình triển khai của Chương 8. Dựng lại cả bảng ấy từ đúng chuỗi đó, rồi thêm hai cột thời gian mà thân chương chỉ nêu hai dòng.

(a) Dựng năm đối tượng, tất cả lấy từ `mo_hinh = joblib.load('models/ch07-mo-hinh-chot.joblib')`: `mo_hinh.estimator_.named_steps['mo_hinh']` (chỉ `LogisticRegression`); `mo_hinh.estimator_.named_steps['tien']` (chỉ bộ tiền xử lý); `mo_hinh.estimator_` (pipeline, không ngưỡng); `mo_hinh` (đối tượng có ngưỡng); và một pipeline rừng 200 cây dựng từ `clone` của bộ tiền xử lý ấy, khớp trên cùng tập huấn luyện. Lưu từng đối tượng hai lần, `compress=0` và `compress=3`, rồi in bảng mười dòng gồm tên đối tượng, mức nén, dung lượng theo kilobyte.

(b) Thêm hai cột: thời gian ghi và thời gian nạp, mỗi con số là trung vị của **năm** lần đo bằng `time.perf_counter`. Nêu rõ vì sao lấy trung vị của nhiều lần chứ không lấy một lần đo.

(c) Với mỗi đối tượng, tính tỉ lệ dung lượng giảm được và mức tăng thời gian ghi, rồi trả lời: ở ngưỡng dung lượng nào thì nén đáng làm. Câu trả lời phải là một ngưỡng có số, kèm điều kiện sử dụng (file được tải qua mạng hay được nạp lại nhiều lần trên cùng máy).

(d) Đối tượng có ngưỡng nặng hơn pipeline không ngưỡng vài kilobyte, trong khi mục 8.1.2 đo được rằng mất ngưỡng đổi 53 trên 603 nhãn. Dùng hai con số ấy để viết một câu về việc dung lượng file có dùng làm phép kiểm cho lỗi mất ngưỡng được hay không. Lưu ý hai tầng đối tượng phải lấy **trong cùng một chuỗi**: 7,6 KB với 9,7 KB cho chuỗi Chương 7, hoặc 8,7 KB với 10,0 KB cho chuỗi Chương 8; ghép 8,7 với 9,7 là ghép hai tệp của hai chuỗi khác nhau.

Các dung lượng tham chiếu của Bảng 8.2 lần lượt là 2,1; 5,6; 7,6; 9,7; 7.321,2 KB khi không nén và 1,3; 1,9; 2,7; 3,5; 1.482,4 KB khi nén mức 3. Số byte và thời gian có thể khác theo môi trường. Khi chấm, kiểm tra đúng cấu trúc đối tượng, mức nén, phương pháp đo và khả năng nạp lại; không dùng riêng dung lượng hoặc chiều chênh lệch thời gian để kết luận chọn sai đối tượng.

**Bài 8.16** (★★ · MTS6 · \~35 phút · Kiến thức: mục 8.2.1, 8.2.2 · dạng: viết mã)

Mã 8.8 thử tám cách làm sai trên cùng một bản ghi nhưng chỉ in sáu ca. Trong bài này, dùng ch08-mo-hinh-kem-nguong.joblib của chuỗi Chương 7, chưa có bộ điền giá trị thiếu, thay cho mô hình triển khai mặc định. Dựng lại tám ca và phân loại dấu hiệu.

(a) Viết hàm `thu_mot_ca(mo_hinh, ten, bang)` in một khối hai dòng cho mỗi ca, trong đó phân biệt được ba trường hợp: lệnh gọi dừng với `ValueError` và in dòng đầu của thông báo; lệnh gọi chạy xong và sinh cảnh báo, in tên lớp cảnh báo; lệnh gọi chạy xong và không sinh cảnh báo nào. Dùng `warnings.catch_warnings(record=True)` để giữ lại cảnh báo thay vì để nó đi ra `stderr`.

(b) Chạy hàm ấy trên chín bảng dựng từ `X_te.iloc[[0]]`: bản đúng hợp đồng làm mốc so sánh; thiếu cột `diem_giua_ky`; đảo thứ tự cả 14 cột; thêm một cột không có trong hợp đồng; `nganh` bằng `"Trí tuệ nhân tạo"`; `thiet_bi_chinh` để trống; `diem_giua_ky` để trống; đưa vào mảng NumPy thay cho `DataFrame`; `diem_giua_ky` bằng chuỗi `'3,6'`.

(c) Lập bảng ba nhóm theo đúng cách chia của Bảng 8.3, mỗi dòng ghi ca, dấu hiệu, xác suất nhận được và cách xử lý; đếm số ca của từng nhóm. Trả lời: nhóm nào đáng ngại nhất với một chương trình chạy theo lịch mà không ai xem kết quả, và vì sao số ca của một nhóm không nói gì về mức nguy hiểm của nó.

(d) Hai ca cho đúng xác suất của mốc so sánh tới chữ số cuối. Nêu hai ca ấy, nêu điều kiện làm cho chúng vô hại, rồi chỉ ra một thay đổi nhỏ trong chuỗi xử lý làm điều kiện ấy không còn được thỏa mãn.

Tự kiểm: (b) mốc so sánh cho 0,0369; đảo thứ tự và thêm cột lạ cũng cho 0,0369; mức lạ ở `nganh` cho 0,0456; `thiet_bi_chinh` để trống cho 0,0395; bốn ca còn lại dừng với `columns are missing`, `Input X contains NaN.`, `only supported for dataframes.` và `could not convert string to float: '3,6'`; (c) bốn ca dừng chương trình, hai ca không để lại dấu hiệu mà kết quả đúng, hai ca không để lại dấu hiệu mà kết quả lệch.

**Bài 8.17** (★★ · MTS6 · \~40 phút · Kiến thức: mục 8.2.1, 8.2.2, 8.2.3 · dạng: viết mã)

Mục 8.2.1 nêu bốn phần của hợp đồng dữ liệu vào và cho biết hai phần cuối không được thư viện kiểm gì. Viết một hàm kiểm hợp đồng dùng lại được, để Chương 9 và Chương 10 gọi lại mà không viết lại.

(a) Viết hàm doc_hop_dong(mo_hinh, ho_so). Đọc tên cột và tập mức phân loại từ các thuộc tính đã học của mô hình; đọc kiểu dữ liệu, khoảng quan sát khi huấn luyện và miền nghiệp vụ từ hồ sơ đã lưu trong lần huấn luyện. Không suy diễn các giới hạn này từ mean\_ hoặc scale\_. Hàm phải báo thiếu siêu dữ liệu cần thiết thay vì tự gán giá trị.

(b) Viết hàm `kiem_hop_dong(bang, hop_dong)` trả về **ba phép đếm**: phần lệch giữa hai tập tên cột, tách thành cột thiếu và cột thừa; số ô ngoài miền theo từng cột số; số dòng mang mức chưa có trong `categories_` theo từng cột phân loại. Hàm phải chạy được **trước** mọi lời gọi `predict`, và phải dừng với một thông báo của chính nó khi bảng vào có 0 dòng.

(c) Chạy `kiem_hop_dong` trên ba bảng: toàn bộ 603 dòng kiểm tra; lô 40 dòng của Bài 8.8; một lô rỗng dựng bằng `X_te.iloc[:0]`. In kết quả của cả ba, và với mỗi con số khác 0 ghi một dòng quyết định xử lý theo bước 3 và bước 4 của Quy trình 8.2.

(d) Hàm ở phần (b) đọc khoảng quan sát của tập huấn luyện từ hồ sơ, nên nó báo hai ô ngoài miền ngay trên chính tập kiểm tra. Trả lời: đó là lỗi của hàm, lỗi của dữ liệu, hay là kết quả đúng; rồi sửa phần in của hàm sao cho một sinh viên xem kết quả phân biệt được “ngoài miền và phải xử lý” với “ngoài miền và đã có quyết định”, dựa vào ba dấu hiệu của mục 3.2.1.

Tự kiểm: (a) 14 cột, 8 cột số và 6 cột phân loại với tổng 22 mức; (c) 603 dòng kiểm tra cho lệch tập tên cột rỗng, **2** ô ngoài miền (`tong_thoi_luong_phut` 2198 và `nop_tre_tb_gio` 644) và **0** dòng mức lạ; lô 40 dòng cho 1 cột thừa, 1 ô ngoài miền và 5 dòng mức lạ; lô rỗng dừng với thông báo của chính hàm chứ không để `SimpleImputer` báo hộ.

**Bài 8.18** (★★ · MTS6 · \~30 phút · Kiến thức: mục 8.2.3 · dạng: viết mã)

Ghi nhớ của mục 8.2.3 nêu một dải thời gian đo trên tám lần chạy. Dựng lại phép đo ấy và trả lời một câu hỏi về quy mô.

(a) Với ba cỡ lô 40, 150 và 603 dòng đầu của tập kiểm tra: đo thời gian gọi `predict_proba` một lần cho cả lô, và thời gian gọi một lần cho mỗi dòng trong một vòng lặp. Mỗi cấu hình đo **tám** lần, báo cáo trung vị cùng giá trị nhỏ nhất và lớn nhất, tính cả thời gian mỗi dòng.

(b) In mức lệch lớn nhất giữa hai mảng xác suất của hai cách gọi, rồi trả lời: con số ấy có nghĩa hai cách cho cùng kết quả hay không, và căn cứ là gì.

(c) Tỉ lệ chậm hơn có giữ nguyên qua ba cỡ lô hay không. Giải thích bằng chỗ phát sinh chi phí, tức lớp kiểm dữ liệu vào chạy một lần cho cả bảng hoặc $n$ lần cho $n$ bảng một dòng.

(d) Một dịch vụ nhận 1.200 bản ghi mỗi phút. Dùng dải đo của phần (a) tính thời gian hai cách cho 1.200 bản ghi, rồi nêu điều kiện nghiệp vụ mà dưới nó vẫn phải gọi lẻ dù chậm hơn.

Tự kiểm: (a) trên 603 dòng, gọi theo lô cho 0,015 tới 0,035 ms mỗi dòng và gọi lẻ cho 8,1 tới 11,2 ms mỗi dòng, tức chậm hơn 245 tới 667 lần; một lần chạy cho 20,9 ms cho cả lô và 6.758,4 ms khi gọi lẻ; (b) lệch lớn nhất $4,44 \times 10^{- 16}$. Thời gian trên máy của sinh viên sẽ khác; phần phải khớp là bậc độ lớn của tỉ lệ chậm hơn và mức lệch xác suất.

**Bài 8.19** (★★ · MTS3, MTS6 · \~40 phút · Kiến thức: mục 8.3.4 · dạng: viết mã)

Mục 8.3.4 đo mức lạm phát của kết quả báo cáo khi bước tiền xử lý nằm ngoài xác thực chéo, trên hai cấu hình. Thêm cấu hình thứ ba rồi rút ra quy tắc phân biệt.

(a) Với `skf = StratifiedKFold(5, shuffle=True, random_state=42)` và bản dữ liệu đã khoét lại 313 ô thiếu ở 8 cột số, đo F1 theo hai cách cho ba cấu hình tiền xử lý: điền trung vị cộng chuẩn hóa z; cấu hình ấy thêm `SelectKBest(f_classif, k=5)`; cấu hình ấy thêm SMOTE. Cách thứ nhất đặt bước tiền xử lý **trong** pipeline đưa vào `cross_val_score`; cách thứ hai học tiền xử lý trên toàn bộ dữ liệu rồi mới xác thực chéo. Với SMOTE, dùng fit_resample và giữ X, y đồng bộ, không dùng fit_transform cho bước lấy mẫu lại.

(b) Với mỗi cấu hình in các đại lượng: trung bình và độ lệch chuẩn của từng cách, mức lạm phát, số phần có hiệu dương trên 5, và mức lệch lớn nhất ở một phần. Lập bảng ba dòng.

(c) Hai cấu hình đầu cho lạm phát nhỏ hơn độ lệch chuẩn giữa các phần. Trả lời: con số báo cáo có phải lý do dùng pipeline ở bài toán này hay không, và nếu không thì lý do thật là gì, kèm một con số của mục 8.3.4 làm căn cứ.

(d) Viết quy tắc phân biệt cấu hình cho lạm phát nhỏ với cấu hình cho lạm phát lớn. Quy tắc phải nêu được **tính chất của bước tiền xử lý** chứ không nêu tên bước, và phải giải thích được cả ba dòng của bảng ở phần (b).

Tự kiểm: (b) điền cộng chuẩn hóa cho 0,6949 ± 0,0451 trong pipeline và 0,6936 ± 0,0459 ngoài pipeline, lạm phát âm 0,0013 với hiệu dương 0 trên 5 phần và lệch lớn nhất âm 0,0066; thêm chọn 5 đặc trưng cho 0,6530 ± 0,0288 và 0,6579 ± 0,0349, lạm phát 0,0048 với hiệu dương 2 trên 5 và lệch lớn nhất 0,0139; (c) và (d) con số tham chiếu của sách cho SMOTE đặt sai chỗ là 0,1809, **đo ở mục 7.4.3 trên một bộ chia và một bản dữ liệu khác phần (a)**, nên con số đo được ở đây sẽ khác; không dùng độ lớn hoặc dấu của mức chênh lệch làm điều kiện chấm bắt buộc. Cần kiểm tra đúng giao thức, báo cáo kết quả thực đo và giải thích phạm vi của quan sát.

### Thực tế và liên ngành

**Bài 8.11** (★★★ · MTS6 · \~45 phút · Kiến thức: mục 8.2.2, 8.2.3 · dạng: bảo vệ phương án, Điện tử – Viễn thông)

Công ty Điện tử An Khê đặt bộ thu dữ liệu trên 1.200 tủ thiết bị và muốn chạy một mô hình cảnh báo hỏng hóc ngay trên bộ thu. Bộ thu viết bằng C: nó đọc 30 giá trị cảm biến ra một mảng số thực rồi truyền mảng ấy sang một hàm Python. Nhóm phần mềm đề nghị giữ cách truyền mảng, vì “mảng nhanh hơn bảng và tiết kiệm bộ nhớ của bộ thu”. Dùng mô hình trần của dự án, tức bước `LogisticRegression` lấy từ bên trong pipeline với 30 đặc trưng sau biến đổi, để thay cho mô hình trên bộ thu. Cấu hình mô hình trần với ngưỡng áp bằng tay chỉ dùng ở đây để đo hậu quả của việc xếp sai thứ tự cột; nó không phải cách triển khai được khuyến nghị, vì mục 8.3.3 đã nêu đối tượng đem triển khai phải mang sẵn ngưỡng.

(a) Tính mốc so sánh bằng cách đưa `DataFrame` 30 đặc trưng vào mô hình trần: F1, recall, số dòng vượt ngưỡng ở ngưỡng 0,2727104755205902. Đưa cùng bảng ấy vào sau khi đảo thứ tự cột và in thông báo nhận được, **nguyên văn cả hai dòng**.

(b) Chuyển bảng sang mảng NumPy rồi thử bốn cách xếp sai thứ tự: dịch vòng toàn bộ cột một vị trí; đổi chỗ hai cột đầu; đổi chỗ `so__diem_tb_bai_tap` với `so__diem_giua_ky`; đảo thứ tự 8 cột số và giữ nguyên 22 cột một nóng. Mỗi cách in ba con số, recall và loại cảnh báo nhận được, rồi xếp bốn cách theo mức dễ phát hiện khi chỉ xem báo cáo độ đo hằng tháng.

(c) Bảo vệ phương án 5 phút trước hội đồng kỹ thuật. Trả lời đề nghị của nhóm phần mềm bằng ba căn cứ, mỗi căn cứ kèm một con số của phần (a) hoặc (b), và nêu rõ chi phí của phương án đã chọn: thứ gì phải thêm vào bộ thu, thứ gì không còn phải nêu lại ở hai chương trình.

(d) Hội đồng hỏi: “Mỗi tủ gửi 30 giá trị mỗi phút, tổng 1.200 tủ. Gọi một lần cho mỗi tủ hay gộp lại gọi một lần cho cả đợt?” Dựa vào Mã 8.12, tính thời gian hai cách cho 1.200 bản ghi theo hai mức đo của sách, rồi nêu điều kiện nghiệp vụ mà dưới nó phải gọi lẻ dù chậm hơn.

**Bài 8.12** (★★★ · MTS6 · \~50 phút · Kiến thức: mục 8.2.1, 8.2.2 · dạng: tình huống nghiệp vụ và vấn đáp, Ngôn ngữ Trung)

Trung tâm Hán ngữ Thanh Hà nhận file mô hình cảnh báo sớm của phòng đào tạo và muốn dùng cho học viên của mình. Học viên trung tâm theo ngành **Ngôn ngữ Trung**, một mức chưa có trong bảy mức của cột `nganh`; hồ sơ của trung tâm còn ghi `gioi_tinh` bằng `M`, `F`, `O`, ghi `nhom_tuoi` kèm hậu tố `" tuổi"`, và ghi `hinh_thuc_hoc` bằng chữ thường. Lô mới dựng từ 150 dòng chỉ số 100 tới 249 của tập kiểm tra, cho 40 dòng vượt ngưỡng với F1 0,7941.

(a) In hợp đồng dữ liệu vào đọc từ mô hình và hồ sơ đi kèm, theo Mã 8.7: 14 tên cột theo thứ tự, miền giá trị của 8 cột số, tập mức của 6 cột phân loại.

(b) Dựng bốn lô, mỗi lô mắc đúng một trong bốn cách ghi trên: 45 dòng đầu mang `nganh` bằng `"Ngôn ngữ Trung"`; cả cột `gioi_tinh` đổi sang `M`, `F`, `O`; cả cột `nhom_tuoi` thêm hậu tố `" tuổi"`; cả cột `hinh_thuc_hoc` viết chữ thường. Mỗi lô in ba con số và số dòng mang mức lạ theo từng cột; lập bảng bốn dòng xếp theo mức lệch xác suất.

(c) Một lô cho lệch F1 **đúng 0,0000** với 0 nhãn đổi. Nêu lô đó, rồi trả lời: vì sao kết quả ấy không có nghĩa là lô đó hợp lệ, và phải đổi lô ấy thế nào để cùng một loại sai trở nên đo được. Thực hiện phép đổi đó rồi in lại ba con số.

(d) Trả lời vấn đáp 3 phút. Trung tâm hỏi: “Có phải chỉ cần đổi cách ghi ba cột cho khớp mức của mô hình là dùng được không?” Trả lời có hoặc không cho **từng** cột trong bốn cột, nêu rõ cột nào sửa được bằng một phép đổi cách ghi, cột nào không, vì sao, và cách xử lý còn lại cho cột không sửa được.

**Bài 8.13** (★★★ · MTS1, MTS6 · \~180 phút cho cả nhóm · Kiến thức: mục 8.1.2, 8.1.3, 8.3.1 · dạng: làm nhóm ba người, có trình bày, Kinh tế)

Viện Nghiên cứu Kinh tế Lạc Việt nhận ba file mô hình từ ba nhóm cộng tác bên ngoài, kèm lời hứa “nạp rồi gọi `predict` là xong”. Viện yêu cầu dựng một quy trình nhận file mô hình từ bên ngoài trước khi cho phép nạp trên máy có dữ liệu thật.

(a) Dựng một file mô hình giả theo cơ chế `__reduce__` của Mã 8.5, trả về một lời gọi `print` với một chuỗi tham số **ngắn do nhóm tự chọn**; in dung lượng file và mã băm. Dung lượng sẽ khác 146 byte của Mã 8.5, và chính chênh lệch ấy là nội dung của câu hỏi. Dựng thêm một file giả thứ hai chỉ khác ở độ dài chuỗi tham số rồi dùng hai dung lượng để trả lời: dung lượng file có phân biệt được file mô hình thật với file chỉ chứa một lời gọi hàm hay không.

(b) Viết bộ nạp có hạn chế theo Mã 8.6, rồi **thu hẹp danh sách cho phép bằng cách đo**: lần lượt bỏ từng tên khỏi `BUILTINS_CHO_PHEP`, mỗi lần thử nạp lại cả chín file `.joblib` trong `models/`, ghi lại tên nào bỏ đi thì có file bị từ chối. In danh sách tối thiểu cần cho chín file ấy kèm số tên đã bỏ được, rồi thử bộ nạp đã thu hẹp trên hai file giả của phần (a).

(c) Dựng ba phép kiểm chạy được trước khi nạp hoặc ngay sau khi nạp, mỗi phép kiểm kèm một ca đáng phải bắt và một ca đáng phải bỏ qua: đối chiếu mã băm SHA-256 với dòng do bên gửi công bố; nạp bằng bộ nạp đã thu hẹp; biến `InconsistentVersionWarning` thành lỗi rồi in phiên bản ghi trong file. Với phép kiểm mã băm, dựng ca đáng phải bắt bằng cách đảo một bit trên một bản sao.

(d) Một nhóm cộng tác gửi kèm câu: “Mô hình đạt F1 0,7266, cứ nạp rồi gọi `predict` là ra đúng con số đó.” Dùng Mã 8.4 và Mã 8.13 để nêu **hai** điều kiện mà câu ấy bỏ qua, mỗi điều kiện kèm một con số đo được và một phép kiểm phát hiện ra khi nó không được thỏa mãn.

(e) Chia việc cho ba người: một người làm (a) và (b), một người làm (c), một người làm (d); cả nhóm làm bài trình bày. Nộp mã nguồn chạy lại được, bảng danh sách cho phép trước và sau khi thu hẹp, bảng ba phép kiểm với hai ca mỗi phép, và bài trình bày 7 phút trả lời một câu hỏi của viện: cho phép nạp file mô hình từ bên ngoài trên máy có dữ liệu thật hay không, nếu không thì thay bằng gì. Căn cứ phải là ba phép kiểm của phần (c) cùng hai giới hạn của bộ nạp có hạn chế.

### Thử thách có AI

**Bài 8.14** (★★★★ · MTS6 · \~100 phút · Kiến thức: toàn chương · dạng: viết bộ ca thử bắt buộc phải đạt trước, rồi đặt đề bài cho AI và chấm bản AI trả về)

**Bài toán.** Phòng đào tạo cần một chương trình dự đoán nhận một file CSV sinh viên mới và trả về danh sách cần liên hệ. Chương trình chạy theo lịch hằng tuần, không có người ngồi xem kết quả, nên mọi lỗi dữ liệu vào phải hoặc dừng chương trình, hoặc đi vào một dòng ghi nhận đếm được. Người viết chương trình là một trợ lý AI tạo sinh; việc của sinh viên là **đặt đề bài** và **chấm bản nhận được**.

**Bước một, trước khi mở công cụ AI: viết bộ ca thử bắt buộc phải đạt.** Bộ ca gồm **sáu** ca, mỗi ca ghi đủ bốn phần: lô dữ liệu vào dựng thế nào, hành vi đòi hỏi, con số kỳ vọng, câu lệnh kiểm. Sáu ca phải phủ đủ ba nhóm dấu hiệu của Bảng 8.3 và phải có ít nhất một ca mà **cả ba con số đều bằng 0 trong khi dữ liệu vào đã sai**. Chạy thật cả sáu ca trên `ch08-mo-hinh-trien-khai.joblib` để chốt con số kỳ vọng rồi niêm phong bộ ca lại; đây là thước đo của cả bài, không được sửa sau khi đã xem mã do công cụ sinh.

**Bước hai: đặt đề bài cho AI.** Viết một bản đặc tả rồi đưa cho một công cụ AI tạo sinh. Bản đặc tả phải nêu được, không bằng cách dẫn tên sách hay tên mục mà bằng yêu cầu kiểm chứng được: đối tượng nào được nạp và ngưỡng đọc từ đâu; hợp đồng dữ liệu vào gồm những phần nào và mỗi phần kiểm bằng phép đếm nào; cấm những lời gọi nào trong phần dự đoán; dạng dữ liệu đưa vào `predict`; file ra có những cột nào. **Không** dán bộ ca thử ấy vào bản đặc tả.

**Bước ba: chấm bản nhận được và sửa đặc tả.** Chạy mã nhận được trên cả sáu ca, ghi ca nào đạt ca nào trượt. Với mỗi ca trượt, sửa **bản đặc tả** chứ không sửa mã, rồi yêu cầu công cụ sinh lại; lặp tối đa ba vòng, ghi số ca đạt sau từng vòng. Hết ba vòng mà còn ca trượt thì tự sửa mã và ghi rõ chỗ sửa.

**Nhật ký prompt.** Nộp toàn văn ba bản đặc tả theo thứ tự, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi ứng với từng lần sinh lại, và bản niêm phong bộ ca thử của bước một. Chỉ dán tên cột, kiểu cột và số mức; không dán dữ liệu của sinh viên.

**Kiểm chứng.** Nộp bảng sáu ca × ba vòng, mỗi ô ghi đạt hoặc trượt kèm con số thực đo. Ngoài bảng ấy, trả lời bốn câu bằng số đo của chính bài làm:

1.  Mã nhận được ở vòng một có lời gọi `fit` hoặc `fit_transform` nào trong phần dự đoán hay không. Nếu có, đo hậu quả trên hai cỡ lô: một lô dừng với `ValueError` và một lô chạy sạch mà xác suất lệch, kèm ba con số của lô chạy sạch.
2.  Ngưỡng được áp thế nào: gắn vào đối tượng, đọc từ hồ sơ rồi so, hay viết cứng trong mã. Nếu viết cứng, đếm số chỗ con số ấy xuất hiện rồi đo số nhãn đổi khi dùng giá trị làm tròn bốn chữ số với phép so `>` thay cho `>=`. Con số đo được trên 603 dòng kiểm tra là một ca đáng chú ý: nêu nó, rồi nêu phép đếm nào mới phát hiện được lỗi này.
3.  Chương trình kiểm mức phân loại lạ bằng cách nào. Hỏi lại công cụ “chương trình này phát hiện mức phân loại lạ bằng cách nào” rồi đối chiếu câu trả lời với phép đếm trong mã; ghi lại khi câu trả lời viện tới `handle_unknown="ignore"` mà trong mã không có phép đếm nào.
4.  Ca “cả ba con số bằng 0 trong khi dữ liệu vào đã sai”: mã nhận được phát hiện được ca ấy hay không, và nhờ dòng nào.

**Phản tư** (100 tới 200 từ). Việc đặt đề bài cho công cụ AI khó hơn hay dễ hơn việc tự viết chương trình, đo bằng chính số vòng sinh lại của bài làm; yêu cầu nào trong bản đặc tả phải viết tới vòng thứ ba mới được thực hiện đúng và điều đó nói gì về thứ công cụ không tự nghĩ ra; rủi ro khi một chương trình chạy theo lịch hằng tuần gọi tên sinh viên mà không ai xem kết quả, kèm trách nhiệm với một lô dữ liệu đã sai mà ba con số đều báo không có gì xảy ra.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất lượng kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi tiết ở tài liệu hướng dẫn chấm đi kèm sách.
