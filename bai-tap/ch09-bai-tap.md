# Bài tập – Chương 9

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Mọi bài làm trên mô hình `models/ch08-mo-hinh-trien-khai.joblib` (10,0 KB) với ngưỡng 0,2727104755205902 đọc từ `models/ch08-ho-so-trien-khai.json`, và trên lô 40 dòng `data/raw/ch09-lo-moi.csv`; nhãn thật của lô nằm ở `data/raw/ch09-lo-moi-nhan-that.csv`. Từ tầng hai trở lên nộp mã nguồn, dòng lệnh đã gõ, **mã trả về** của từng lần chạy và kết quả in ra.

Câu hỏi “thay đổi này có ảnh hưởng không” phải in đủ **ba con số**: số dòng đổi nhãn trên tổng số dòng, mức lệch xác suất lớn nhất, và số dòng cần liên hệ. Chương này có nhiều ca mà chương trình chạy sạch và trả mã 0 trong khi dữ liệu vào đã sai, nên bài nào kiểm dữ liệu vào còn phải in **ba phép đếm** của Quy trình 9.2: phần lệch giữa hai tập tên cột, số ô không đọc được thành số theo từng cột số, và số dòng mang mức phân loại chưa có trong mô hình theo từng cột phân loại.

Mọi con số thời gian phụ thuộc máy. Bài nào đo thời gian phải nộp **ba lần đo** và dẫn kèm dải, nói rõ hệ điều hành và loại ổ đĩa; một giá trị đơn không được tính là kết quả.

### Củng cố

**Bài 9.1** (★ · MTS6 · \~12 phút · Kiến thức: mục 9.1.3, 9.2.1, 9.4.3 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một bản của chương trình xử lý trường hợp không thấy file vào bằng lệnh `sys.exit("không đọc được file vào: %s" % duong_dan)`. Mã trả về quan sát được của tiến trình: A. 3, vì chuỗi mô tả đúng tình huống mà Bảng 9.7 gán mã 3; B. 1, vì `sys.exit` nhận một chuỗi thông báo thì in chuỗi ra `stderr` rồi trả mã 1; C. 2, vì mọi lỗi liên quan tới đường dẫn đều do `argparse` xử lý; D. 0, vì chương trình kết thúc ngay sau khi in thông báo.

(2) Ở Bảng 9.2, dòng `pd.read_csv(p)` ghi chữ “không kiểm được” ở cột “Cột số không đọc được” thay cho một con số. Lý do: A. phép đếm cần `decimal=","` mới chạy được; B. file tách ra đúng 1 cột nên không còn cột nào mang tên của hợp đồng để đếm, và một con số 0 ở ô ấy sẽ là 0 vì không có gì để đếm chứ không phải vì dữ liệu sạch; C. phép đếm chỉ áp cho cột phân loại; D. `read_csv` đã nêu ngoại lệ nên phép đếm không chạy tới.

(3) Lịch chạy tự động gọi chương trình 10 lần, mỗi lần một dòng, thay cho một lần cho cả 10 dòng. Theo số đo của mục 9.1.3, thời gian tăng khoảng: A. 1,1 lần, vì phần dự đoán chiếm dưới 3,0% tổng; B. 9,8 tới 11,1 lần, xấp xỉ số lần gọi, vì phần cố định của mỗi lần gọi phải trả lại từ đầu; C. 40 lần, theo số dòng của lô mẫu; D. 245 tới 667 lần, theo tỉ số giữa dự đoán cả lô và dự đoán gọi lẻ.

**Bài 9.2** (★ · MTS6 · \~18 phút · Kiến thức: mục 9.1.1, 9.1.2, 9.3.1, 9.4.2 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số hoặc có một ca cụ thể. Phát biểu chỉ đúng trong một phạm vi thì nêu rõ điều kiện áp dụng còn thiếu.

(1) Chương trình chạy từ hai thư mục làm việc khác nhau mà cho cùng một kết quả thì mọi đường dẫn trong nó đã đủ bền để đem sang máy khác.

(2) Trong một lần chạy bất kỳ của chương trình dự đoán, `joblib.load` được gọi đúng một lần.

(3) Dẫn `stdout` của chương trình sang `pd.read_csv`: đọc được 40 dòng thì log đã đặt đúng chỗ, đọc không được thì log đã lẫn vào `stdout`.

(4) Tìm hết file chương trình mà không còn lệnh `print` nào thì mọi lệnh báo trạng thái đã được chuyển sang log.

(5) Phép tự kiểm ở bước 2 Quy trình 9.1 áp được cho mọi tham số của chương trình.

**Bài 9.3** (★ · MTS6 · \~15 phút · Kiến thức: mục 9.1.3 · dạng: tính toán)

Bốn phần của một lần chạy trên lô 40 dòng, lấy lại từ Bảng 9.1; mỗi lần đo trong một tiến trình mới.

**Bảng cho Bài 9.3.** Thời gian bốn phần của một lần chạy, đơn vị mi-li-giây.

  ------------------------------------------------------
  Phần                     Lần 1     Lần 2     Lần 3
  ------------------------ --------- --------- ---------
  nhập thư viện            2.819,7   2.785,8   2.479,4

  `joblib.load`            314,6     234,2     134,6

  `pd.read_csv`            34,7      13,3      5,7

  `predict_proba`          97,5      95,5      35,4

  tổng in trong Bảng 9.1   3.266,5   3.128,7   2.655,1
  ------------------------------------------------------

(a) Cộng bốn phần của từng lần đo rồi so với dòng tổng. Một lần đo lệch 0,1 ms: nêu lần đo đó và nguyên nhân của độ lệch.

(b) Lô thật của phòng đào tạo có 300 dòng. Giả thiết để tính: phần nhập thư viện và `joblib.load` không đổi theo số dòng, còn `pd.read_csv` và `predict_proba` tỉ lệ thuận với số dòng. Với từng lần đo, tính thời gian của một lần gọi cho cả 300 dòng, rồi tính thời gian của 300 lần gọi mỗi lần một dòng; nêu tỉ số giữa hai cách và dải của tỉ số ấy qua ba lần đo.

(c) Tính trần tiết kiệm nếu `predict_proba` trở thành tức thời, với lô 300 dòng, cho cả ba lần đo: ghi cả số mi-li-giây và phần trăm của tổng. Nêu phần đáng tối ưu trước và phần không đáng tối ưu.

(d) Lần chạy đầu sau khi mở máy cho `import sklearn` 15.648 ms, so với 2.237 ms ở lần chạy nóng. Tính tỉ số, rồi cho biết câu “chương trình chạy mất 2,7 giây” sai ở đâu khi in vào tài liệu hướng dẫn vận hành, và câu thay thế phải nêu thêm những gì.

**Bài 9.4** (★ · MTS6 · \~15 phút · Kiến thức: mục 9.2.3 · dạng: tính toán)

File `data/raw/ch09-lo-moi.csv` có 41 dòng văn bản và 40 dòng dữ liệu.

(a) Nêu số dòng trong file ứng với bản ghi `index` 0, rồi viết công thức tính số dòng trong file từ chỉ số của pandas. Công thức phải nói rõ dòng tiêu đề có được tính vào “số dòng bị bỏ ở đầu file” hay không, và phải cho ra đúng giá trị ở ca không bỏ dòng nào. Nêu thêm **điều kiện mà công thức ấy đúng**, và một cách đọc file làm điều kiện đó không thỏa.

(b) Thêm một dòng ghi chú `# lô tháng 11, phòng đào tạo gửi ngày 01/11/2026` vào đầu file rồi đọc bằng `pd.read_csv(p, comment="#")`. Tính lại số dòng văn bản, số dòng dữ liệu và độ lệch. Làm lại với `pd.read_csv(p, skiprows=1)` và cho biết độ lệch có đổi hay không.

(c) Chương trình báo `dòng 23 ty_le_hoan_thanh_video=3.5: ngoài miền nghiệp vụ [0.0; 1.0]`. Người nhận mở file bằng Excel, xem dòng 23 và thấy một dòng khác. Nêu hai nguyên nhân khả dĩ, mỗi nguyên nhân kèm một phép kiểm phân biệt được nó.

(d) Thông báo dùng được có bốn phần. Nêu bốn phần ấy, rồi cho biết vì sao giá trị được in qua `repr` thay cho `str`; kèm một cặp ô mà hai cách in cho hai thông báo không phân biệt được.

**Bài 9.5** (★ · MTS6 · \~12 phút · Kiến thức: mục 9.4.1, 9.4.3 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, rồi ghi thêm **mã trả về** mà lịch chạy tự động nhận được trong từng ca.

Dấu hiệu: (1) `du_doan: error: the following arguments are required: --vao`; (2) thông báo truy vết lỗi kết thúc bằng `FileNotFoundError: [Errno 2] No such file or directory` và trỏ vào một dòng của mô đun `logging`; (3) chương trình in bốn thông điệp log ở mức INFO, ghi ra một file kết quả rỗng, không có thông điệp nào ở mức ERROR; (4) log ghi `file chỉ có 1 cột, dấu phân cách có thể không phải dấu phẩy`; (5) log ghi `cả file diem_giua_ky=: thiếu cột trong hợp đồng`; (6) lệnh nhận `stdout` dừng với `ParserError: Error tokenizing data. C error: Expected 3 fields in line 4, saw 4`; (7) chương trình dừng với `UnicodeEncodeError` ngay khi `stdout` bị dẫn sang một file, dù việc dự đoán đã xong; (8) file kết quả có 38 dòng trên lô 40 dòng, không có thông điệp log nào nói hai dòng còn lại đã đi đâu.

Nguyên nhân: A. `argparse` từ chối một lời gọi thiếu tham số bắt buộc; B. thư mục chứa file log không tồn tại và ngoại lệ đi hết thân chương trình; C. một khối `except Exception` bọc cả thân chương trình rồi trả về giá trị của ca thành công; D. file xuất từ Excel đặt vùng Việt Nam dùng dấu chấm phẩy làm dấu phân cách; E. nguồn dữ liệu đổi tên một cột của hợp đồng; F. lệnh báo trạng thái viết bằng `print` nên đi ra cùng đường với kết quả; G. thiếu hai lệnh đặt lại bảng mã cho `stdout` và `stderr` trên cửa sổ lệnh Windows; H. một khối `except Exception` bọc vòng lặp theo từng dòng.

**Bài 9.6** (★ · MTS6 · \~15 phút · Kiến thức: mục 9.3.1, 9.3.2 · dạng: đọc bảng)

Mười dòng đầu của lô 40 dòng sau khi xếp theo nguy cơ giảm dần, lấy từ `output/ch09-danh-sach-lien-he.csv` và ba dòng tiếp sau nó.

**Bảng cho Bài 9.6.** Mười dòng đầu của lô 40 dòng theo nguy cơ giảm dần.

  ----------------------------------------------
  ma_sv       nguy_co_phan_tram   muc_uu_tien
  ----------- ------------------- --------------
  SV2602425   95,9                Liên hệ ngay

  SV2600105   95,5                Liên hệ ngay

  SV2601749   87,0                Liên hệ ngay

  SV2600335   55,5                Liên hệ ngay

  SV2600855   45,1                Cần xem lại

  SV2600393   36,2                Cần xem lại

  SV2601614   33,2                Cần xem lại

  SV2601466   9,5                 Chưa cần

  SV2602232   6,4                 Chưa cần

  SV2602134   5,2                 Chưa cần
  ----------------------------------------------

(a) Nêu hai điểm cắt chia ba mức ưu tiên và số dòng của từng mức trên cả lô 40 dòng. Cho biết mức nào là mức mà điểm cắt mặc định 0,50 bỏ sót, kèm số dòng.

(b) Tính khoảng cách giữa dòng cuối cùng còn trong danh sách giao đi và dòng đầu tiên nằm ngoài danh sách, đơn vị điểm phần trăm. Làm lại phép tính cho trường hợp danh sách chỉ lấy các dòng từ 50,0% trở lên. So hai khoảng cách rồi cho biết bảng nào cần thêm một dòng ghi chú về số sinh viên nằm sát dưới ngưỡng.

(c) Nêu bốn thay đổi của bảng giao đi so với bảng kỹ thuật bốn cột. Với mỗi thay đổi, nêu điều người nhận hiểu sai nếu không có nó.

(d) Một nhóm đề nghị thêm cột `xac_suat_tho` bên cạnh cột `nguy_co_phan_tram` để “giữ lại con số gốc”. Nêu hậu quả theo mục 9.3.2 và cách giữ con số gốc mà không gây hậu quả ấy.

### Vận dụng

**Bài 9.7** (★★ · MTS6 · \~60 phút · Kiến thức: mục 9.1.1, 9.1.2, 9.1.3, 9.4.3 · dạng: viết mã) **\[HTTT\]**

Dựng chương trình dự đoán cho phòng đào tạo theo trọn bảy bước Quy trình 9.1, rồi đo nó bằng tám tình huống của Bảng 9.7.

(a) Viết chương trình nhận `--vao`, `--ra`, `--log-muc`, `--file-log` và `--bo-dong-loi`. Đường dẫn file mô hình và file hồ sơ dựng từ `__file__`; mã băm SHA-256 của `models/ch08-mo-hinh-trien-khai.joblib` kiểm **trước** khi nạp. Nộp phép tự kiểm của bước 1: chạy hai lần từ hai thư mục làm việc khác nhau với `--vao` và `--ra` là đường dẫn tuyệt đối, rồi so hai file ra bằng mã băm.

(b) Áp phép tự kiểm của bước 2 cho năm tham số trên và cho hai tham số bị từ chối là `--nguong` và `--diem-cat-giua`. Phép kiểm chỉ áp cho tham số cấu hình nên hai tham số chỉ nguồn dữ liệu, `--vao` và `--ra`, nằm ngoài phạm vi: nói rõ điều đó trong bài làm. Với `--nguong`, đo hậu quả nếu mở nó: in số dòng cần liên hệ, recall và precision trên nhãn thật ở hai giá trị 0,2727104755205902 và 0,50.

(c) Áp bước 3 và bước 7: chạy lô sạch với `--log-muc ERROR` và không truyền `--ra`, đếm số dòng của `stdout`, rồi dẫn `stdout` sang `pd.read_csv` và in kích thước bảng đọc được. Lập bảng tám tình huống của Bảng 9.7, mỗi dòng ghi dòng lệnh đã gõ, mã trả về quan sát được và thông điệp log cuối cùng.

(d) Áp bước 4: đếm số lần `joblib.load` được gọi trong một lần chạy thành công và trong một lần chạy dừng ở tình huống thiếu tham số `--vao`. Đo thêm thời gian của lần nạp đầu và của lần gọi `nap_mo_hinh` thứ hai trong cùng tiến trình, ba lần đo mỗi loại, rồi cho biết con số ấy có là lý lẽ cho chương trình dòng lệnh hay không.

Tự kiểm: (a) hai file ra cùng mã băm; (b) ngưỡng 0,2727104755205902 cho 7 dòng cần liên hệ với recall 1,0000 và precision 0,8571, điểm cắt 0,50 cho 4 dòng với recall 0,6667 và precision 1,0000; (c) `stdout` có 41 dòng và `pd.read_csv` đọc được bảng 40 dòng 4 cột; tám tình huống cho mã 0 · 1 · 2 · 2 · 3 · 3 · 4 · 5; (d) 1 lần và 0 lần.

**Bài 9.8** (★★ · MTS6 · \~55 phút · Kiến thức: mục 9.2.1, 9.2.2 · dạng: viết mã)

Đi trọn Quy trình 9.2 trên lô có lỗi, rồi dựng ba lô mới để xác định phạm vi của hai kết luận trong mục 9.2.2.

(a) Chạy bảy bước Quy trình 9.2 trên `data/raw/ch09-lo-loi.csv`. In ba phép đếm của hợp đồng, rồi in danh sách lỗi và danh sách cảnh báo, mỗi phần tử đủ bốn phần: số dòng trong file, tên cột, giá trị và lý do.

(b) Dựng lô B từ lô sạch: đặt chuỗi `'mười'` vào cột `so_lan_dang_nhap` ở **hai dòng có xác suất thấp nhất lô**. Lập lại bảng ba quyết định của Bảng 9.3 cho lô B, rồi so với bảng của mục 9.2.2. Nêu đại lượng phải đo để biết đang ở lô nào, và viết lại kết luận “đổi ô sai thành ô thiếu tốt hơn bỏ dòng” kèm phạm vi áp dụng.

(c) Dựng lô C: đặt `'mười'` vào cột `diem_giua_ky` ở dòng 16 của file. Lập lại bảng ba quyết định. Hai phương án cho cùng số sinh viên bị bỏ sót; nêu điều phân biệt chúng, và nêu chỗ trong file kết quả hoặc trong log mà người nhận thấy được sự phân biệt ấy.

(d) Dựng lô D: giữ nguyên tên cột `diem_giua_ky` nhưng để cả 40 ô của cột đó rỗng. Chú ý mục 9.2.1 chốt rằng ngưỡng cảnh báo phải đặt trên **tỉ lệ ô rỗng**, vì cả số nhãn đổi và mức lệch xác suất đều đòi một lô sạch để đối chiếu mà lúc chạy thật thì không có. Hãy dựng phép cảnh báo ấy: một hàm nhận bảng vào và danh sách 14 cột của hợp đồng, trả về danh sách cột có tỉ lệ ô rỗng đạt ngưỡng. Hiệu chuẩn ngưỡng bằng đường cong năm điểm ở Chú ý mục 9.2.1, nêu lý do chọn mức ấy, rồi đo **tỉ lệ báo nhầm** của nó trên bốn lô: lô sạch, lô B, lô C và lô D. Cuối cùng nêu một lô mà phép cảnh báo này bỏ lọt.

(e) Bước 4 Quy trình 9.2 nêu ba ca thử và hai hướng hỏng, trong đó hướng thứ hai đã có số đo trên `ch09-lo-loi.csv`. Hãy dựng **ca thử thứ tư** nhắm vào bước 3 chứ không vào bước 4: đặt một **trường trống** vào ba ô của cột `diem_giua_ky` rồi đọc file hai lần, một lần bằng `pd.read_csv` mặc định và một lần thêm `keep_default_na=False`. Với mỗi cách đọc, in số ô mà `isna()` cho là rỗng và số ô mà phép kiểm của bước 3 xếp vào “ô sai kiểu”. Nêu cách đọc làm phép tự kiểm của bước 3 kết luận sai, giải thích vì sao, rồi viết lại bộ ca thử của bước 3 và bước 4 thành bốn ca kèm bảng ca thử × bản mắc lỗi, mỗi ô ghi đạt hay trượt.

Tự kiểm: (a) 3 lỗi ở dòng 5, 32, 23 và 1 cảnh báo ở dòng 18; (b) hai phương án đều bỏ sót 0 trên 7 với 0 trên 40 dòng đổi nhãn và mức lệch xác suất lớn nhất 0,0002; (c) cả hai bỏ sót 1 trên 7, phương án điền cho 1 trên 40 dòng đổi nhãn và mức lệch 0,1877; (d) tỉ lệ ô rỗng lớn nhất là 0,000 ở cả lô sạch, lô B và lô C, và 1,000 ở cột `diem_giua_ky` của lô D, nên ngưỡng 0,20 cho 0 báo nhầm trên ba lô đầu và bắt được lô D; (e) với cách đọc mặc định, 3 ô rỗng và 0 ô sai kiểu; với `keep_default_na=False`, 0 ô rỗng và 3 ô sai kiểu.

**Bài 9.9** (★★ · MTS6 · \~45 phút · Kiến thức: mục 9.4.2 · dạng: viết mã)

Chuyển một chương trình dự đoán đang báo trạng thái bằng `print` sang log theo Quy trình 9.3, rồi đo hai phép tự kiểm của quy trình ấy.

(a) Bản xuất phát có năm lệnh `print`: số dòng đã đọc, tên file mô hình, số dòng vượt ngưỡng, bảng kết quả, tên file đã ghi. Đi sáu bước Quy trình 9.3. Nộp bảng năm lệnh gồm nhóm, mức log đã chọn và lý do chọn mức ấy.

(b) Áp bước 4 và bước 3: chạy lô sạch với `--log-muc ERROR` và không truyền `--ra`, đếm số dòng của `stdout`; rồi gộp log vào `stdout` bằng một bộ ghi log trỏ về `sys.stdout` và dẫn `stdout` sang `pd.read_csv`, nộp thông báo nguyên văn.

(c) Bước 5 Quy trình 9.3 chốt phép kiểm ở **dạng của mọi đối số**. Đo chi phí của hai cách viết bằng cách gọi 200.000 lần ở mức `ERROR`, ba lần đo mỗi cách, in dải và tỉ số; lặp phép đo với **bốn thứ được nội suy khác nhau**: một chuỗi ngắn, một số nguyên, một từ điển ba khóa, một từ điển 14 khóa. Lập bảng bốn dòng, rồi trả lời một câu: vì sao không thể nêu một tỉ số duy nhất cho cách viết ấy. Cuối cùng áp phép kiểm của bước 5 vào **cả mười lệnh ghi log của** `code/69_du_doan.py`: in dạng từng đối số và kết luận đạt hay trượt, rồi cho biết bảy lệnh mang lời gọi hàm ở các đối số sau có vi phạm phép kiểm hay không, và vì sao.

(d) Phép tự kiểm viết lại ở bước 6 đòi mọi đối số của các lệnh mức `DEBUG` và `INFO` là một số đếm, một đại lượng tổng hợp hay một tên file. Chạy ba bản ở mức `DEBUG` với `--file-log`: bản in cả bản ghi, bản in cả bản ghi sau khi loại cột `ma_sv`, và bản chỉ in số dòng trong file. Với mỗi bản, in số dòng `DEBUG`, số mã sinh viên thật tìm thấy, số ô dữ liệu cá nhân và dung lượng file log kèm tên file. Rồi áp phép tự kiểm ấy vào **chính** `ghi_loi` **của Mã 9.5** khi chạy `ch09-lo-loi.csv` với `--file-log`, và trả lời ba câu: phép kiểm kết luận gì, kết luận ấy có đúng không, và phạm vi đúng của phép kiểm là gì.

Tự kiểm: (b) `stdout` có 41 dòng; (c) tỉ số **tăng theo giá dựng chuỗi của thứ được nội suy** và trải rất rộng, nên bản thân dải của nó cũng đổi theo máy: trên máy đã dùng để đo, khoảng 1 tới 2 lần với một chuỗi ngắn, khoảng 1 tới 2 với một số nguyên, khoảng 7 tới 12 với từ điển ba khóa và khoảng 20 tới 33 với từ điển 14 khóa; thêm một dòng tham chiếu nữa là lệnh log mà `code/73_log_va_ma_tra_ve.py` đo, một chuỗi ngắn kèm một số nguyên, cho khoảng 1,6 tới 2,8 lần. Hai dải cuối tự chúng đã trải rộng qua các lần chạy, nên bài làm nộp một dải nằm trong cùng bậc độ lớn là đạt. Bài làm nêu **một** con số mà không nói nội suy thứ gì thì chưa đạt, còn bài làm cho một dải lệch với dải trên mà nêu rõ máy và nộp đủ ba lần đo thì đạt; mười lệnh log của `code/69` đều đạt phép kiểm của bước 5. (d) bản thứ hai có 0 mã sinh viên thật mà vẫn mang 560 ô dữ liệu cá nhân, và `ghi_loi` của Mã 9.5 trượt phép kiểm ở các dòng mức `ERROR` nhưng đó là ngoài phạm vi phép kiểm.

**Bài 9.10** (★★ · MTS6 · \~40 phút · Kiến thức: mục 9.2.3, 9.3.2 · dạng: viết mã)

Làm hai sản phẩm mà người nhận không làm kỹ thuật đọc được: thông báo lỗi theo bước 7 Quy trình 9.2 và bảng giao đi theo bước 6 Quy trình 9.1.

(a) Viết hàm `so_dong_trong_file(chi_so, so_dong_bo_o_dau)` trả về số dòng trong file của một bản ghi. Nộp ba ca thử: file không bỏ dòng nào, file có một dòng ghi chú đọc bằng `comment="#"`, và file có hai dòng ghi chú. Mỗi ca ghi giá trị kỳ vọng và cách lấy giá trị ấy từ chính file.

(b) Viết hàm ghi thông báo in tối đa năm lỗi rồi cho biết còn bao nhiêu. Đo nó trên `data/raw/ch09-lo-excel-viet.csv` đọc bằng `sep=";"` mà không khai `decimal=","`: in số lỗi, số cột sinh ra lỗi, số thông điệp log thực tế, rồi suy ra hai con số ấy cho một lô 300 dòng cùng dạng. Cho biết vì sao log mất tác dụng nếu bỏ trần năm thông điệp.

(c) Sinh bảng giao đi từ bảng kết quả bốn cột: giữ các dòng cần làm gì, xếp theo nguy cơ giảm dần, một chữ số thập phân, nhãn bằng chữ. In số dòng của bảng giao đi, khoảng cách giữa dòng cuối trong bảng và dòng đầu ngoài bảng, rồi quyết định có thêm dòng ghi chú về nhóm nằm sát dưới ngưỡng hay không.

(d) Phép tự kiểm ở bước 6 Quy trình 9.1 đòi số dòng ra bằng số dòng vào trừ số dòng đã bỏ. Cho biết phép kiểm ấy áp cho file nào trong hai file vừa sinh, và nêu phép kiểm thay thế cho file còn lại.

Tự kiểm: (a) ba ca cho độ lệch 2, 3 và 4; (b) 160 lỗi ở 4 cột, log in 6 thông điệp; (c) bảng giao đi có 7 dòng với khoảng cách 23,7 điểm phần trăm.

### Thực tế và liên ngành

**Bài 9.11** (★★★ · MTS6 · \~45 phút · Kiến thức: mục 9.4.2, 9.4.3 · dạng: bảo vệ phương án, Luật)

Văn phòng Luật Trường An dùng một mô hình để xếp thứ tự hồ sơ cần thẩm định trước. Bộ phận pháp chế đặt hai yêu cầu: file log được sao lưu cùng hệ thống nên không được mang dữ liệu nhận dạng được cá nhân, và mọi lần chạy phải để lại một trạng thái mà hệ thống giám sát đọc được. Phần đo được của bài làm trên lô 40 dòng của chương.

(a) Lập bảng ba file log như ở Bài 9.9(d), thêm một cột “yêu cầu của pháp chế: đạt hay không đạt”. Cho biết bản nào đạt phép tự kiểm của bước 6 Quy trình 9.3 mà không đạt yêu cầu của pháp chế, kèm số ô dữ liệu cá nhân.

(b) Thiết kế lược đồ log chỉ cho phép các trường cần thiết cho vận hành và không ghi giá trị ô dữ liệu. Viết phép kiểm tự động cho cấu trúc lẫn nội dung log, dùng thêm giá trị đánh dấu để phát hiện việc phản chiếu đầu vào. Giải thích vì sao một lệnh tìm mã sinh viên, hoặc tìm các chuỗi đủ dài, chưa chứng minh log không có dữ liệu cá nhân.

(c) Hệ thống giám sát chỉ đọc mã trả về. Chia tám tình huống của Bảng 9.7 thành hai nhóm theo việc người nhận phải làm, rồi viết quy định vận hành: nhóm nào gửi lại file, nhóm nào gọi người trực, và chuỗi điều kiện mà hệ thống giám sát dùng để phân nhóm.

(d) Chuẩn bị bảo vệ phương án 5 phút trước bộ phận pháp chế. Trả lời ý kiến “cứ để mức `DEBUG` cho dễ tra khi có khiếu nại” bằng ba con số của phần (a) và một phương án thay thế vẫn tra lại được, theo khối Chú ý ở mục 9.4.2.

**Bài 9.12** (★★★ · MTS6 · \~50 phút · Kiến thức: mục 9.2.1, 9.3.1, 9.4.3 · dạng: tình huống nghiệp vụ và vấn đáp, Kế toán)

Công ty Dịch vụ Kế toán Phúc Khang chạy một chương trình dự đoán vào ngày cuối mỗi tháng. Lô đầu vào do kế toán viên xuất từ Excel đặt vùng Việt Nam, và lịch chạy đặt lúc 23 giờ, không có người ngồi xem.

(a) Lập lại bảng bốn cách đọc `data/raw/ch09-lo-excel-viet.csv` theo Bảng 9.2. Nêu hai cách sai mà `read_csv` không nêu ngoại lệ, dấu hiệu phát hiện được từng cách, và lý do một ô của bảng phải ghi “không kiểm được” thay cho con số 0.

(b) In ba byte đầu của hai file kết quả, một ghi bằng `utf-8-sig` và một ghi bằng `utf-8`, kèm tên từng file. Cho biết đường ra nào cần bảng mã nào và vì sao, rồi viết phép kiểm tự động phân biệt hai file; nói rõ vì sao phép kiểm “đọc lại bằng `pd.read_csv` rồi so số dòng” không phân biệt được.

(c) Lịch chạy lúc 23 giờ gửi thư báo động theo mã trả về. Lập bảng quy định: với mỗi mã của Bảng 9.7, ghi việc phải làm, người nhận thư và thời hạn. Nêu mã nào không được gộp với mã nào và vì sao.

(d) Chuẩn bị trả lời vấn đáp 3 phút. Công ty đề nghị “cho chương trình luôn trả 0 để lịch chạy không gửi thư báo động lúc nửa đêm, sáng mai xem log là được”. Trả lời bằng hai hậu quả đo được, mỗi hậu quả kèm một con số của Bảng 9.6 hoặc của phần (a).

**Bài 9.13** (★★★ · MTS6 · \~180 phút cho cả nhóm · Kiến thức: mục 9.1.1, 9.2.2, 9.3.2, 9.4.3 · dạng: làm nhóm ba người, có trình bày, Quản trị kinh doanh)

Công ty Bán lẻ Vạn Xuân muốn dự báo khách hàng thành viên sắp rời đi, rồi gửi danh sách cho bộ phận chăm sóc khách hàng vào ngày 1 mỗi tháng. Bộ phận ấy làm việc trên Excel và không viết mã. Nhóm ba người dựng trọn chương trình, dùng mô hình và lô 40 dòng của chương làm vật liệu.

(a) Người thứ nhất làm bảy bước Quy trình 9.1 và bảng tám tình huống của Bảng 9.7.

(b) Người thứ hai làm bảy bước Quy trình 9.2, dựng ba lô đối kháng của Bài 9.8 và bảng ba quyết định cho từng lô, rồi đề nghị một quyết định mặc định kèm lý do bằng số.

(c) Người thứ ba làm bảng giao đi theo mục 9.3.2, sáu bước Quy trình 9.3 và quy định vận hành theo mã trả về.

(d) Nộp chung: mã nguồn chạy lại được từ một thư mục làm việc bất kỳ, bảng tám tình huống, bảng giao đi, file log của một lần chạy ở mức `INFO`, và bài trình bày 7 phút trả lời một câu hỏi: bộ phận chăm sóc khách hàng mỗi tháng gọi được 10 khách, vậy chương trình nên nhận thêm một tham số giới hạn số dòng giao đi, hay nên giao trọn danh sách kèm thứ tự ưu tiên? Căn cứ phải là phép tự kiểm ở bước 2 Quy trình 9.1 và số dòng của từng mức ưu tiên, không phải cảm nhận về độ tiện.

(e) Mỗi người nộp thêm một đoạn 100 từ: phần của hai người kia có chỗ nào mà phần của mình phụ thuộc vào, và chỗ ấy đã được kiểm bằng gì.

### Thử thách có AI

**Bài 9.14** (★★★★ · MTS6 · \~100 phút · Kiến thức: toàn chương · dạng: chấm bản do AI viết bằng ma trận tình huống và mã trả về, rồi sửa các lỗi không sinh dấu hiệu)

**Bài toán.** Phòng đào tạo nhờ một trợ lý AI tạo sinh viết chương trình dự đoán và nhận về đoạn mã dưới đây, kèm bốn câu báo cáo: “chương trình đã chạy thành công trên lô 40 dòng và trả mã 0”; “mọi ngoại lệ đều được xử lý nên chương trình không bao giờ dừng giữa đường”; “miền hợp lệ của từng cột lấy đúng theo khoảng quan sát được trên tập huấn luyện nên không lọt giá trị bất thường”; “ngưỡng để mặc định 0,5 là điểm cắt tiêu chuẩn của bài toán phân loại hai lớp”. Phần nhập thư viện giữ nguyên như các chương trình của chương.

>     MO_HINH = Path("models/ch08-mo-hinh-trien-khai.joblib")
>     HO_SO = Path("models/ch08-ho-so-trien-khai.json")
>     KHOANG_QUAN_SAT = {"so_lan_dang_nhap": (1, 100),
>                        "tong_thoi_luong_phut": (0, 4000),
>                        "ty_le_hoan_thanh_video": (0.0, 1.0),
>                        "so_bai_tap_nop": (0, 12),
>                        "diem_tb_bai_tap": (0.0, 10.0),
>                        "so_lan_dien_dan": (0, 40),
>                        "nop_tre_tb_gio": (0, 240),
>                        "diem_giua_ky": (0.0, 10.0)}
>     log = logging.getLogger(__name__)
>
>
>     def main(dong_lenh=None) -> int:
>         bo = argparse.ArgumentParser(prog="du_doan")
>         bo.add_argument("--vao", type=Path, required=True)
>         bo.add_argument("--ra", type=Path, default=None)
>         bo.add_argument("--nguong", type=float, default=0.5)
>         bo.add_argument("--log-muc", default="INFO")
>         ts = bo.parse_args(dong_lenh)
>         sys.stdout.reconfigure(encoding="utf-8")
>         sys.stderr.reconfigure(encoding="utf-8")
>         logging.basicConfig(level=getattr(logging, ts.log_muc),
>                             format="%(levelname)s %(message)s")
>         try:
>             if not ts.vao.is_file():
>                 sys.exit("không đọc được file vào: %s" % ts.vao)
>             bang = pd.read_csv(ts.vao)
>             print("đã đọc %d dòng, %d cột" % (len(bang), bang.shape[1]))
>             ho_so = json.loads(HO_SO.read_text(encoding="utf-8"))
>             cot = ho_so["cot_vao_theo_thu_tu"]
>             loi = []
>             for c, (duoi, tren) in KHOANG_QUAN_SAT.items():
>                 gt = pd.to_numeric(bang[c], errors="coerce")
>                 for i in bang.index[(gt < duoi) | (gt > tren)]:
>                     loi.append((i + 2, c, gt[i]))
>             if loi:
>                 for dong, c, gt in loi:
>                     log.error("dòng %d, cột %s, giá trị %s: ngoài"
>                               " khoảng cho phép", dong, c, gt)
>                 return 4
>             ket_qua = []
>             for i, ban_ghi in bang.iterrows():
>                 log.debug("bản ghi: %s", ban_ghi.to_dict())
>                 mo_hinh = joblib.load(MO_HINH)
>                 p = mo_hinh.predict_proba(
>                     pd.DataFrame([ban_ghi])[cot])[0, 1]
>                 ket_qua.append({
>                     "ma_sv": ban_ghi["ma_sv"],
>                     "nguy_co_phan_tram": round(p * 100, 1),
>                     "can_lien_he": "Có" if p >= ts.nguong else "Không",
>                 })
>             ra = pd.DataFrame(ket_qua).sort_values(
>                 "nguy_co_phan_tram", ascending=False)
>             print("đã dự đoán xong %d dòng" % len(ra))
>             if ts.ra is None:
>                 ra.to_csv(sys.stdout, index=False)
>             else:
>                 ra.to_csv(ts.ra, index=False, encoding="utf-8-sig")
>             return 0
>         except Exception as e:
>             log.error("có lỗi: %s", e)
>             return 0

**Nhiệm vụ với AI.** Ba bước, theo đúng thứ tự. Bước một, **chưa** mở công cụ AI: đọc đoạn mã và bốn câu báo cáo, rồi dựng một **ma trận tình huống** gồm mười dòng và niêm phong nó lại. Tám dòng đầu là tám tình huống của Bảng 9.7; hai dòng còn lại do sinh viên tự thêm, trong đó một dòng là một lần chạy từ thư mục làm việc khác theo bước 1 Quy trình 9.1, và một dòng là một ca mà chương trình chạy sạch trong khi dữ liệu vào đã sai. Mỗi dòng ghi đủ: lô dựng thế nào, mã trả về đòi hỏi, số dòng của file kết quả đòi hỏi, và câu lệnh kiểm. Bước hai: đưa đoạn mã và bốn câu báo cáo cho một công cụ AI tạo sinh, yêu cầu nó liệt kê mọi lỗi; **không** dán ma trận tình huống vào prompt. Bước ba: chạy thật đoạn mã trên cả mười dòng của ma trận, rồi sửa lần lượt từng lỗi và chạy lại cả ma trận sau mỗi lần sửa.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi ứng với từng lỗi, cùng bản niêm phong của bước một. Chỉ dán tên cột, kiểu cột và số mức; không dán dữ liệu của sinh viên.

**Kiểm chứng.** Đoạn mã và bốn câu báo cáo có nhiều hơn sáu chỗ sai, và **không chỗ nào làm chương trình dừng kèm thông báo truy vết lỗi**. Sáu câu hỏi dưới đây xác định sáu chỗ phải phát hiện và phải đo bằng số; mỗi chỗ sai tìm thêm được chỉ tính khi nộp kèm một dòng của ma trận tình huống phát hiện ra nó. Nộp ma trận mười dòng × ba cột trạng thái: bản nhận được, bản sau khi sửa ba chỗ đầu, bản sau khi sửa cả sáu. Ngoài ma trận ấy, trả lời sáu câu bằng số đo của chính bài làm.

1.  Chạy trên `data/raw/ch09-lo-thieu-cot.csv`. Nộp mã trả về, số dòng của file kết quả và thông điệp log duy nhất mà chương trình để lại. Cho biết lịch chạy tự động kết luận gì từ mã ấy, rồi nêu dòng mã sinh ra hậu quả đó và cách sửa theo dòng thứ ba của Bảng 9.6.
2.  Chạy trên một lô có một ô `so_lan_dang_nhap` bằng 104. Nộp mã trả về và số dòng cần liên hệ của cả bản nhận được và bản của Quy trình 9.2, rồi cho biết câu báo cáo thứ ba của công cụ sai ở đâu, theo khối Chú ý ở mục 9.2.1.
3.  Chạy trên `data/raw/ch09-lo-loi.csv`. Bản nhận được báo mấy lỗi, bản của Quy trình 9.2 báo mấy lỗi, và những dòng nào bị bỏ lọt? Chỉ ra biểu thức chịu trách nhiệm, giải thích bằng giá trị mà phép so trả về khi một toán hạng là giá trị thiếu, rồi sửa bằng bước 3 Quy trình 9.2.
4.  Chạy lô sạch mà không truyền `--ra`, dẫn `stdout` sang `pd.read_csv`. Nộp thông báo nguyên văn, rồi nộp lại kích thước bảng đọc được sau khi sửa theo bước 2 và bước 3 Quy trình 9.3.
5.  So danh sách giao đi của bản nhận được với bản để ngưỡng của hồ sơ: in **ba con số**, rồi in recall và precision của cả hai trên nhãn thật. Cho biết bước 2 Quy trình 9.1 từ chối tham số `--nguong` vì lý do gì, và vì sao một giá trị mặc định “tiêu chuẩn” không xử lý được lý do ấy.
6.  Chạy ở mức `DEBUG` với đường ra log dẫn vào một file. Nộp số dòng `DEBUG`, số mã sinh viên thật tìm thấy, số ô dữ liệu cá nhân và dung lượng file log kèm tên file. Đo thêm thời gian một lần chạy của bản nhận được và của bản nạp mô hình một lần, ba lần đo mỗi bản, rồi nộp dải và tỉ số.

Cuối cùng so ba danh sách: lỗi nào tự phát hiện được mà công cụ bỏ sót, lỗi nào ngược lại, lỗi nào công cụ báo oan. Với mỗi lỗi công cụ bỏ sót, nêu dòng nào của ma trận tình huống đã phát hiện ra nó.

**Phản tư** (100 tới 200 từ). Việc đọc mã và việc chạy ma trận tình huống phát hiện được những loại lỗi khác nhau nào, đo bằng chính số lỗi mỗi cách tìm ra trong bài làm; vì sao một chương trình trả mã 0 lại là thứ khó tin hơn một chương trình dừng kèm thông báo truy vết lỗi; rủi ro khi một chương trình chạy theo lịch gọi tên sinh viên mà mọi lần chạy đều báo thành công, kèm trách nhiệm với các ô dữ liệu cá nhân đã đi vào file log.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất lượng kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi tiết ở tài liệu hướng dẫn chấm đi kèm sách.
