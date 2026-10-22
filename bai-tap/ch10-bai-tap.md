# Bài tập – Chương 10

Đề bài tập bốn tầng của chương. Đáp án và hướng dẫn chấm phát hành riêng cho giảng viên, không có trong kho này.

Tài liệu này đi kèm sách *Phân tích dữ liệu và triển khai mô hình AI với Python*. Mã nguồn chạy được và dữ liệu mẫu nằm ở [`du-an-ptdl/`](https://github.com/FITHOUVN/Data-Analysis-and-AI-by-Python/tree/main/du-an-ptdl).

---

Mọi bài làm trên dịch vụ bọc mô hình `models/ch08-mo-hinh-trien-khai.joblib` (10,0 KB) với ngưỡng 0,2727104755205902 đọc từ `models/ch08-ho-so-trien-khai.json`, trên lô 40 dòng `data/raw/ch09-lo-moi.csv`, nhãn thật ở `data/raw/ch09-lo-moi-nhan-that.csv`, và lô có lỗi `data/raw/ch09-lo-loi.csv`. Từ tầng hai trở lên nộp mã nguồn, dòng lệnh đã gõ, **mã trạng thái** của từng lời gọi và thân phản hồi.

Câu hỏi “thay đổi này có ảnh hưởng không” phải in đủ **ba con số**: số bản ghi đổi nhãn trên tổng số bản ghi, mức lệch xác suất lớn nhất, và số bản ghi cần liên hệ trước và sau. Chương này có nhiều ca dịch vụ trả mã 200 trong khi dữ liệu vào đã sai, nên bài nào kiểm dữ liệu vào còn phải nói rõ **lớp nào** phát hiện ca ấy trong ba lớp của Bảng 10.2: lớp khai báo, lớp kiểm miền nghiệp vụ, hay không lớp nào.

Mọi con số thời gian phụ thuộc máy. Bài nào đo thời gian phải nộp **ba lần đo** và dẫn kèm dải, nói rõ hệ điều hành và loại ổ đĩa; một giá trị đơn không được tính là kết quả. Tỉ số giữa hai cách làm cũng phải khai rõ **đo trên cái gì**: tỉ số giữa một lô và nhiều yêu cầu lẻ đổi theo số bản ghi của lô, nên một tỉ số không kèm số bản ghi là một con số không đọc được.

Bài nào dựng máy chủ thật phải **tự chọn một cổng còn trống** trên máy đang dùng rồi ghi số cổng ấy vào bài làm; cổng 8000 và 8765 trên máy soạn chương này đã bị dịch vụ khác chiếm. Bộ kiểm thử trong chương chạy bằng một vòng lặp tự viết chứ không bằng công cụ kiểm thử bên ngoài, nên các bài dưới đây cũng theo cách đó: không bài nào đòi cài thêm gói.

### Củng cố

**Bài 10.1** (★ · MTS6 · \~12 phút · Kiến thức: mục 10.1.1, 10.1.2, 10.1.3 · dạng: trắc nghiệm)

Chọn một phương án đúng cho mỗi câu.

(1) Một nhóm đo dịch vụ của mình, thấy các yêu cầu sau yêu cầu đầu đều mất 14 tới 33 ms ở cả hai cách đặt lệnh nạp mô hình, rồi kết luận nơi đặt lệnh nạp không quan trọng. Theo mục 10.1.1, cái mất thật khi nạp trong hàm xử lý yêu cầu là: A. tổng thời gian của một lô 40 bản ghi tăng 29,8 tới 50,3 lần; B. mã băm SHA-256 của file mô hình không còn được kiểm trước khi nạp; C. điểm cuối `/suc-khoe` trả `ok` sau 7 tới 10 ms trong khi yêu cầu dự đoán đầu tiên còn mất 1.312,6 tới 2.136,9 ms, nên bộ phân phối tải đọc `/suc-khoe` đẩy người dùng vào tiến trình chưa sẵn sàng; D. thời gian khởi động tăng từ 7 tới 10 ms lên 1.484 tới 2.289 ms.

(2) Một nhóm muốn biết dịch vụ có từ chối `NaN` hay không. Họ gọi `httpx.post(dia_chi, json={..., "diem_giua_ky": float("nan")})`, nhận `ValueError: Out of range float values are not JSON compliant` ngay tại máy gọi, rồi kết luận dịch vụ đã chặn `NaN`. Chỗ sai của kết luận: A. không có chỗ sai, vì ngoại lệ ấy là cách `httpx` báo lại mã 422 của dịch vụ; B. lời gọi chưa rời khỏi máy gọi, vì `httpx` dựng thân yêu cầu bằng `json.dumps` với `allow_nan=False` nên nó dừng trước khi gửi, mà `json.dumps` để mặc định thì ghi ra `NaN`, nên muốn thử phải dựng thân thô; C. phải đổi `float("nan")` thành chuỗi `"NaN"`, vì JSON chỉ nhận `NaN` dưới dạng chuỗi; D. `NaN` chỉ chặn được ở lớp kiểm miền nghiệp vụ, nên phép thử phải gửi qua `/du-doan-lo`.

(3) Bước 4 của Quy trình 10.1 không cho phép kết luận “dịch vụ đã chạy” từ việc `/openapi.json` trả mã 200. Lý do: A. khi cổng đã bị một dịch vụ khác chiếm, lời gọi nhận 200 từ dịch vụ kia, nên phải so trường `info.title` với tên đã khai trong `FastAPI(title=...)`; B. `/openapi.json` chỉ trả 200 sau khi hàm vòng đời chạy xong, nên mã 200 đến quá muộn để dùng làm dấu hiệu; C. `/openapi.json` sinh ra từ lần chạy nên nó không mô tả được các khai báo kiểu; D. mã 200 ở đây do `/docs` sinh ra, và `/docs` chỉ là một khung HTML 1.012 byte.

**Bài 10.2** (★ · MTS6, MTS7 · \~18 phút · Kiến thức: mục 10.2.2, 10.3.2, 10.3.3, 10.4.1 · dạng: đúng hoặc sai, có giải thích)

Cho biết mỗi phát biểu đúng hay sai, kèm một câu giải thích có số hoặc có một ca cụ thể. Phát biểu chỉ đúng trong một phạm vi thì nêu rõ điều kiện áp dụng còn thiếu.

(1) Lược đồ Pydantic chặt làm dịch vụ nghiêm khắc hơn chương trình dòng lệnh, nên mọi lô mà chương trình dòng lệnh xử lý được thì dịch vụ cũng xử lý được.

(2) Phép so `len(str(v)) >= 4` của Mã 10.6 đạt thì phản hồi 422 không mang dữ liệu cá nhân nào. Nêu luôn phép kiểm mà bước 6 Quy trình 10.2 chọn làm phép chính, và cho biết nó khác phép trên ở điểm nào.

(3) Mọi lỗi dữ liệu của người gọi đều nhận lại một mã 4xx.

(4) Đặt `timeout` nhỏ hơn làm lời gọi tới một cổng không có ai nghe trả về sớm hơn.

(5) Bước 2 Quy trình 10.3 đòi khóa giá trị thay vì khóa kết luận, nên một bộ kiểm thử đã khóa giá trị xác suất của bản ghi mốc sẽ phát hiện mọi thay đổi làm sai kết quả dịch vụ.

**Bài 10.3** (★ · MTS6 · \~18 phút · Kiến thức: mục 10.1.1, 10.2.4, 10.3.3, 10.4.1 · dạng: tính toán)

Dùng Bảng 10.1, Bảng 10.3, Bảng 10.6 và con số 416 byte cho một bản ghi.

(a) Tính tỉ số thời gian **yêu cầu đầu tiên** giữa hai cách đặt lệnh nạp, lấy cả cận dưới và cận trên. Rồi tính tỉ số **yêu cầu đầu trên yêu cầu sau** trong từng cách, và cho biết cách nào vi phạm phép tự kiểm ở bước 2 Quy trình 10.2.

(b) Dịch vụ chạy bốn tiến trình công nhân. Tính tổng thời gian mà bốn người dùng đầu tiên phải chờ ở mỗi cách nạp. Nêu việc phải làm sau khi khởi động để bốn khoản chờ ấy không rơi vào người dùng thật.

(c) Tính lại cột tỉ số của Bảng 10.3 từ hai cột thời gian in trong bảng. Một dòng cho kết quả khác con số in sẵn: nêu dòng đó, nêu độ lệch, và nêu nguyên nhân. Cho biết cách ghi bảng nào tránh được độ lệch ấy.

(d) Thân một bản ghi là 416 byte. Tính dung lượng thân yêu cầu cho 500 bản ghi và cho 50.000 bản ghi. Rồi tính trần số bản ghi lớn nhất giữ thân yêu cầu dưới 1 MiB (1.048.576 byte), tính cả phần bao JSON. Tính tỉ số giữa phản hồi 422 mặc định và phản hồi 422 sau khi lọc, theo hai con số của mục 10.3.3.

(e) Thời gian chờ mặc định của `httpx` là 5,0 giây, còn một lô 40 bản ghi mất 18,3 tới 82,5 ms qua máy chủ thật khi dịch vụ đã nóng và được gọi liên tục. Tính tỉ số giữa mức chờ 3,0 giây và thời gian trả lời lớn nhất ấy, rồi tính **tổng ba mức chờ và hai khoảng giãn cách** của hàm gọi có thử lại ở Mã 10.8 và nói rõ vì sao con số ấy **không** phải trần của tổng thời gian một lần chạy.

**Bài 10.4** (★ · MTS6 · \~15 phút · Kiến thức: mục 10.2.1, 10.2.2 · dạng: đọc bảng)

Dùng Bảng 10.2, mười bốn dòng.

(a) Chia mười bốn dòng thành ba nhóm theo lớp phát hiện. Rồi nêu hai dòng chứng minh rằng lớp khai báo không biết gì về miền nghiệp vụ, và hai dòng chứng minh rằng lớp kiểm miền nghiệp vụ cũng chưa đủ.

(b) Ba dòng cuối bảng có mã 200 kèm hậu quả đo được. Xếp ba dòng ấy theo mức nguy hiểm với phòng đào tạo, lấy căn cứ từ ba con số của từng dòng. Nêu lý do mức lệch xác suất lớn nhất một mình không đủ để xếp thứ tự.

(c) Dòng `tong_thoi_luong_phut` bằng 99999 có mã 422 mà vẫn có đủ ba con số hậu quả. Giải thích vì sao một dòng đã bị chặn lại cần in hậu quả, và cho biết ba con số ấy đo trên cấu hình dịch vụ nào.

(d) Hai dòng `nganh` lạ và `gioi_tinh` viết hoa hết đều cho 0 trên 40 nhãn đổi. Viết lại kết luận “mức phân loại lạ không gây hậu quả” kèm phạm vi áp dụng, rồi nêu đại lượng phải đo trên một lô mới để biết lô ấy có nằm trong phạm vi đó hay không.

**Bài 10.5** (★ · MTS6 · \~12 phút · Kiến thức: mục 10.1.3, 10.2.2, 10.3.2, 10.3.3, 10.4.1 · dạng: ghép đôi)

Ghép mỗi dấu hiệu với một nguyên nhân, rồi ghi thêm **mã trạng thái** mà người gọi nhận được trong từng ca, hoặc ghi “không có mã trạng thái” nếu lời gọi không tới được dịch vụ.

Dấu hiệu: (1) `/openapi.json` trả 200 nhưng `info.title` là tên của một dịch vụ khác; (2) `/docs` hiện một trang trắng trong khi `/openapi.json` trả về đúng 3 đường dẫn; (3) một phép kiểm thử nhận `trang_thai` bằng `chua-nap`; (4) người gọi gửi mã sinh viên nhưng trường `ma_sv` trong phản hồi bằng `null`, và không có dòng lỗi nào; (5) chương trình gọi dừng với `ValueError: Out of range float values are not JSON compliant`; (6) một yêu cầu vừa thiếu một trường vừa có `NaN` ở trường khác làm người gọi nhận một lỗi phía dịch vụ; (7) lời gọi về sau 2.193 tới 2.458 ms kèm `ConnectError`; (8) lời gọi trả về kèm thân `{"detail":"Method Not Allowed"}`; (9) câu lệnh `import 75_api_du_doan` cho `SyntaxError: invalid decimal literal`.

Nguyên nhân: A. cổng đã bị một dịch vụ khác chiếm; B. máy không ra được Internet nên khung HTML 1.012 byte của `/docs` không nạp được thư viện trình bày; C. khách kiểm thử dựng ngoài câu lệnh `with` nên hàm vòng đời không chạy; D. tên trường gõ sai trong khi lược đồ để mặc định `extra="ignore"` và trường ấy có giá trị mặc định; E. `httpx` dựng thân yêu cầu với `allow_nan=False`; F. `JSONResponse` của bộ xử lý 422 mặc định ghi JSON với `allow_nan=False` trên khóa `input`; G. không có tiến trình nào nghe ở cổng, và hệ điều hành thử lại việc nối trước khi báo lỗi; H. đường dẫn đúng nhưng phương thức sai; I. tên file mã bắt đầu bằng chữ số.

**Bài 10.6** (★ · MTS6 · \~15 phút · Kiến thức: mục 10.3.1, 10.3.2, 10.3.3 · dạng: đọc bảng)

Dùng Bảng 10.4 và Bảng 10.5.

(a) Dòng đầu Bảng 10.4 cho 4 phép trượt trên 15, tám dòng còn lại chỉ cho 1. Giải thích con số 4 bằng hai việc mà bộ xử lý 422 của Mã 10.7 cùng gánh.

(b) Tám lần phá mã đều làm trượt đúng một phép kiểm thử. Cho biết điều đó khớp với bước nào của Quy trình 10.3, rồi nêu một rủi ro đi kèm khi mỗi lớp chỉ có một phép kiểm thử phủ.

(c) Nêu một chỗ phá mã thứ mười mà cả mười lăm phép kiểm thử đều đạt. Chỗ ấy phải đổi được một giá trị mà người gọi đọc trong phản hồi, và phải nêu kèm phép kiểm thử bắt được nó.

(d) Hai dòng đầu Bảng 10.5 cho mã 404 và 405. Với mỗi mã, cho biết người gọi phải sửa gì, và vì sao tài liệu tự sinh của điểm cuối `/du-doan` không khai hai mã ấy dù dịch vụ vẫn trả chúng.

### Vận dụng

**Bài 10.7** (★★ · MTS6 · \~60 phút · Kiến thức: mục 10.1.3, 10.2.2, 10.2.3, 10.2.4 · dạng: viết mã) **\[HTTT\]**

Dựng dịch vụ dự đoán cho phòng đào tạo, đi trọn Quy trình 10.1 rồi trọn Quy trình 10.2.

(a) Đi năm bước Quy trình 10.1 trên một ứng dụng tối thiểu có ba điểm cuối tự chọn. Nộp số cổng đã dùng, trường `info.title` đọc được từ `/openapi.json`, và bằng chứng của phép tự kiểm bước 4: nếu cổng thử đầu tiên trả về một `info.title` khác thì ghi lại cả tên ấy và số cổng đã đổi sang.

(b) Đi bảy bước Quy trình 10.2 để bọc mô hình thành ba điểm cuối `GET /suc-khoe`, `POST /du-doan` và `POST /du-doan-lo`. Mỗi bước nộp kết quả của phép tự kiểm của chính bước ấy.

(c) Bước 3 đòi so **từng tên trường theo thứ tự** với khóa `cot_vao_theo_thu_tu`. Dựng một bản lược đồ gõ sai tên đúng một trường, rồi nộp hai con số: số trường mà `/openapi.json` báo cho bản gõ sai, và số phần tử lệch khi so từng tên theo thứ tự. Từ hai con số ấy nói rõ vì sao phép đếm không thay được phép so.

(d) Bước 6 đòi lọc khóa `input`. Đo và nộp dung lượng thân phản hồi 422 của hai bản, bản dùng bộ xử lý mặc định và bản dùng bộ xử lý riêng, trên cùng một yêu cầu thiếu trường `diem_giua_ky`; kèm số khóa và số giá trị của bản ghi còn sót trong mỗi thân.

Tự kiểm: (b) `/suc-khoe` báo ngưỡng 0,2727, `so_cot_vao` bằng 14 và `toi_da_mot_lo` bằng 500; lô 40 bản ghi cho 7 bản ghi cần liên hệ; bản ghi `SV2601614` cho xác suất 0,331912; lô 501 bản ghi và lô rỗng đều cho 422; (c) 15 trường và 1 phần tử lệch; kích thước phản hồi là số đo tham chiếu của đúng phiên bản và cách mã hóa đã nêu; (d) 445 byte với 4 khóa và 14 giá trị, so với 80 byte với 3 khóa và 0 giá trị.

**Bài 10.8** (★★ · MTS6 · \~50 phút · Kiến thức: mục 10.1.2, 10.2.2, 10.3.1 · dạng: viết mã)

Đo xem mỗi cách khai một trường số thực chặn thêm được những gì, bằng ba bản lược đồ cho cùng một điểm cuối. Bản L khai mọi trường số thực bằng `float` trần; bản C thêm `allow_inf_nan=False`; bản M thêm `Field(ge=..., le=...)` cho ba trường có miền đóng. Cả ba bản giữ nguyên lớp kiểm miền nghiệp vụ của Chương 9 và bước 4 Quy trình 10.2.

(a) Gửi cho bản L một thân yêu cầu thô có `NaN` ở `ty_le_hoan_thanh_video`. Nộp mã trạng thái, giá trị của trường ấy trong phản hồi, và nội dung trường `ngoai_khoang_huan_luyen`. Rồi nộp ba con số của ca `ty_le_hoan_thanh_video` bằng `NaN` trên cả lô 40 bản ghi.

(b) Lặp lại phép thử ở (a) với bản C. Nộp mã trạng thái và khóa `loai` của phần tử `detail`. Nêu đường đi mà `NaN` phải qua để ra được thành `null` trong phản hồi của bản L, và nói rõ chỗ nào trong đường đi ấy thay giá trị thiếu bằng một số cụ thể.

(c) Gửi cho bản C và bản M cùng một bản ghi có `ty_le_hoan_thanh_video` bằng 3,5. Nộp mã trạng thái, khóa `loai` và thông điệp của mỗi bản, rồi nộp đoạn mô tả trường ấy trong `/openapi.json` của hai bản. Cho biết bản M giành được gì và không giành được gì so với bản C.

(d) Gửi cho cả ba bản một bản ghi có `tong_thoi_luong_phut` bằng 99999. Nộp mã trạng thái của ba bản khi chưa khai cận trên nghiệp vụ, rồi nộp lại sau khi khai cận trên 50.400 phút. Kèm ba con số của ca ấy trên lô 40 bản ghi.

(e) Phép tự kiểm ở bước 4 Quy trình 10.2 nêu hai hướng hỏng. Dựng một bản lược đồ và một bộ ca thử mà phép tự kiểm ấy đạt trong khi lớp kiểm miền nghiệp vụ không chạy một dòng nào, rồi viết lại phép tự kiểm cho bắt được bản vừa dựng.

Tự kiểm: (a) mã 200, trường trong phản hồi bằng `null`, `ngoai_khoang_huan_luyen` chứa đúng `ty_le_hoan_thanh_video`, và ba con số là 2 trên 40 nhãn, lệch lớn nhất 0,4138, số cần liên hệ 7 còn 5; (b) mã 422 với `loai` bằng `finite_number`; (c) bản C báo lỗi từ lớp nghiệp vụ, bản M báo `less_than_equal`, và chỉ bản M có `minimum` cùng `maximum` trong `/openapi.json`; (d) cả ba bản cho 200 trước khi khai cận trên, cả ba cho 422 sau khi khai, ba con số là 7 trên 40, lệch 0,9587, 7 còn 0.

**Bài 10.9** (★★ · MTS6 · \~55 phút · Kiến thức: mục 10.3.2 · dạng: viết mã)

Viết bộ kiểm thử theo Quy trình 10.3, rồi chứng minh nó có tác dụng bằng năm chỗ phá mã **khác** chín chỗ của Bảng 10.4.

(a) Đi sáu bước Quy trình 10.3 và viết ít nhất tám phép kiểm thử, trong đó mỗi lớp kiểm của Bảng 10.2 có đúng một phép phủ. Nộp bảng ghi phép kiểm thử nào phủ lớp nào, theo bước 3.

(b) Phá năm chỗ sau, mỗi lần đúng một chỗ rồi phục hồi file: bỏ `min_length=1` khỏi lớp khai báo lô; đổi `round(float(x), 6)` thành `round(float(x), 3)`; đổi `x >= nguong` thành `x > nguong` khi tính nhãn; thay lời đọc ngưỡng từ hồ sơ bằng hằng số 0,2727 viết trong mã dịch vụ; đổi hai dấu `<=` thành `<` khi so giá trị với khoảng quan sát lúc huấn luyện. Lập bảng năm dòng: chỗ bị phá, số phép trượt, tên phép trượt.

(c) Ba trong năm chỗ không làm phép nào trượt. Với từng chỗ, viết một phép kiểm thử bắt được nó và nộp con số mà phép kiểm thử ấy so sánh. Chỗ nào không bắt được bằng một lời gọi qua lược đồ thì nói rõ phải tách phần nào của mã ra để kiểm riêng.

(d) Bước 2 đòi chọn bản ghi mốc là bản ghi gần ngưỡng nhất của lô. Tính khoảng cách tới ngưỡng của cả 40 bản ghi, nộp bản ghi gần nhất và bản ghi xa nhất. Rồi chứng minh bằng số rằng một phép kiểm thử so nhãn trên bản ghi xa nhất vẫn đạt khi xác suất đã lệch rất nhiều.

(e) Bước 7 Quy trình 10.2 nêu hướng hỏng “phép kiểm đạt mà không thử gì”. Viết một phép kiểm thử trần lô mắc đúng lỗi ấy, chạy nó trên bản dịch vụ đã bỏ trần lô, và nộp kết quả. Rồi sửa phép kiểm thử theo đúng lời bước 7 và chạy lại.

Tự kiểm: (b) hai chỗ đầu làm trượt đúng một phép mỗi chỗ, ba chỗ sau không làm trượt phép nào; (c) phép kiểm thử cho chỗ đổi ngưỡng thành hằng số phải so số bản ghi cần liên hệ sau khi đổi ngưỡng trong một bản sao của hồ sơ, đo được 5 trên 40 với ngưỡng 0,40; phép kiểm thử cho chỗ đổi `<=` thành `<` phải gửi cả lô 40 bản ghi và đếm số bản ghi bị đánh dấu ngoại suy, đo được 0 trên 40 ở bản đúng và 18 trên 40 ở bản bị phá; (d) bản ghi gần nhất cách ngưỡng 0,059202 khi tính với ngưỡng đủ chữ số, và 0,059212 khi tính với ngưỡng đã làm tròn 0,2727 như mục 10.2.1 in ra; cả hai giá trị đều được tính là đúng nếu bài làm nói rõ đã dùng ngưỡng nào.

**Bài 10.10** (★★ · MTS6 · \~45 phút · Kiến thức: mục 10.1.3, 10.4.1 · dạng: viết mã)

Gọi dịch vụ từ một chương trình khác, qua một máy chủ thật, rồi đo bốn tình huống lỗi. Dùng bước 4 và bước 5 Quy trình 10.1 để dựng và kiểm máy chủ trước khi đo.

(a) Khởi động dịch vụ trên một cổng còn trống, nộp số cổng và kết quả phép tự kiểm bước 4. Nêu dấu hiệu phân biệt “cổng bị chiếm” với “dịch vụ của mình đã chạy”, và nói rõ vì sao chỉ xem mã trạng thái là không đủ.

(b) Lập lại Bảng 10.6 trên máy của mình, năm dòng, mỗi dòng ba lần đo, nộp dải. Dòng không truyền `timeout` thì nộp giá trị mặc định đọc được từ `httpx.Client().timeout`. Một dòng của Bảng 10.6 có thể cho mã 200 thay cho một ngoại lệ chờ quá hạn: nêu dòng đó, nêu điều kiện áp dụng còn thiếu của nó, rồi dựng một lời gọi tái hiện được ngoại lệ ấy và nộp cả thời gian chờ đã đặt lẫn thời gian thật đo được.

(c) Đo số lần gọi thật trong bốn tình huống: dịch vụ không chạy; dịch vụ trả 422 vì lô thiếu một trường; đường dẫn sai nên dịch vụ trả 404; dịch vụ trả một mã 5xx, dựng bằng một điểm cuối tạm nêu một ngoại lệ. Lập bảng bốn dòng: số lần gọi, tổng thời gian, mã trả về của chương trình gọi. Giải thích vì sao hai dòng giữa chỉ gọi một lần, bằng nhánh chia 4xx và 5xx ở Mã 10.8.

(d) Chọn một giá trị `timeout` cho chương trình gọi, dựa trên số đo của chính mình ở (b) chứ không dựa trên con số in trong chương. Nộp phép tính và **tổng ba mức chờ cộng hai khoảng giãn cách**, kèm một câu nói rõ con số ấy không phải trần của tổng thời gian một lần chạy và nêu lý do.

(e) Thử lại một yêu cầu `POST` cần bảo đảm không lặp tác động nghiệp vụ ngoài ý muốn, bằng tính idempotent hoặc cơ chế chống thực hiện lặp. Thêm vào dịch vụ một việc làm mất điều kiện ấy, rồi nộp một phép kiểm phát hiện được tình trạng mất điều kiện; phép kiểm không được dựa vào việc đọc mã nguồn của dịch vụ.

Tự kiểm: (b) thời gian chờ mặc định là `Timeout(timeout=5.0)`; cổng không có ai nghe cho `ConnectError` sau 2.193 tới 2.458 ms với mức chờ mặc định, và `ConnectTimeout` sau 1.153 tới 1.433 ms với `timeout=1.0`; dòng không tái lập được là dòng `timeout=0.05`; (c) một lần gọi cho hai ca 422 và 404, ba lần gọi cho hai ca còn lại, mã trả về 7 cho hai ca giữa và 6 cho hai ca còn lại.

### Thực tế và liên ngành

**Bài 10.11** (★★★ · MTS6, MTS7 · \~45 phút · Kiến thức: mục 10.2.2, 10.3.3, 10.4.2 · dạng: bảo vệ phương án, Kiến trúc)

Công ty Kiến trúc Nam Phong dùng một mô hình xếp thứ tự hồ sơ thiết kế theo mức nguy cơ phải làm lại, rồi hiện mức ấy trên màn hình nhận hồ sơ. Màn hình do một nhóm khác viết và gọi dịch vụ qua mạng nội bộ. Bộ phận hợp đồng đặt hai yêu cầu: phản hồi lỗi không được mang lại nội dung hồ sơ mà khách gửi lên, và mọi lần từ chối phải truy được về nguyên nhân. Phần đo được của bài làm thực hiện trên dịch vụ và lô 40 bản ghi của chương.

(a) Lập bảng hai bộ xử lý lỗi 422, mặc định và riêng, với bốn cột: dung lượng thân phản hồi, số khóa của một phần tử `detail`, số giá trị của bản ghi còn trong thân, và kết luận đạt hay không đạt yêu cầu của bộ phận hợp đồng.

(b) Dòng `len(str(v)) >= 4` của Mã 10.6 chỉ xét các giá trị từ 4 ký tự trở lên. Dựng một bản ghi mà phép so ấy đạt trong khi thân phản hồi vẫn mang một giá trị của hồ sơ, rồi viết lại nó cho bắt được cả ca vừa dựng. Phép kiểm mới phải chạy được trong một dòng lệnh. Bước 6 Quy trình 10.2 đã chọn một dạng khác làm phép chính; nêu dạng ấy và cho biết vì sao nó không hở theo cùng kiểu.

(c) Dựng một yêu cầu vừa thiếu một trường vừa có `NaN` ở trường khác, gửi cho bản dùng bộ xử lý mặc định, rồi nộp mã trạng thái. Cho biết vì sao công ty không được kết luận “dịch vụ hỏng” từ mã ấy, và nêu chỗ phải sửa.

(d) Chuẩn bị bảo vệ phương án 5 phút. Nhóm viết màn hình đề nghị: “mọi dữ liệu sai đều trả 200 kèm một trường cảnh báo, để màn hình không phải xử lý nhánh lỗi”. Trả lời bằng ba con số của một dòng Bảng 10.2 và bằng một hậu quả với người bị mô hình đánh giá.

**Bài 10.12** (★★★ · MTS6 · \~50 phút · Kiến thức: mục 10.1.1, 10.2.4, 10.4.1, 10.4.2 · dạng: tình huống nghiệp vụ và vấn đáp, Thiết kế công nghiệp)

Công ty Thiết kế Công nghiệp Tân Hải dự đoán nguy cơ lỗi khuôn cho từng bản thiết kế. Một màn hình đặt tại xưởng gọi dịch vụ từng bản một trong lúc người đứng máy chờ; mỗi đêm một lịch chạy gửi cả 2.400 bản thiết kế của ngày hôm đó. Hồ sơ một bản thiết kế có 30 trường, thân JSON đo được 912 byte. Dịch vụ chạy bốn tiến trình công nhân và được khởi động lại mỗi đêm.

(a) Tính tổng thời gian mà bốn người đứng máy đầu tiên phải chờ sau mỗi lần khởi động lại, ở cả hai cách đặt lệnh nạp mô hình, dùng dải của Bảng 10.1. Nêu lý do `/suc-khoe` trả `ok` không đủ để kết luận dịch vụ đã sẵn sàng, rồi đề nghị một phép kiểm sẵn sàng thay thế và nói rõ nó đo gì.

(b) Chọn trần số bản ghi mỗi lô cho hồ sơ 912 byte, với điều kiện thân yêu cầu không vượt 1 MiB (1.048.576 byte), tính cả phần bao JSON. Nộp phép tính, trần đã chọn, và số khối cho 2.400 bản thiết kế. Cho biết con số nào trong phản hồi `/suc-khoe` phải mang trần ấy, và điều gì xảy ra nếu lịch chạy giữ trần trong một hằng số viết trong chương trình gọi.

(c) Màn hình xưởng chỉ có 2 giây trước khi người đứng máy coi là treo. Chọn thời gian chờ, số lần thử lại và giãn cách, rồi nộp **tổng các mức chờ và các khoảng giãn cách**; đối chiếu với 2 giây, cho biết con số ấy có **bảo đảm** được ràng buộc 2 giây hay không, và nêu chỗ phải nhượng bộ. Căn cứ phải là một dải đo được, không phải một con số chọn sẵn.

(d) Chuẩn bị trả lời vấn đáp 3 phút. Người phụ trách hệ thống đề nghị nạp mô hình trong hàm xử lý yêu cầu, vì “dịch vụ khởi động trong 10 ms nên triển khai lại nhanh hơn nhiều”. Trả lời bằng hai hậu quả đo được, mỗi hậu quả kèm một dải của Bảng 10.1, và nêu một tình huống vận hành ở xưởng mà hậu quả ấy xuất hiện mỗi đêm.

**Bài 10.13** (★★★ · MTS6, MTS7 · \~180 phút cho cả nhóm · Kiến thức: mục 10.2, 10.3, 10.4 · dạng: làm nhóm ba người, có trình bày, Ngôn ngữ Anh)

Trung tâm Ngoại ngữ Hướng Dương dạy tiếng Anh trực tuyến và muốn hiện mức nguy cơ bỏ dở khóa học ngay trên màn hình của giáo viên phụ trách, từng học viên một. Hệ thống quản lý học tập do một nhà cung cấp khác viết và không cài được thư viện học máy. Nhóm ba người dựng trọn dịch vụ, dùng mô hình và lô 40 bản ghi của chương làm vật liệu.

(a) Người thứ nhất đi Quy trình 10.1 rồi Quy trình 10.2, nộp ba điểm cuối, bảy phép tự kiểm của Quy trình 10.2, và bảng mười bốn ca của Bảng 10.2 đo lại trên dịch vụ của nhóm.

(b) Người thứ hai đi Quy trình 10.3, nộp bộ kiểm thử ít nhất mười phép, bảng phủ lớp theo bước 3, và bảng năm chỗ phá mã kèm phép trượt theo bước 5. Trong năm chỗ phải có ít nhất một chỗ mà không phép nào trượt, kèm phép kiểm thử bổ sung bắt được nó.

(c) Người thứ ba viết chương trình gọi theo Mã 10.8, nộp bảng thời gian chờ năm dòng theo Bảng 10.6, bảng ba mã trả về của chương trình gọi, và một bảng ghi rõ log của dịch vụ ghi những gì; log không được mang giá trị nào của học viên.

(d) Nộp chung: mã nguồn chạy lại được, phản hồi `/suc-khoe`, bộ kiểm thử chạy ra số đạt và số trượt, file log của một lần chạy, và bài trình bày 7 phút trả lời một câu hỏi: màn hình giáo viên nên gọi từng học viên một, hay nên gọi cả lớp 300 học viên một lần rồi lưu kết quả lại? Căn cứ phải là dải của Bảng 10.3 và thời gian của một lời gọi lẻ, cùng hệ quả khi kết quả đã lưu trở nên cũ.

(e) Mỗi người nộp thêm một đoạn 100 từ: phần của hai người kia có chỗ nào mà phần của mình phụ thuộc vào, và chỗ ấy đã được kiểm bằng gì.

### Thử thách có AI

**Bài 10.14** (★★★★ · MTS6, MTS7 · \~110 phút · Kiến thức: toàn chương · dạng: dự đoán phép kiểm thử nào trượt, chạy bộ kiểm thử lên bản AI, rồi tìm các lỗi không phép nào bắt)

**Bài toán.** Trung tâm Ngoại ngữ Hướng Dương nhờ một trợ lý AI tạo sinh bọc mô hình thành dịch vụ và nhận về đoạn mã dưới đây, kèm bốn câu báo cáo: “dịch vụ đã chạy và trả 200 cho cả lô 40 bản ghi”; “mô hình được nạp một lần nên `/suc-khoe` trả `ok` ngay sau khi tiến trình lên”; “lược đồ Pydantic chặn mọi dữ liệu vào không hợp lệ trước khi tới mô hình”; “đã chạy bộ kiểm thử của mục 10.3.2 và không phép nào trượt”. Bản nhận được **không** có hàm nào trang trí bằng `@app.exception_handler`.

>     cli = nap_mo_dun(FILE_CLI, "du_doan_cli")
>     TEN_PHIEN_BAN = "%s@%s" % (cli.FILE_MO_HINH.name,
>                                cli.BAM_MONG_DOI[:8])
>     _goi: dict = {}
>     _goi.update(cli.nap_mo_hinh())
>     app = FastAPI(title="API dự đoán nguy cơ không đạt", version="1.0")
>
>
>     class HoSoSinhVien(BaseModel):
>         ma_sv: str | None = None
>         nganh: str
>         so_lan_dang_nhap: int
>         tong_thoi_luong_phut: int
>         ty_le_hoan_thanh_video: float
>         diem_giua_ky: float = Field(allow_inf_nan=False)
>
>
>     class YeuCauLo(BaseModel):
>         ho_so: list[HoSoSinhVien] = Field(min_length=1)
>
>
>     def du_doan_api(ho_so: list[HoSoSinhVien]) -> list[KetQua]:
>         bang = pd.DataFrame([h.model_dump() for h in ho_so])
>         loi, canh_bao = cli.kiem_hop_dong(bang, _goi)
>         chi_tiet = [{"ban_ghi": None if dong == 0 else dong - 2,
>                      "truong": cot, "ly_do": ly_do}
>                     for dong, cot, _, ly_do in loi]
>         chi_tiet += kiem_can_tren(bang)
>         if chi_tiet:
>             raise HTTPException(status_code=422, detail=chi_tiet)
>         cot = _goi["ho_so"]["cot_vao_theo_thu_tu"]
>         nguong = _goi["ho_so"]["nguong_quyet_dinh"]
>         p = _goi["mo_hinh"].predict_proba(bang[cot])[:, 1]
>         return [KetQua(ma_sv=h.ma_sv, xac_suat=round(float(x), 6),
>                        can_lien_he=bool(x >= nguong),
>                        muc_uu_tien=("Liên hệ ngay" if x >= 0.66 else
>                                     "Cần xem lại" if x >= 0.33 else
>                                     "Chưa cần"),
>                        ngoai_khoang_huan_luyen=ngoai_khoang(h),
>                        nguong=round(nguong, 2), mo_hinh=TEN_PHIEN_BAN)
>                 for h, x in zip(ho_so, p)]

Các phần còn lại của bản nhận được giữ nguyên như Mã 10.3 tới Mã 10.5; riêng **chín** trường còn lại của hợp đồng thì khai **theo đúng mẫu lỏng của sáu trường trên**, tức `str` và `int` trần, còn `diem_tb_bai_tap` và `nop_tre_tb_gio` khai `float` trần không kèm `allow_inf_nan`. Hai hàm `kiem_can_tren` và `ngoai_khoang` giữ nguyên, lớp `KetQua` khai bảy trường, ba điểm cuối khai như Mã 10.5.

**Nhiệm vụ với AI.** Bốn bước, theo đúng thứ tự. Bước một, **chưa** mở công cụ AI: đọc đoạn mã và bốn câu báo cáo, rồi viết ra **danh sách dự đoán** gồm tên từng phép kiểm thử của mục 10.3.2 kèm chữ “đạt” hoặc “trượt”, và với mỗi phép dự đoán trượt thì ghi dòng mã chịu trách nhiệm. Niêm phong danh sách ấy. Bước hai: chạy mười hai phép kiểm thử API gốc lên bản nhận được; trong bài này tách riêng ba phép mới có tên kiem_thu_cli_lo_rong, kiem_thu_cli_so_huu_han_va_so_nguyen và kiem_thu_toan_ven_mo_hinh_va_ho_so, nộp số đạt và số trượt, rồi so với bản niêm phong. Bước ba: đưa đoạn mã và bốn câu báo cáo cho một công cụ AI tạo sinh, yêu cầu nó liệt kê mọi lỗi; **không** dán danh sách dự đoán và không dán kết quả chạy vào prompt. Bước bốn: tìm các lỗi mà không phép kiểm thử nào trượt, viết phép kiểm thử bắt từng lỗi, rồi chạy lại cả bộ.

**Nhật ký prompt.** Nộp toàn văn các prompt, tên và phiên bản công cụ, ngày dùng, trích đoạn phản hồi ứng với từng lỗi, cùng bản niêm phong của bước một. Chỉ dán tên trường, kiểu trường và lược đồ; không dán dữ liệu của học viên.

Kiểm chứng. Bản nhận được có nhiều chỗ sai. Dùng bộ 12 phép gốc để phân loại lỗi đã bị bắt, lỗi làm nhiều phép cùng trượt và lỗi còn bỏ sót. Số phép trượt không bằng số lỗi. Trả lời sáu câu dưới đây bằng kết quả chạy của chính bài làm.

1.  Nộp số đạt và số trượt của bản nhận được, kèm tên từng phép trượt. Ghép mỗi phép trượt với chỗ sai sinh ra nó. Có một chỗ sai làm trượt ba phép cùng lúc: nêu chỗ đó và giải thích bằng hai việc mà bộ xử lý 422 riêng cùng gánh.
2.  Một phép kiểm thử trượt kèm `ValueError: Out of range float values are not JSON compliant` chứ không kèm một phép so sai. Giải thích đường đi sinh ra ngoại lệ ấy, cho biết người gọi nhận mã nào trên một máy chủ thật, rồi nối với bước 6 Quy trình 10.3.
3.  Dựng khách kiểm thử **không** dùng câu lệnh `with` rồi gọi `/suc-khoe` trên bản nhận được và trên bản của Mã 10.5. Nộp hai giá trị `trang_thai`, rồi cho biết phép tự kiểm ở bước 1 Quy trình 10.3 kết luận gì trên từng bản và vì sao một trong hai kết luận là kết luận rỗng.
4.  Gửi một thân thô có NaN ở ty_le_hoan_thanh_video cho bản nhận được. Nộp mã trạng thái, tập khóa phản hồi và trường ngoai_khoang_huan_luyen; xác nhận phản hồi có trả lại trường đầu vào ấy hay không. Đo ba con số của ca ấy trên lô 40 bản ghi và giải thích vì sao phép kiểm thử NaN của mục 10.3.2 không bắt được lỗi này.
5.  So trường `nguong` trong phản hồi với khóa `nguong_quyet_dinh` của hồ sơ, rồi đếm số bản ghi của lô 40 mà người gọi suy lại nhãn từ `xac_suat` và `nguong` trong cùng phản hồi sẽ suy sai. Từ con số ấy nói rõ vì sao phép tự kiểm ở bước 5 Quy trình 10.2 đạt trên lô này, và nêu điều kiện áp dụng còn thiếu của nó.
6.  So trường `muc_uu_tien` của bản nhận được với ba mức do hai điểm cắt của Chương 9 sinh ra, trên cả 40 bản ghi. Nộp số bản ghi lệch và mã của từng bản ghi lệch. Rồi dựng một lô mà cùng chỗ sai ấy cho một bản ghi có `can_lien_he` bằng `true` trong khi `muc_uu_tien` bằng `Chưa cần`, và nêu hậu quả với giáo viên đọc danh sách.

Cuối cùng so ba danh sách: chỗ sai nào tự dự đoán được ở bước một, chỗ nào chỉ lộ ra khi chạy, chỗ nào chỉ công cụ AI nêu, và chỗ nào công cụ báo oan. Với mỗi chỗ sai mà không phép kiểm thử nào bắt, nộp phép kiểm thử mới và con số nó so sánh.

**Phản tư** (100 tới 200 từ). Việc đọc mã, việc chạy bộ kiểm thử và việc hỏi công cụ AI phát hiện được những loại chỗ sai khác nhau nào, đo bằng chính số chỗ sai mỗi cách tìm ra trong bài làm; vì sao câu báo cáo “bộ kiểm thử không phép nào trượt” là câu khó phản bác nhất trong bốn câu; và rủi ro khi một dịch vụ trả về mức ưu tiên sai cho một học viên, kèm trách nhiệm với các giá trị của hồ sơ đi ra ngoài trong phản hồi 422 mặc định.

**Rubric.** Bốn tiêu chí, mỗi tiêu chí 25%, theo thang năm mức của trường: Đúng đắn kỹ thuật; Chất lượng kiểm chứng phần công cụ AI tạo ra; Vận dụng kiến thức của chương; Phản tư và đạo đức. Bảng chi tiết ở tài liệu hướng dẫn chấm đi kèm sách.
