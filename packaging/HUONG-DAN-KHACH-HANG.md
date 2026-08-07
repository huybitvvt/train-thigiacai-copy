# Trạm cân QR Việt Nhật IPT — hướng dẫn cài đặt

Phiên bản: `0.2.0-rc8` — bản chạy thử nghiệm thu tại xưởng.

## 1. Cài đặt

1. Đóng bản đang chạy, sau đó chạy file `TramCanQR-Setup-0.2.0-rc8.exe`
   trên Windows 10/11 64-bit. Có thể cài đè bản cũ; `config.env` và dữ liệu
   trong `%LOCALAPPDATA%\TramCanQR` được giữ nguyên.
2. Nếu Windows SmartScreen cảnh báo, đối chiếu SHA-256 với file
   `SHA256SUMS.txt` do đơn vị triển khai gửi trước khi tiếp tục.
3. Mở **Trạm cân QR** từ Desktop. Trình duyệt sẽ mở địa chỉ
   `http://127.0.0.1:8080`.
4. Cho phép Chrome/Edge sử dụng camera khi trình duyệt hỏi.

Không cần cài Python. Không đổi tên, di chuyển hoặc xóa thư mục `_internal`
trong bản portable.

## 2. Chọn số camera/trạm

Bản cài mặc định hiển thị 3 trạm. Muốn đổi cấu hình, sao chép
`customer-config.env.example` trong thư mục cài đặt thành:

```text
%LOCALAPPDATA%\TramCanQR\config.env
```

Mở `config.env` bằng Notepad và đặt `ROLL_SCALE_STATION_COUNT` thành `1`, `2`
hoặc `3`. Danh sách `ROLL_SCALE_STATION_IDS` và `ROLL_SCALE_CAMERA_IDS` phải có
đúng số phần tử tương ứng. Đóng ứng dụng rồi mở lại sau khi sửa.

Mỗi camera chỉ được gán cho một trạm. Trên giao diện, chọn đúng camera ở từng
thẻ trạm rồi bấm **Mở camera đã gán**. Trình duyệt lưu ánh xạ này trên chính
máy khách.

Với `local`/`hybrid`, mặc định mỗi lần chụp dùng 5 frame và PaddleOCR chọn 3
frame đầu–giữa–cuối. Với `gemini`, mỗi lần nhấn `Space` chỉ chụp và gửi đúng
một ảnh đầy đủ. Có thể giữ cấu hình:

```text
ROLL_SCALE_WEIGHT_BURST_FRAMES=5
```

Gemini mặc định tắt. Chỉ đơn vị triển khai được bật sau khi khách hàng chấp
thuận việc gửi ảnh camera tới Google và cấp key bằng kênh riêng:

```text
ROLL_SCALE_GEMINI_ENABLED=true
ROLL_SCALE_WEIGHT_ENGINE=hybrid
ROLL_SCALE_GEMINI_API_KEY=replace-with-google-ai-studio-key
ROLL_SCALE_GEMINI_MODEL=gemini-3.5-flash-lite
ROLL_SCALE_GEMINI_ACCURATE_MODEL=gemini-3.1-pro-preview
ROLL_SCALE_GEMINI_TIMEOUT=10.0
ROLL_SCALE_GEMINI_ACCURATE_TIMEOUT=30.0
```

Gemini chỉ xác nhận ứng viên đa số từ PaddleOCR; kết quả cloud đơn lẻ hoặc
khác local luôn bị giữ lại để người vận hành kiểm tra.

Trên giao diện, chọn **Nhanh** để dùng `gemini-3.5-flash-lite` với thinking
tối thiểu; chọn **Chính xác** để dùng `gemini-3.1-pro-preview` với thinking
trung bình. Model Pro không có Free Tier trên Gemini API và cần bật billing.
Mỗi lần nhấn `Space` ở cả hai chế độ vẫn chỉ gửi đúng một ảnh. Timeout mặc định
lần lượt là 10 giây và 20 giây.

Bản pilot dùng đúng một camera nhưng để Gemini đọc trực tiếp thì đổi
`ROLL_SCALE_WEIGHT_ENGINE=gemini`. Paddle không khởi tạo. Mỗi lần nhấn `Space`
hoặc chọn tệp, đúng một ảnh đầy đủ được gửi để Gemini đọc cả QR và số cân. QR
local vẫn được đọc độc lập. Nếu hai nguồn xung đột nhưng QR local đã được bộ
giải mã QR chuyên dụng xác nhận hợp lệ, hệ thống giữ nguyên nội dung QR local;
Gemini chỉ bổ sung QR khi bộ giải mã local không tìm thấy mã.

Với chế độ `local` hoặc `hybrid`, sau khi camera đã được bắt cố định, đơn vị
triển khai phải hiệu chỉnh ROI hàng số gross cho từng trạm và điền vào
`config.env`. Chế độ `gemini` toàn ảnh bỏ qua ROI. Mỗi ROI có dạng
`x1,y1,x2,y2` từ 0 đến 1; các trạm ngăn cách bằng dấu chấm phẩy:

```text
ROLL_SCALE_WEIGHT_ROIS=0.4500,0.8000,0.5400,0.8500;0.4450,0.7950,0.5350,0.8450;0.4550,0.8050,0.5450,0.8550
```

Các số trên chỉ minh họa, không sao chép sang xưởng. ROI phải ôm sát duy nhất
hàng gross, không chứa bàn phím hoặc hai hàng `0.00` phía dưới. Số ROI phải
bằng `ROLL_SCALE_STATION_COUNT` và đúng thứ tự `ROLL_SCALE_STATION_IDS`.

## 3. Vận hành

- Phím `1`, `2`, `3`: chọn trạm đang làm việc.
- `Space`: chụp ảnh cân lõi và nhận diện đúng camera đang chọn. Giữ yên cân
  cho đến khi trạng thái chuyển sang `CHỜ ẢNH QR`.
- `Q`: sau khi có số cân lõi, đưa tem QR vào camera và chụp ảnh thứ hai. Mã SP
  được tự điền từ ảnh này; ảnh cân lõi vẫn được giữ nguyên.
- `Backspace`: bỏ ngay lần đang xem, không hỏi xác nhận. Khi đang đặt con trỏ
  trong ô QR hoặc số cân, Backspace vẫn chỉ xóa ký tự như bình thường.
- Kiểm tra số cân, mã SP và cả hai ảnh bằng chứng.
- `Enter`: lưu mã SP + số cân lõi + hai ảnh trong đúng một event. Sau khi lưu thành công, giao diện tự chọn
  trạm tiếp theo nếu đang bật luân phiên.
- Không rút camera hoặc tắt máy khi còn bản ghi chưa đồng bộ.

Dữ liệu local, ảnh, SQLite và log nằm tại:

```text
%LOCALAPPDATA%\TramCanQR
```

Sao lưu toàn bộ thư mục này trước khi đổi máy, gỡ ứng dụng hoặc nâng cấp.

Đóng tab trình duyệt chưa dừng gateway nền. Trước khi khởi động lại hoặc nâng
cấp, mở **Task Manager**, chọn `TramCanQR.exe` và bấm **End task**; sau đó mở
lại shortcut. Không chạy hai bản gateway cùng lúc trên cổng 8080.

## 4. Đồng bộ cloud

Cloud là tùy chọn. Khi chưa cấu hình API, ứng dụng vẫn lưu local. Token ingest
và lookup phải được đơn vị triển khai chuyển bằng kênh riêng rồi điền vào
`config.env`.

Không đặt Cloudinary API secret, Supabase service-role key hoặc bất kỳ secret
quản trị nào trên máy khách. Máy khách chỉ dùng token thiết bị có quyền tối
thiểu.

## 5. Kiểm tra bàn giao tại xưởng

Trước khi ký nghiệm thu, thực hiện tối thiểu:

1. Chụp 100 lượt bằng ảnh thật độc lập ở đúng vị trí lắp đặt.
2. Kiểm tra đủ QR, cân, ảnh và đúng trạm/camera của từng lượt.
3. Thử mất mạng, retry, rút/cắm lại từng camera và khởi động lại máy.
4. Chạy liên tục ít nhất 60 phút trên đúng máy và hub USB sẽ sử dụng.
5. Xác nhận outbox về 0 sau khi mạng phục hồi.

Nếu số cân hoặc QR sai, không lưu tiếp hàng loạt. Giữ nguyên ảnh lỗi và file
log `%LOCALAPPDATA%\TramCanQR\logs\app.log` để hiệu chỉnh.

Hệ thống cố ý không tự nhận khi lõi nét LED nguồn thấp hơn 16 px hoặc các frame
không đồng thuận. Đây là trạng thái cần chỉnh lại góc/độ phân giải camera, không
được hạ ngưỡng để ép lưu.
