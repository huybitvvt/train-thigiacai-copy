# Checklist bàn giao trạm cân

## Cấu hình bắt buộc

- Render Web Service dùng gói trả phí, region Singapore; không còn cảnh báo spin down.
- Persistent Disk gắn đúng service: mount path `/var/data`, tối thiểu 1 GB.
- Health check `/api/health` trả `ok: true` và `release` đúng commit Git vừa bàn giao.
- Các secret chỉ nằm trong Render Environment/Supabase Secrets; không gửi file `.env` có key thật.
- Sau khi thay Gemini key trên giao diện phải thấy thông báo `ĐÃ ÁP DỤNG KEY GEMINI MỚI`.

## Test nghiệm thu tại máy trạm

1. Cho phép camera vĩnh viễn cho hostname Render và chọn đúng USB camera.
2. Tải lại trang: camera đã gán phải tự mở, không bật/tắt camera mặc định trước.
3. Chụp cân lõi: nhận đúng số, giữ ảnh và số cân lõi.
4. Chụp cân sản phẩm: nhận đúng số cân sản phẩm; QR độc lập tự điền mã nếu đọc được.
5. Nhập/kiểm tra lệnh sản xuất rồi nhấn Enter đúng một lần.
6. Kiểm tra một bản ghi Supabase có cùng `event_id`, mã SP, hai số cân và hai ảnh.
7. Lặp 5 vòng liên tiếp; không được trắng khung, kẹt nút hoặc tạo bản ghi trùng.

## Test sự cố

- Ngắt mạng trước khi lưu: giao diện phải báo lưu cục bộ; nối mạng lại phải tự đồng bộ.
- Nhập Gemini key sai: hệ thống phải từ chối và giữ key đang hoạt động.
- Che màn hình LED hoặc để ảnh mờ: hệ thống phải yêu cầu chụp lại, không tự lưu số đoán.
- Rút/cắm lại USB camera: camera phải tự kết nối lại hoặc mở lại bằng nút `Mở đã gán`.

## Sau mỗi lần cập nhật

1. Push lên nhánh `main`.
2. Render tự deploy commit mới nhất.
3. Đợi trạng thái Live, mở `/api/health` để đối chiếu `release`.
4. Hard refresh trình duyệt (`Ctrl+F5`) rồi chạy lại một vòng cân hoàn chỉnh.

Lưu ý: thay code Edge Function trong `backend/supabase/functions` cần deploy lại Supabase Function; Render không tự làm bước này.
