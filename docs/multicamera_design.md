# Thiết kế và tiêu chí nghiệm thu multi-camera

## 1. Phạm vi và quyết định chính

Một gateway Python phục vụ tối đa ba điểm cân. Trình duyệt mở đồng thời các
preview, nhưng QR/OCR/YOLO chỉ chạy khi người vận hành nhấn `Space` và mọi tác
vụ nhận diện đi qua đúng một hàng đợi FIFO. `Enter` chỉ lưu session đang được
chọn. Cloud không nằm trên đường thành công của thao tác lưu: SQLite commit
thành công là mốc hoàn tất tại trạm, outbox đồng bộ nền sau đó.

Ba định danh không được dùng thay nhau:

| Trường | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| `gateway_id` | Máy tính/gateway dùng chung | `gateway-01` |
| `station_id` | Điểm cân vật lý | `station-02` |
| `camera_id` | Bí danh ổn định của camera vật lý tại điểm cân | `camera-02` |

`MediaDeviceInfo.deviceId` là định danh cục bộ của browser. UI lưu ánh xạ
`station_id -> deviceId` trong `localStorage`, mở stream bằng ràng buộc
`deviceId.exact`, rồi gắn camera đó với `camera_id` đã cấu hình phía backend.
Không gửi `deviceId` lên cloud vì giá trị này phụ thuộc origin/browser và có
thể đổi khi đổi profile hoặc cổng USB.

## 2. Luồng dữ liệu và ranh giới an toàn

```text
preview song song (browser)
        |
Space ở station đang chọn
        |
event_id mới + station_id + camera_id + JPEG
        |
staging JPEG bất biến + analysis_id + frame_sha256
        |
hàng đợi FIFO một worker -> QR/OCR/YOLO
        |
review riêng của station
        |
Enter ở station đang chọn
        |
đối chiếu analysis/event/station/camera/hash
        |
SQLite WAL + ảnh local + outbox (một transaction logic)
        |
worker nền -> Supabase + Cloudinary
```

Các bất biến bắt buộc:

1. Mỗi lần `Space` hợp lệ tạo `event_id` mới. Retry lưu dùng lại đúng
   `event_id` và không thêm hàng.
2. Một `analysis_id` khóa cứng `event_id`, `station_id`, `camera_id`, ảnh staged
   và `frame_sha256`. Bất kỳ trường nào khác trả HTTP `409`.
3. Một station chỉ có tối đa một lần cân chưa lưu. Muốn thay ảnh phải hủy rõ
   ràng qua `/api/session/discard`.
4. Response bất đồng bộ chỉ được cập nhật session đã tạo request; đổi card khi
   request đang chạy không được xóa hay ghi kết quả của card mới.
5. Một browser `deviceId` không được gán đồng thời cho hai station. Reconnect
   chỉ dùng lại chính `deviceId` đã gán; không fallback sang camera mặc định.
6. Hàng SQLite là nguồn danh tính cho payload outbox. Giá trị cấu hình trên
   worker chỉ là fallback cho hàng legacy.
7. Supabase giữ unique `event_id`; cùng event và cùng payload/hash trả
   `duplicate=true`, khác payload/hash trả conflict. Cloudinary dùng public ID
   xác định theo gateway/ngày/event và không overwrite.

## 3. Trạng thái của từng station

| Trạng thái | Ý nghĩa | Thao tác hợp lệ |
| --- | --- | --- |
| `idle` | Chưa có stream/ảnh chờ | gán camera, mở camera, chọn tệp/demo |
| `live` | Preview đang chạy | `Space` |
| `analyzing` | Đã khóa frame, đang chờ/chạy FIFO | chọn station khác |
| `review` | Có kết quả nhưng chưa đủ điều kiện tự động | sửa dữ liệu hoặc hủy |
| `ready` | Ảnh đạt và có QR/cân | `Enter` hoặc hủy |
| `saving` | Đang commit local | retry sau lỗi; không chụp đè |
| `error` | Phân tích thất bại nhưng frame vẫn được giữ | hủy rõ ràng |
| `disconnected` | Camera đã gán bị mất | reconnect đúng camera |

Sau commit, session quay về `live` nếu stream còn hoạt động, nếu không về
`idle`. Khi bật auto-advance, UI chọn station kế tiếp theo vòng tròn dựa trên
station vừa lưu, không dựa trên card người dùng có thể đã chọn trong lúc chờ.

## 4. Hợp đồng API

### `GET /api/status`

Trả `gateway_id`, `station_count`, cấu hình/trạng thái từng station và số liệu
hàng đợi. `inference.worker_count` luôn bằng `1`.

### `POST /api/analyze`

Request bắt buộc đối với luồng multi-camera:

```json
{
  "image": "data:image/jpeg;base64,...",
  "unit": "kg",
  "roi": "auto",
  "event_id": "UUID-v4",
  "station_id": "station-01",
  "camera_id": "camera-01"
}
```

Response thêm `analysis_id`, `event_id`, `station_id`, `camera_id`,
`frame_sha256` và `captured_at`. API cũ không có bộ định danh vẫn được giữ để
không phá luồng một camera hiện hữu.

### `POST /api/capture`

Ngoài QR/cân/ảnh, luồng mới phải gửi lại chính xác năm trường
`event_id`, `analysis_id`, `station_id`, `camera_id`, `frame_sha256`. Backend
kiểm tra binding và hash trước khi gọi `save_idempotent`. Response `201` có
`duplicate`, danh tính đầy đủ và trạng thái outbox.

### `POST /api/session/discard`

Hủy đúng `station_id` và `event_id` đang chờ. Event khác hoặc station khác bị
từ chối; không có thao tác hủy ngầm khi chọn camera/ảnh mới.

## 5. Khả năng tương thích và migration

- SQLite migration chỉ `ADD COLUMN`, tạo index danh tính và tính hash cho ảnh
  legacy còn tồn tại; không xóa hàng hay ảnh.
- `ROLL_SCALE_DEVICE_ID` và `--device-id` là alias legacy của `gateway_id`.
- Các cột danh tính cloud nullable để hàng cũ vẫn tra cứu được. Capture mới
  phải có đủ bộ định danh ở luồng multi-camera.
- Chế độ mặc định vẫn là một station; bật hai/ba station bằng
  `--station-count` nên không thay hành vi cài đặt cũ nếu chưa đổi cấu hình.

## 6. Tiêu chí nghiệm thu tự động

| Gate | Điều kiện đạt |
| --- | --- |
| Hồi quy | Toàn bộ pytest pass; bộ OCR chuẩn vẫn đúng 100/100 và `wrong_accepted=0` |
| Tuần tự inference | Nhiều request đồng thời vẫn FIFO, `worker_count=1`, không có hai hàm vision chạy chồng |
| Cô lập session | Thử chéo event/station/camera/frame đều bị từ chối và không tạo hàng |
| Idempotency local | 100 event tạo đúng 100 hàng; 100 retry giống hệt tạo 0 hàng mới; retry khác danh tính bị conflict |
| Outbox offline | Mất mạng còn đúng 100 pending; phục hồi có 100 synced và 0 pending |
| Phân bố | 100 lượt/3 station theo vòng tròn là `34/33/33`, không sai danh tính |
| Cloud canary | Capture mới có đủ ba ID, hai hash và ảnh; retry trả `duplicate=true`; lookup khớp |

Report stress logic không được coi là nghiệm thu camera hoặc OCR vì nó dùng ảnh
và sender tổng hợp.

## 7. Tiêu chí nghiệm thu tại đúng xưởng

Chỉ ký nghiệm thu nhiều camera khi hoàn thành toàn bộ:

1. Dùng ít nhất 100 ảnh xưởng độc lập, không phải augmentation từ ba ảnh gốc;
   bao phủ nhiều số cân, khoảng cách, góc, độ cong tem và ánh sáng. Yêu cầu
   đúng tuyệt đối QR/cân theo ngưỡng dự án và `wrong_accepted=0`.
2. Chạy lần lượt cấu hình 1, 2 và 3 camera thật. Mọi preview phải còn hoạt động
   khi một station OCR; phím `1/2/3`, `Space`, `Enter` và auto-advance phải tác
   động đúng card trong 100 lượt xen kẽ.
3. Rút/cắm từng camera khi hai camera còn lại đang chạy. Card sai không được
   nhận stream thay thế; sau reconnect phải trở lại đúng station.
4. Ngắt mạng trong 100 lượt, restart gateway/browser, sau đó phục hồi mạng.
   Không mất hàng, không nhân đôi, không ghép ảnh/cân chéo và outbox về 0.
5. So khớp từng `event_id/gateway_id/station_id/camera_id/frame_sha256` giữa
   SQLite, ảnh local, Supabase và Cloudinary.
6. Soak tối thiểu 60 phút trên đúng máy và hub USB sẽ triển khai. Ghi CPU, RAM,
   FPS/độ rớt từng preview, thời gian chờ FIFO và latency QR/OCR p50/p95/max.
   Gate vận hành đề xuất: CPU trung bình dưới 80%, không tăng RAM liên tục,
   không camera disconnect ngoài bài test, local save p95 dưới 250 ms và hàng
   đợi phải trở về 0 sau burst ba request. Nếu không đạt, giảm resolution/FPS
   preview hoặc số camera trên gateway trước khi cho production.

Hiện chỉ có một webcam được phát hiện trên máy phát triển, nên các gate phần
cứng 2–3 camera, băng thông USB và soak vẫn là `CHƯA XÁC MINH`; không được suy
diễn từ test phần mềm.
