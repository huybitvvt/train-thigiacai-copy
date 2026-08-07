# Frontend

Giao diện web của trạm cân nằm trong `index.html`. Backend đọc file này khi
khởi động; bản Windows đóng gói cùng file vào `assets/frontend`.

Frontend gọi các endpoint cùng origin như `/api/status`, `/api/analyze` và
`/api/capture`, nên không cần cấu hình CORS hay URL API riêng.
