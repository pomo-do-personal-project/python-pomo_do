# Pomo-AI: Focus Tracking & Gaze Detection Service

Dịch vụ AI theo dõi sự tập trung thời gian thực (Realtime Focus & Gaze Tracking) dành cho hệ sinh thái **Pomo-do**, hỗ trợ nhận diện qua camera, phát hiện xao nhãng / dùng điện thoại, và hiệu chuẩn đa màn hình linh hoạt.

---

## 🌟 Tính Năng Chính

1. **Nhận diện khuôn mặt & hướng nhìn (Face & Gaze Tracking)**:
   - Sử dụng **MediaPipe Face Mesh** (tương thích cả Solutions API lẫn Tasks Vision API mới nhất).
   - Đo lường góc nghiêng đầu (Yaw, Pitch) và hướng nhìn mắt so với màn hình.
   - Phát hiện nhắm mắt / ngủ gật khi mắt nhắm liên tục > 2.5 giây.

2. **Hiệu chuẩn nhiều màn hình động (Dynamic Multi-Screen Calibration)**:
   - Cho phép người dùng cấu hình từ 1 đến $N$ màn hình làm việc thông qua giao diện web.
   - Tọa độ các màn hình được tự động lưu và tải từ `config.json`.

3. **Phát hiện điện thoại & xao nhãng (Distraction Detection)**:
   - Tích hợp **YOLOv8n** và **YOLOv8n-pose** từ Ultralytics.
   - Nhận diện khi người dùng cầm điện thoại hoặc nhìn xa khỏi khu vực làm việc.

4. **Làm mượt trạng thái (Smoothing & Debounce)**:
   - Hàng đợi frame (deque) với cơ chế biểu quyết đa số (Majority Voting) giúp trạng thái ổn định, chống giật rung.

5. **FastAPI & WebSocket Server**:
   - Truyền tải dữ liệu trạng thái tập trung và khung hình trực quan (base64) theo thời gian thực tới frontend Next.js qua WebSocket.

---

## 📁 Cấu Trúc Thư Mục

```text
pomo-ai/
├── config.json          # Cấu hình tọa độ các màn hình đã hiệu chuẩn
├── detector.py          # Module xử lý AI: MediaPipe + YOLO + Logic tập trung
├── server.py            # FastAPI WebSocket server cung cấp API thời gian thực
├── test_cam.py          # Script kiểm tra webcam và AI độc lập bằng OpenCV
├── requirements.txt     # Danh sách thư viện phụ thuộc
├── .gitignore           # File loại trừ cho git
└── README.md            # Tài liệu dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt

### 1. Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python 3.10 trở lên:

```bash
# Tạo môi trường ảo (Virtual Environment)
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Kích hoạt môi trường (Linux / macOS)
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

> **Lưu ý về Model Weights:**
> - Các file mô hình (`yolov8n.pt`, `yolov8n-pose.pt`, `face_landmarker.task`) đã được đưa vào `.gitignore`.
> - Khi chạy lần đầu, `detector.py` sẽ **tự động tải** các file mô hình cần thiết từ máy chủ Google và Ultralytics.

---

## 💻 Cách Chạy Ứng Dụng

### Chạy WebSocket Server (cho Frontend Pomo-do kết nối)
```bash
python server.py
```
Mặc định server sẽ khởi chạy tại:
- **Health check**: `http://localhost:8000/`
- **Screens API**: `http://localhost:8000/screens`
- **WebSocket**: `ws://localhost:8000/ws`

### Kiểm tra Webcam độc lập (Standalone Preview)
Để kiểm tra góc nhìn, nhận diện mắt, điện thoại trực tiếp trên cửa sổ OpenCV:
```bash
python test_cam.py
```
- Nhấn phím `q` hoặc `Esc` trên cửa sổ camera để thoát.

---

## 📡 Giao Thức WebSocket (`/ws`)

### 1. Dữ liệu Server gửi về Client (mỗi frame):
```json
{
  "has_face": true,
  "looking_at_screen": true,
  "sleeping": false,
  "using_phone": false,
  "distracted": false,
  "focus_score": 100,
  "status_text": "Đang tập trung",
  "active_screen_id": 1,
  "annotated_frame": "data:image/jpeg;base64,..."
}
```

### 2. Lệnh Client gửi lên Server:
- **Thêm màn hình hiện tại**:
  ```json
  {"action": "add_screen"}
  ```
- **Xóa toàn bộ màn hình đã lưu**:
  ```json
  {"action": "clear_screens"}
  ```
