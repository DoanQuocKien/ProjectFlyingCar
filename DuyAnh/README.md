# 🚗 Hand Gesture Controlled ESP32 Car using MediaPipe & Deep Learning

## 📌 Tổng quan

Dự án xây dựng hệ thống điều khiển xe robot ESP32 bằng cử chỉ tay theo thời gian thực thông qua webcam.

Hệ thống sử dụng:

- **MediaPipe Hands** để phát hiện và theo dõi 21 điểm mốc (landmarks) của bàn tay.
- **PyTorch MLP Classifier** để nhận dạng cử chỉ.
- **OpenCV** để xử lý và hiển thị video.
- **ESP32** để điều khiển động cơ xe.
- **HTTP Communication** để truyền lệnh từ máy tính tới ESP32 qua WiFi.

Người dùng chỉ cần thực hiện các cử chỉ tay trước camera để điều khiển xe tiến, lùi, rẽ trái, rẽ phải hoặc dừng.

---



# 🏗️ Kiến trúc hệ thống

```text
Webcam
   │
   ▼
MediaPipe Hands
   │
   ▼
Feature Extraction
   │
   ▼
MLP Gesture Classifier
   │
   ▼
Gesture Mapping
   │
   ▼
HTTP Request
   │
   ▼
ESP32 Controller
   │
   ▼
Motor Driver + Robot Car
```

---

# ✋ Các cử chỉ được hỗ trợ

| Cử chỉ | Nhãn | Lệnh |
|---------|---------|---------|
| ✊ Fist | fist | Stop |
| ☝️ One | one | Forward |
| ✌️ Peace | peace | Right |
| 🤟 Three | three | Left |
| 🖐️ Four | four | Backward |

Bảng ánh xạ trong chương trình:

```python
GESTURE_MAP = {
    "one": "forward",
    "peace": "right",
    "three": "left",
    "four": "backward",
    "fist": "stop"
}
```

---




### Ưu điểm

- Nhẹ.
- Tốc độ suy luận nhanh.
- Phù hợp với hệ thống thời gian thực.
- Dễ triển khai trên máy tính cấu hình thấp.


---

# 🔧 Hiệu chỉnh động cơ

Do hai động cơ thường không quay chính xác như nhau nên hệ thống hỗ trợ cân chỉnh:

```python
LEFT_TRIM = 1.0
RIGHT_TRIM = 0.82
```

Tốc độ thực tế:

```python
speed_l = active_speed * LEFT_TRIM
speed_r = active_speed * RIGHT_TRIM
```

Giúp xe chạy thẳng ổn định hơn.

---

# 📉 Chống gửi lệnh dư thừa

Để giảm tải mạng WiFi và ESP32:

Chương trình chỉ gửi lệnh mới khi:

- Cử chỉ thay đổi.
- Trạng thái Boost thay đổi.

Nhờ đó:

- Giảm số lượng request.
- Giảm độ trễ.
- Tăng độ ổn định.


---

# ⚙️ Cài đặt

## 1. Clone repository

```bash
git clone <repository-url>
cd project
```

## 2. Cài đặt thư viện

```bash
pip install opencv-python
pip install mediapipe
pip install torch
pip install numpy
pip install requests
```

Hoặc:

```bash
pip install -r requirements.txt
```

---

# ▶️ Chạy chương trình

Cập nhật IP của ESP32:

```python
ESP32_IP = "http://YOUR_ESP32_IP"
```

Khởi chạy:

```bash
python main.py
```

Thoát chương trình:

```text
ESC
```

---

# 📊 Kết quả

Hệ thống có khả năng:

- Nhận dạng cử chỉ theo thời gian thực.
- Điều khiển xe thông qua WiFi.
- Hỗ trợ tăng tốc bằng khoảng cách tay.
- Hoạt động ổn định với độ trễ thấp.

