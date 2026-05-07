import cv2
import mediapipe as mp
import torch
import numpy as np
import requests
import time
import math # Thêm thư viện math để tính khoảng cách

# =========================
# CẤU HÌNH ĐIỀU KHIỂN XE & TUNING
# =========================
ESP32_IP = "http://192.168.137.12"

# -- KHU VỰC CÂN CHỈNH (TUNE) --
BASE_SPEED = 150  # Tốc độ cơ sở bình thường (0 - 255)
BOOST_SPEED = 250 # Tốc độ tối đa khi đưa tay lại gần
TURN_RATIO = 0.82 # Tốc độ khi rẽ

LEFT_TRIM = 1.0   # Bù lệch trái
RIGHT_TRIM = 0.82 # Bù lệch phải

# NGƯỠNG KHOẢNG CÁCH: Kích thước tay trên màn hình càng lớn -> tay càng gần. 
# Bạn hãy nhìn thông số "Distance: ..." hiện trên màn hình camera để tự chỉnh lại số này cho vừa tay!
DISTANCE_THRESHOLD = 0.36 

# Bảng dịch từ Cử chỉ -> Lệnh HTTP cho xe
GESTURE_MAP = {
    "one": "forward",
    "peace": "right",
    "three": "left",
    "four": "backward",
    "fist": "stop"
}

# Cập nhật hàm: Nhận thêm tham số target_base_speed
def send_car_command(cmd, target_base_speed=BASE_SPEED):
    try:
        # Nếu là lệnh rẽ, áp dụng hệ số giảm tốc độ vào tốc độ mục tiêu hiện tại (thường/boost)
        if cmd in ["left", "right"]:
            active_speed = int(target_base_speed * TURN_RATIO)
        else:
            active_speed = target_base_speed

        speed_l = int(active_speed * LEFT_TRIM)
        speed_r = int(active_speed * RIGHT_TRIM)

        url = f"{ESP32_IP}/{cmd}?speedL={speed_l}&speedR={speed_r}"
        
        requests.get(url, timeout=0.3)
        print(f"🚗 Lệnh: {cmd.upper()} | Tốc độ: {'BOOST ' if target_base_speed == BOOST_SPEED else 'NORMAL'} (L:{speed_l} | R:{speed_r})")
        return True 
    except requests.exceptions.RequestException:
        print(f"⚠️ Mạng lag! Lệnh {cmd.upper()} bị rớt giữa đường, đang thử lại...")
        return False

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("gesture_model.pth", weights_only=False)

classes = checkpoint["classes"]
input_dim = checkpoint["input_dim"]
num_classes = len(classes)

class GestureMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GestureMLP(input_dim, num_classes).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================
# PREPROCESS
# =========================
def extract_feature(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y])

    coords = np.array(coords, dtype=np.float32)
    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale

    return coords.flatten()

# =========================
# PREDICT
# =========================
def predict(vector):
    x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
    return classes[pred]

# =========================
# CAMERA LOOP
# =========================
cap = cv2.VideoCapture(0)

last_pred = None
last_is_boost = False # Cờ lưu trạng thái tốc độ cuối cùng để chống spam

print("📷 Hệ thống đã sẵn sàng. Đưa tay lên để điều khiển xe!")
print("Nút chức năng: ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    current_pred = None
    is_boost = False
    hand_distance = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # TÍNH KHOẢNG CÁCH TAY ĐỂ XÉT ĐIỀU KIỆN BOOST
            x1 = hand_landmarks.landmark[0].x # Cổ tay
            y1 = hand_landmarks.landmark[0].y
            x2 = hand_landmarks.landmark[9].x # Khớp ngón giữa
            y2 = hand_landmarks.landmark[9].y
            
            hand_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if hand_distance > DISTANCE_THRESHOLD:
                is_boost = True

            vector = extract_feature(hand_landmarks)
            current_pred = predict(vector)

    # =========================
    # LOGIC KÍCH HOẠT (ĐÃ CẬP NHẬT)
    # =========================
    if current_pred is not None:
        # Kích hoạt GỬI LỆNH MỚI khi: Cử chỉ đổi HOẶC trạng thái Boost đổi
        if current_pred != last_pred or is_boost != last_is_boost:
            cmd = GESTURE_MAP.get(current_pred, "stop")
            
            # Chọn tốc độ truyền đi
            target_speed = BOOST_SPEED if is_boost else BASE_SPEED
            
            if send_car_command(cmd, target_base_speed=target_speed):
                last_pred = current_pred
                last_is_boost = is_boost 
                
    else:
        # Mất tay -> Gửi lệnh dừng bằng tốc độ BASE
        if last_pred is not None:
            if send_car_command("stop", BASE_SPEED):
                last_pred = None
                last_is_boost = False

    # =========================
    # HIỂN THỊ THÔNG TIN LÊN MÀN HÌNH
    # =========================
    if current_pred:
        cv2.putText(frame, f"C/Chi: {current_pred}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        active_cmd = GESTURE_MAP.get(current_pred, "stop").upper()
        cv2.putText(frame, f"Lenh Xe: {active_cmd}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Hiển thị thông số khoảng cách để bạn dễ TUNE lại biến DISTANCE_THRESHOLD
        cv2.putText(frame, f"Dist: {hand_distance:.3f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Hiển thị hiệu ứng Boost
        if is_boost:
            cv2.putText(frame, ">>> BOOST MODE <<<", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    cv2.imshow("Hand Gesture Control ESP32", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        send_car_command("stop")
        break

cap.release()
cv2.destroyAllWindows()