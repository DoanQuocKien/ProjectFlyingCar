import cv2
import mediapipe as mp
import torch
import numpy as np
import time

# =========================
# LOAD MODEL
# =========================
try:
    checkpoint = torch.load("gesture_model.pth", weights_only=False)
    classes = checkpoint["classes"]
    input_dim = checkpoint["input_dim"]
    num_classes = len(classes)
except FileNotFoundError:
    print("⚠️ Không tìm thấy file 'gesture_model.pth'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

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
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_feature(hand_landmarks):
    coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
    coords = np.array(coords, dtype=np.float32)
    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0: 
        coords = coords / scale
    return coords.flatten()

def predict(vector):
    x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
    return classes[pred]

# =========================
# CAMERA LOOP & TEST FPS
# =========================
cap = cv2.VideoCapture(0)

# Biến để tính FPS
prev_frame_time = 0
new_frame_time = 0

print(f"📷 Đang chạy model trên thiết bị: {device.type.upper()}")
print("📷 Hệ thống test FPS đã sẵn sàng. Bấm ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # --- TÍNH TOÁN FPS ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    current_pred = None

    # Nếu có tay thì trích xuất và đưa vào model
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            vector = extract_feature(hand_landmarks)
            current_pred = predict(vector)

    # =========================
    # HIỂN THỊ THÔNG TIN
    # =========================
    # Vẽ FPS
    cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Vẽ kết quả dự đoán
    if current_pred:
        cv2.putText(frame, f"C/Chi: {current_pred}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Test Model FPS", frame)
    
    # Bấm ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()