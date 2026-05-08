import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================================
# 1. TRÍCH XUẤT VÀ CHUẨN HÓA DỮ LIỆU JSON
# ==========================================

def load_data_from_json(json_dir):
    features = []
    labels = []
    
    # Duyệt qua tất cả các file JSON trong thư mục
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(json_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ File lỗi JSON: {filename}")
                print(f"   → {e}")
                continue  # bỏ qua file này
            
            # Duyệt qua các id ảnh trong file JSON
            for img_id, info in data.items():
                if not info.get("landmarks") or not info.get("labels"):
                    continue
                
                # Lấy bàn tay đầu tiên và nhãn
                hand_landmarks = info["landmarks"][0] # Cấu trúc: [[x0,y0], [x1,y1], ...]
                label = info["labels"][0]
                
                # Chuẩn hóa tọa độ tương đối (lấy cổ tay [0] làm mốc)
                coords = np.array(hand_landmarks, dtype=np.float32)  # (21, 2)
                
                # 👉 (1) lấy gốc (wrist)
                coords = coords - coords[0]
                
                # 👉 (2) tính scale
                scale = np.max(np.linalg.norm(coords, axis=1))
                
                # 👉 (3) normalize
                if scale > 0:
                    coords = coords / scale
                
                vector = []
                
                # (A) tọa độ ()
                vector.extend(coords.flatten())
                
                features.append(vector)
                labels.append(label)
    return np.array(features, dtype=np.float32), np.array(labels)

TRAIN_DIR = r'C:\maycuaduyanh\archive\ann_train_val'
TEST_DIR = r'C:\maycuaduyanh\archive\ann_test'

print("Đang load train...")
X_train, y_train_text = load_data_from_json(TRAIN_DIR)

print("Đang load test...")
X_test, y_test_text = load_data_from_json(TEST_DIR)

# Mã hóa nhãn dạng text ('one', 'two'...) sang số (0, 1...)
label_encoder = LabelEncoder()

all_labels = np.concatenate([y_train_text, y_test_text])
label_encoder.fit(all_labels)

y_train = label_encoder.transform(y_train_text)
y_test = label_encoder.transform(y_test_text)

num_classes = len(label_encoder.classes_)

# ==========================================
# 2. XÂY DỰNG DATALOADER & KIẾN TRÚC MLP
# ==========================================
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch.nn as nn

class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureMLP(num_classes).to(device)



# ==========================================
# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, epochs=30):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        acc = correct / total

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Test Acc={acc:.4f}")


#train
train_model(model, train_loader, test_loader)
#save
torch.save({
    "model_state": model.state_dict(),
    "classes": label_encoder.classes_,
    "input_dim": X_train.shape[1]
}, "gesture_model.pth")

print("✅ Đã lưu model vào gesture_model.pth")
print("📁 Save tại:", os.getcwd())
#test
def evaluate_model(model, test_loader, label_encoder, device):
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = correct / total
    print(f"\n✅ Test Accuracy: {acc:.4f} ({correct}/{total})")

    # Decode label về text
    pred_text = label_encoder.inverse_transform(all_preds)
    true_text = label_encoder.inverse_transform(all_labels)

    return pred_text, true_text

pred_text, true_text = evaluate_model(model, test_loader, label_encoder, device)
print(classification_report(true_text, pred_text))