import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # Thêm thư viện vẽ biểu đồ
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score # Thêm f1_score

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
# 3. HUẤN LUYỆN, LƯU LẠI LOSS & F1 SCORE
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, epochs=30):
    train_losses = []
    val_f1_scores = []

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

        # Tính Loss trung bình của epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # evaluate (Để lấy F1 Score)
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        acc = correct / total
        
        # Tính F1-Score (Macro) cho mỗi epoch
        epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        val_f1_scores.append(epoch_f1)

        print(f"Epoch {epoch+1:02d}/{epochs}: Loss={avg_loss:.4f}, Test Acc={acc:.4f}, Test F1={epoch_f1:.4f}")
        
    return train_losses, val_f1_scores

# 1. Train model và lấy dữ liệu vẽ biểu đồ
print("\n🚀 BẮT ĐẦU TRAINING...")
train_losses, val_f1_scores = train_model(model, train_loader, test_loader, epochs=30)


# ==========================================
# 4. ĐÁNH GIÁ CHI TIẾT & TÌM MẪU SAI
# ==========================================
def evaluate_and_show_errors(model, test_loader, label_encoder, device):
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
    print(f"\n✅ Tổng kết Test Accuracy: {acc:.4f} ({correct}/{total})")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Decode label về text
    pred_text = label_encoder.inverse_transform(all_preds)
    true_text = label_encoder.inverse_transform(all_labels)

    # In báo cáo
    print("\n" + "="*50)
    print("BÁO CÁO ĐÁNH GIÁ (CLASSIFICATION REPORT)")
    print("="*50)
    print(classification_report(true_text, pred_text))
    
    # In danh sách dự đoán sai
    print("\n" + "="*50)
    print("CHI TIẾT CÁC DỰ ĐOÁN SAI")
    print("="*50)
    
    incorrect_indices = np.where(all_preds != all_labels)[0]
    
    if len(incorrect_indices) == 0:
        print("🔥 Tuyệt vời! Mô hình đoán đúng 100% trên tập test này.")
    else:
        print(f"⚠️ Có {len(incorrect_indices)} / {len(all_labels)} mẫu bị dự đoán sai:\n")
        for idx in incorrect_indices:
            print(f" - Mẫu test số {idx:04d}: Đáng lẽ là [{true_text[idx].upper()}], model lại đoán thành [{pred_text[idx].upper()}]")

evaluate_and_show_errors(model, test_loader, label_encoder, device)

# ==========================================
# 5. VẼ BIỂU ĐỒ LOSS VÀ F1 SCORE
# ==========================================
def plot_training_metrics(losses, f1_scores):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', label='Training Loss', marker='o', markersize=4)
    plt.title('Training Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Biểu đồ F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, 'g-', label='Validation F1 Score', marker='s', markersize=4)
    plt.title('Validation F1 Score over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Gọi hàm vẽ biểu đồ
print("\n📈 Đang hiển thị biểu đồ...")
plot_training_metrics(train_losses, val_f1_scores)