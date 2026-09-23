"""
Kịch Bản Huấn Luyện Mô Hình Học Sâu VSL (VSL PyTorch Training Pipeline)
Chức năng:
1. Tải tập dữ liệu từ backend/data/vsl_dataset.npz
2. Phân chia Train/Validation (80/20)
3. Huấn luyện mô hình VSLGestureNet với AdamW + CosineAnnealingLR
4. Đánh giá độ chính xác (Accuracy, Loss)
5. Lưu trọng số mô hình đã được hiệu chỉnh vào backend/models/weights.pt
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.models.sign_model import VSLGestureNet, CLASSES

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vsl_dataset.npz")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights.pt")

def train():
    if not os.path.exists(DATA_PATH):
        print(f"[Train] Không tìm thấy tập dữ liệu tại {DATA_PATH}. Đang tiến hành tạo dữ liệu...")
        from backend.training.download_dataset import build_augmented_dataset
        build_augmented_dataset()

    data = np.load(DATA_PATH)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    # Phân chia ngẫu nhiên Train 80% / Val 20%
    num_samples = len(X)
    indices = torch.randperm(num_samples)
    split_idx = int(num_samples * 0.8)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print(f"[Train] Tổng số mẫu: {num_samples} (Huấn luyện: {len(train_idx)}, Kiểm định: {len(val_idx)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VSLGestureNet(input_dim=72, num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    epochs = 25
    print(f"[Train] Bắt đầu huấn luyện {epochs} epochs trên thiết bị {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        # Đánh giá trên tập Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vout = model(vx)
                _, vpred = torch.max(vout.data, 1)
                val_total += vy.size(0)
                val_correct += (vpred == vy).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Lưu trọng số mô hình
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[Train] ✅ Huấn luyện hoàn tất! Trọng số đã lưu tại: {WEIGHTS_PATH}")
    print(f"[Train] Độ chính xác cuối cùng trên tập kiểm định: {val_acc:.2f}%")

if __name__ == "__main__":
    train()
