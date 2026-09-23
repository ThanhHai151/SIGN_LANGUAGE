"""
Mô hình Nhận diện Chuỗi Ký hiệu VSL Thời gian thực: SSTCN_Attention
Kiến trúc: Separable Spatial-Temporal Convolution Network kết hợp Joint Attention
Input shape: (Batch, Channels=3, Time=30, Joints=75)
Tương thích với bộ trọng số 102 lớp VSL được huấn luyện từ giáo trình chuẩn VSL.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

POSE_START, POSE_END = 0, 33
L_HAND_START, L_HAND_END = 33, 54
R_HAND_START, R_HAND_END = 54, 75

class JointAttention(nn.Module):
    def __init__(self, in_channels=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        # x: [B, C, J] -> [B, J, C] cho MultiheadAttention
        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)  # [B, C, J]
        return x

class SSTCN_Attention(nn.Module):
    def __init__(self, num_classes=102, num_joints=75, in_channels=3, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints

        # --- Spatial Convolution ---
        self.spatial_conv = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.spatial_bn = nn.BatchNorm1d(128)
        self.spatial_drop = nn.Dropout(dropout)
        
        # --- Multi-head Joint Attention riêng cho 2 bàn tay ---
        self.left_hand_attn = JointAttention(128, num_heads=4)
        self.right_hand_attn = JointAttention(128, num_heads=4)
        
        # --- Temporal Convolutions (Quét theo trục thời gian) ---
        self.temporal_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_bn1 = nn.BatchNorm2d(256)
        self.temporal_drop1 = nn.Dropout2d(dropout)
        
        self.temporal_conv2 = nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_bn2 = nn.BatchNorm2d(512)
        self.temporal_drop2 = nn.Dropout2d(dropout)
        
        # --- Phân loại cuối cùng ---
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
        # Chỉ mục các khớp bàn tay
        self.left_hand_joints = list(range(L_HAND_START, L_HAND_END))
        self.right_hand_joints = list(range(R_HAND_START, R_HAND_END))
    
    def forward(self, x):
        # x shape: [B, C, T, J] = [B, 3, 30, 75]
        B, C, T, J = x.shape
        
        # --- 1. Spatial conv trên từng frame độc lập ---
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, C, J]
        x = x.view(B * T, C, J)                 # [B*T, C, J]
        x = F.relu(self.spatial_bn(self.spatial_conv(x)))
        x = self.spatial_drop(x)
        
        # --- 2. Attention cho các khớp bàn tay --- 
        left_hand = x[:, :, self.left_hand_joints]   # [B*T, 128, 21]
        left_hand = self.left_hand_attn(left_hand)
        right_hand = x[:, :, self.right_hand_joints] # [B*T, 128, 21]
        right_hand = self.right_hand_attn(right_hand)

        # Cập nhật đặc trưng 2 tay
        x_new = x.clone()
        x_new.scatter_(2, torch.tensor(self.left_hand_joints).view(1, 1, -1).to(x.device).expand(B * T, 128, -1), left_hand)
        x_new.scatter_(2, torch.tensor(self.right_hand_joints).view(1, 1, -1).to(x.device).expand(B * T, 128, -1), right_hand)
        x = x_new
        
        # --- 3. Khôi phục shape cho Temporal Convolution ---
        x = x.view(B, T, 128, J).permute(0, 2, 1, 3)  # [B, 128, T, J]
        
        # --- 4. Temporal conv ---
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x = self.temporal_drop1(x)
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        x = self.temporal_drop2(x)
        
        # --- 5. Global Average Pooling qua thời gian và các khớp ---
        x = x.mean(dim=2).mean(dim=2)  # [B, 512]
        
        # --- 6. Classifier ---
        x = self.fc_drop(x)
        out = self.fc(x)               # [B, num_classes]
        return out

def load_sstcn_model(weights_path=None, device='cpu'):
    """
    Tải và khởi tạo mô hình SSTCN_Attention với trọng số pretrained.
    """
    model = SSTCN_Attention(num_classes=102).to(device)
    model.eval()

    # Đường dẫn mặc định
    default_candidates = [
        weights_path,
        os.path.join(os.path.dirname(__file__), "weights", "sign_sstcn_attention_model.pth"),
        "/home/thanhhai/Documents/UNIVERSITY/TOOL/SignLang/source/sign_sstcn_attention_model.pth"
    ]

    loaded = False
    for path in default_candidates:
        if path and os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)
                print(f"[SSTCN] Tải thành công trọng số pretrained từ: {path}")
                loaded = True
                break
            except Exception as e:
                print(f"[SSTCN] Cảnh báo lỗi tải trọng số từ {path}: {e}")

    if not loaded:
        print("[SSTCN] Cảnh báo: Không tìm thấy file trọng số hợp lệ. Mô hình khởi tạo với tham số ngẫu nhiên.")
        
    return model
