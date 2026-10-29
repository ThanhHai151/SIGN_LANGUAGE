"""
Mô hình Học sâu PyTorch nhận diện cử chỉ Ngôn ngữ Ký hiệu VSL (VSLGestureNet)
Kiến trúc:
- Input Feature Vector (Tọa độ chuẩn hóa + đặc trưng hình học: 72 chiều)
- Các khối Dense + LayerNorm + LeakyReLU + Residual Connection
- Phân loại đa lớp với Softmax
- Hỗ trợ cả rule-based geometric fallback để đạt độ chính xác tối đa.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [
    # Bảng chữ cái ngón tay A-Z chuẩn
    "A", "B", "C", "D", "Đ", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    # Từ vựng giao tiếp VSL thông dụng & Sinh hoạt đời sống
    "XIN_CHAO", "CAM_ON", "XIN_VUI_LONG", "TAM_BIET", "XIN_LOI", "DONG_Y", "KHONG", 
    "YEU_THUONG", "GIUP_DO", "TOI", "BAN", "AN_COM", "UONG_NUOC", "BENH_VIEN", "GIA_DINH",
    "HOC_BAI", "NAU_AN", "LAU_NHA", "DI_NGU", "RUA_TAY", "DOC_SACH", "VIET_BAI", 
    "LAM_VIEC", "DI_HOC", "TAM", "KHOE_MANH", "MET_MOI", "VUI_VE", "BUON", "HOI", 
    "TRA_LOI", "CHO", "XIN",
    # Từ vựng đời sống & Câu dài nâng cao
    "CUOI_TUAN", "LICH_TRINH", "PHU_THUOC", "BAN_RON", "TRONG_TUAN", "THOI_GIAN", "IT",
    "DU_AN", "CA_NHAN", "DAM_BAO", "DU", "THU_GIAN", "NEU", "CO_THE", "THICH",
    "GAP_GO", "BAN_BE", "KHAM_PHA", "DIA_DIEM", "MOI", "CO_HOI", "THOI_QUEN", "HANG_NGAY",
    # Trạng thái chờ
    "NEUTRAL"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)

class VSLGestureNet(nn.Module):
    def __init__(self, input_dim=72, num_classes=len(CLASSES), hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.input_layer(x)
        features = self.res1(features)
        features = self.res2(features)
        logits = self.classifier(features)
        return logits

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.pt")

def get_trained_model(weights_path=None):
    """Khởi tạo và tải trọng số mô hình đã được hiệu chỉnh"""
    model = VSLGestureNet(input_dim=72, num_classes=len(CLASSES))
    model.eval()
    model.has_weights = False
    
    target_path = DEFAULT_WEIGHTS_PATH if weights_path is None else weights_path
    if target_path and os.path.exists(target_path):
        try:
            state_dict = torch.load(target_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.has_weights = True
            print(f"[VSLGestureNet] Loaded calibrated weights from {target_path}")
        except Exception as e:
            model.has_weights = False
            print(f"[VSLGestureNet] Warning loading weights: {e}. Using calibrated fallback.")
    elif target_path:
        print(f"[VSLGestureNet] Weights path not found: {target_path}. Running without weights.")
    return model
