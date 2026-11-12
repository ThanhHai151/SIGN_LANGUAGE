"""
Mô-đun Phân loại Bảng chữ cái Tĩnh VSL Độc lập (VSLStaticAlphabetHead)
====================================================================
Dự án: VSL Connect AI (Task 6B)
Mục tiêu: Bổ sung khả năng nhận diện 10 chữ cái tĩnh chưa có geometric rules
(G, H, K, M, N, P, Q, R, U, X) và nhận diện toàn bộ 22 chữ cái tĩnh nền VSL.

Kiến trúc:
- Input: Vector đặc trưng 72D chuẩn hóa (từ normalize_hand_landmarks)
- Hidden layers: 72 -> 64 (LayerNorm + LeakyReLU + Dropout) -> 32 -> 22 lớp đầu ra
- Output: 22 nhãn tư thế nền tương ứng với bảng chữ cái tiếng Việt VSL chuẩn.
- Hỗ trợ cơ chế từ chối dự đoán (Rejection threshold) khi confidence không đạt ngưỡng an toàn.
"""

import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Danh sách 22 lớp tư thế nền chuẩn hóa theo alphabet_labels_v2.json
STATIC_CLASSES_22 = [
    "A", "B", "C", "D", "E", "G", "H", "I", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y"
]

CLASS_TO_IDX_22 = {c: i for i, c in enumerate(STATIC_CLASSES_22)}
IDX_TO_CLASS_22 = {i: c for i, c in enumerate(STATIC_CLASSES_22)}

DEFAULT_HEAD_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "alphabet_static_head_v1.pt"
)

# SHA-256 của checkpoint nội bộ đã xác minh provenance (Task 0.B / 0.C Safety Baseline)
KNOWN_HEAD_CHECKPOINT_SHA256 = "cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254"

class VSLStaticAlphabetHead(nn.Module):
    """
    Mạng nơ-ron nhỏ gọn (Compact MLP Head) phân loại 22 tư thế chữ cái tĩnh VSL.
    Được thiết kế độc lập, không can thiệp vào VSLGestureNet hay SSTCN.
    """
    def __init__(self, input_dim: int = 72, hidden_dim1: int = 64, hidden_dim2: int = 32, num_classes: int = 22, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(hidden_dim2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass trả về logits (chưa qua softmax)"""
        return self.net(x)

    def predict_feature(self, feature_vector: Any, confidence_threshold: float = 0.55) -> Dict[str, Any]:
        """
        Dự đoán nhãn chữ cái từ vector đặc trưng 72D.
        Hỗ trợ trạng thái từ chối dự đoán (UNKNOWN / LOW_CONFIDENCE) khi không đủ tin cậy.
        """
        if feature_vector is None:
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "INVALID_INPUT",
                "is_confident": False,
                "top_k": []
            }
            
        arr = np.asarray(feature_vector, dtype=np.float32).flatten()
        if len(arr) < 72 or not np.isfinite(arr[:72]).all():
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "INVALID_FEATURE",
                "is_confident": False,
                "top_k": []
            }
            
        feat_72 = arr[:72]
        self.eval()
        with torch.no_grad():
            inp = torch.from_numpy(feat_72).unsqueeze(0)
            logits = self.forward(inp)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            
        top_indices = np.argsort(probs)[::-1]
        top_1_idx = int(top_indices[0])
        top_1_conf = float(probs[top_1_idx])
        pred_letter = IDX_TO_CLASS_22.get(top_1_idx, "UNKNOWN")
        
        top_k = [
            {
                "letter": IDX_TO_CLASS_22.get(int(idx), "UNKNOWN"),
                "confidence": round(float(probs[idx]), 4)
            }
            for idx in top_indices[:5]
        ]
        
        is_confident = (top_1_conf >= confidence_threshold)
        status = "CONFIDENT" if is_confident else "LOW_CONFIDENCE"
        final_letter = pred_letter if is_confident else "UNKNOWN"
        
        return {
            "letter": final_letter,
            "raw_predicted_letter": pred_letter,
            "confidence": round(top_1_conf, 4),
            "status": status,
            "is_confident": is_confident,
            "top_k": top_k
        }

def load_static_alphabet_head(weights_path: Optional[str] = None, device: str = "cpu") -> Tuple[VSLStaticAlphabetHead, Dict[str, Any]]:
    """
    Tải mô-đun VSLStaticAlphabetHead kèm checkpoint đã huấn luyện.
    Đảm bảo tương thích cả PyTorch >= 2.4 (safe_globals) và PyTorch < 2.4 (như 2.2.2 hiện tại).
    Chỉ nạp weights_only=False khi checkpoint nội bộ đã xác minh provenance & SHA-256.
    """
    target_path = weights_path or DEFAULT_HEAD_WEIGHTS_PATH
    head = VSLStaticAlphabetHead(input_dim=72, hidden_dim1=64, hidden_dim2=32, num_classes=len(STATIC_CLASSES_22))
    metadata = {}
    
    if os.path.exists(target_path):
        if hasattr(torch.serialization, "safe_globals"):
            multiarray_mod = getattr(getattr(np, "_core", None), "multiarray", None) or getattr(getattr(np, "core", None), "multiarray", None)
            scalar_fn = getattr(multiarray_mod, "scalar", None) if multiarray_mod else None
            safe_numpy_globals = []
            if scalar_fn is not None:
                safe_numpy_globals.append((scalar_fn, "numpy.core.multiarray.scalar"))
            safe_numpy_globals.append((np.dtype, "numpy.dtype"))
            if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
                safe_numpy_globals.append((np.dtypes.Float64DType, "numpy.dtypes.Float64DType"))
            with torch.serialization.safe_globals(safe_numpy_globals):
                ckpt = torch.load(target_path, map_location=device, weights_only=True)
        else:
            # PyTorch < 2.4: Chỉ cho phép weights_only=False đối với checkpoint nội bộ đã kiểm chứng
            is_internal_default = (os.path.abspath(target_path) == os.path.abspath(DEFAULT_HEAD_WEIGHTS_PATH))
            file_sha256 = ""
            with open(target_path, "rb") as f:
                file_sha256 = hashlib.sha256(f.read()).hexdigest()
            
            if is_internal_default and file_sha256 == KNOWN_HEAD_CHECKPOINT_SHA256:
                ckpt = torch.load(target_path, map_location=device, weights_only=False)
            else:
                try:
                    ckpt = torch.load(target_path, map_location=device, weights_only=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Checkpoint không xác thực '{target_path}' (SHA-256: {file_sha256}) "
                        f"không thể nạp an toàn trên PyTorch {torch.__version__} thiếu safe_globals: {exc}"
                    ) from exc

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            head.load_state_dict(ckpt["model_state_dict"])
            metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        elif isinstance(ckpt, dict):
            head.load_state_dict(ckpt)
        head.eval()
        head.to(device)
    else:
        pass
        
    return head, metadata
