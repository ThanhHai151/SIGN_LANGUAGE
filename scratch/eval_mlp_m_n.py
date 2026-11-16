"""
Đánh giá độ chính xác của mô hình MLP Head (VSLStaticAlphabetHead) trên 22 lớp và riêng 2 lớp M, N.
"""
import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.models.alphabet_static_head import VSLStaticAlphabetHead, STATIC_CLASSES_22, DEFAULT_HEAD_WEIGHTS_PATH, load_static_alphabet_head

head_tuple = load_static_alphabet_head(DEFAULT_HEAD_WEIGHTS_PATH)
if head_tuple is None:
    print("Không nạp được static head weights!")
    sys.exit(1)
head = head_tuple[0]

from backend.vision.feature_extractor import normalize_hand_landmarks

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

m_idx = classes.index("M")
n_idx = classes.index("N")

# Dự đoán toàn bộ dataset bằng MLP
tp_m = fn_m = fp_m = 0
confused_m = {}
tp_n = fn_n = fp_n = 0
confused_n = {}

for lm, label in zip(landmarks, y):
    feat_72, _ = normalize_hand_landmarks(lm.tolist())
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    pred = res["raw_predicted_letter"]
    actual = classes[label]
    
    if actual == "M":
        if pred == "M": tp_m += 1
        else:
            fn_m += 1
            confused_m[pred] = confused_m.get(pred, 0) + 1
    elif pred == "M":
        fp_m += 1
        
    if actual == "N":
        if pred == "N": tp_n += 1
        else:
            fn_n += 1
            confused_n[pred] = confused_n.get(pred, 0) + 1
    elif pred == "N":
        fp_n += 1

print(f"MLP Head on M: TP={tp_m}/19 ({tp_m/19*100:.1f}%), FP={fp_m}, Confused to: {confused_m}")
print(f"MLP Head on N: TP={tp_n}/21 ({tp_n/21*100:.1f}%), FP={fp_n}, Confused to: {confused_n}")

# Kiểm tra xem những chữ nào bị MLP đoán nhầm thành M hoặc N
print("\nCác mẫu ngoài bị nhầm thành M:")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    feat_72, _ = normalize_hand_landmarks(lm.tolist())
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    pred = res["raw_predicted_letter"]
    actual = classes[label]
    if actual != "M" and pred == "M":
        print(f"  Sample {i} (Actual: {actual}): conf={res['confidence']:.3f}")

print("\nCác mẫu ngoài bị nhầm thành N:")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    feat_72, _ = normalize_hand_landmarks(lm.tolist())
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    pred = res["raw_predicted_letter"]
    actual = classes[label]
    if actual != "N" and pred == "N":
        print(f"  Sample {i} (Actual: {actual}): conf={res['confidence']:.3f}")
