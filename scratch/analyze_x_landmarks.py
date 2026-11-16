import json
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

with open("backend/data/alphabet_v2/static_landmarks/X.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for s in data["samples"]:
    if s["frame_index"] in [40, 48, 116, 128]:
        lms = s["landmarks"]
        feat = c.extract_features(lms)
        wrist = lms[0]
        idx_mcp = lms[5]
        idx_pip = lms[6]
        idx_dip = lms[7]
        idx_tip = lms[8]
        thumb_tip = lms[4]
        
        print(f"\nFrame {s['frame_index']} (phase={s['phase']}):")
        print(f"  Wrist y: {wrist[1]:.3f}, Index MCP y: {idx_mcp[1]:.3f}, Index PIP y: {idx_pip[1]:.3f}, Index TIP y: {idx_tip[1]:.3f}")
        print(f"  Thumb TIP y: {thumb_tip[1]:.3f}")
        print(f"  idx_tip_y (rel to wrist): {feat['idx_tip_y']:.3f}, idx_dy: {feat['idx_dy']:.3f}")
        print(f"  d_TI (thumb-index): {feat['d_TI']:.3f}, d_IW: {feat['d_IW']:.3f}")
        print(f"  pred: {c.classify(lms)}")

