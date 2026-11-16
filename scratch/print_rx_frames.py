import json
import sys

sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# 1. R frames
with open('backend/data/alphabet_v2/static_landmarks/R.json', 'r', encoding='utf-8') as f:
    r_data = json.load(f)

print("--- R SAMPLES FRAME BY FRAME ---")
for s in r_data['samples']:
    idx = s.get('frame_index') or s.get('frame_idx')
    lm = s.get('landmarks')
    if not lm:
        print(idx, "no landmarks")
        continue
    feat = clf.extract_features(lm)
    ps = feat['palm_scale']
    # Check landmarks of index tip (8) and middle tip (12)
    # Check X coordinates of index vs middle
    x8 = lm[8][0]
    x12 = lm[12][0]
    x5 = lm[5][0]
    x9 = lm[9][0]
    # MCP5 is base of index, MCP9 is base of middle
    # In standard right hand facing camera: MCP5.x < MCP9.x
    # If middle crosses behind/over index: does Tip 12 cross Tip 8?
    print(f"R f{idx:02d} ({s.get('phase'):11s}) | d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f} | idx_tip_y={feat['idx_tip_y']:.2f}, mid_tip_y={feat['mid_tip_y']:.2f} | d_IM={feat['d_IM']:.2f}, diff_x={feat['diff_x_IM']:.2f} | MCP5_x={x5:.3f}, MCP9_x={x9:.3f}, Tip8_x={x8:.3f}, Tip12_x={x12:.3f}")

# 2. X frames
with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

print("\n--- X SAMPLES FRAME BY FRAME ---")
for s in x_data['samples']:
    idx = s.get('frame_index') or s.get('frame_idx')
    lm = s.get('landmarks')
    if not lm:
        continue
    feat = clf.extract_features(lm)
    print(f"X f{idx:03d} ({s.get('phase'):11s}) | d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_TW={feat['d_TW']:.2f}, d_TI={feat['d_TI']:.2f}, d_TM={feat['d_TM']:.2f} | idx_str={feat['idx_straight']:.2f}, mid_str={feat['mid_straight']:.2f}, i_curl={feat['index_curl']:.2f} | idx_tip_y={feat['idx_tip_y']:.2f}, idx_dx={feat['idx_dx']:.2f}, idx_dy={feat.get('idx_dy', 0.0):.2f}, thumb_lat={feat['thumb_lat']:.2f}")
