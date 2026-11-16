import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

print("=== VERIFYING EXIT FRAMES KINEMATICS ===")

def inspect_frames(letter, frame_list, desc):
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n--- {letter}: {desc} ---")
    for s in data['samples']:
        f_idx = s.get('frame_index') or s.get('frame_idx')
        if f_idx in frame_list and s.get('landmarks'):
            feat = clf.extract_features(s['landmarks'])
            print(f"  Frame {f_idx:2d}: idx_tip_y={feat['idx_tip_y']:+.3f}, d_IW={feat['d_IW']:.3f}, d_MW={feat['d_MW']:.3f}, d_RW={feat['d_RW']:.3f}, d_PW={feat['d_PW']:.3f}, ang_IM={feat['ang_IM_dir']:4.1f}°")

inspect_frames('P', [44, 46, 48, 50, 52, 54, 58, 60], "Stable (44-48) vs Exit (50-60)")
inspect_frames('K', [40, 44, 46, 48], "Stable (40, 44) vs Exit (46, 48)")
inspect_frames('Q', [54, 56, 58, 60, 62, 64], "Stable (54-58) vs Exit (60-64)")
inspect_frames('D', [48, 50, 52, 54, 56], "Stable (48-52) vs Exit (54, 56)")

