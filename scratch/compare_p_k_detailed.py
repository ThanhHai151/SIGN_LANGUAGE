import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/K.json', 'r', encoding='utf-8') as f:
    k_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/P.json', 'r', encoding='utf-8') as f:
    p_data = json.load(f)

print('=== K (D0501) ===')
for s in k_data['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        feat = clf.extract_features(s['landmarks'])
        f_idx = s.get('frame_index') or s.get('frame_idx')
        print(f"K f{f_idx}: IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, TI={feat['d_TI']:.2f}, TM={feat['d_TM']:.2f}, TW={feat['d_TW']:.2f}, ang_IM={feat['ang_IM_dir']:.1f}, thumb_lat={feat['thumb_lat']:.2f}")

print('\n=== P (D0509B) ===')
for s in p_data['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        feat = clf.extract_features(s['landmarks'])
        f_idx = s.get('frame_index') or s.get('frame_idx')
        print(f"P f{f_idx}: IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, TI={feat['d_TI']:.2f}, TM={feat['d_TM']:.2f}, TW={feat['d_TW']:.2f}, ang_IM={feat['ang_IM_dir']:.1f}, thumb_lat={feat['thumb_lat']:.2f}")

