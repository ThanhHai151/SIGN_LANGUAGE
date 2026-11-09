import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

for letter in ['P', 'Q']:
    with open(f'backend/data/alphabet_v2/static_landmarks/{letter}.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'============================== LETTER {letter} ({d.get("video_id")}) ==============================')
    for s in d['samples']:
        if s.get('landmarks'):
            f_idx = s.get('frame_index')
            phase = s.get('phase')
            feat = clf.extract_features(s['landmarks'])
            print(f"F{f_idx} ({phase}): "
                  f"IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, RW={feat['d_RW']:.2f}, PW={feat['d_PW']:.2f}, "
                  f"TI={feat['d_TI']:.2f}, TM={feat['d_TM']:.2f}, TW={feat['d_TW']:.2f}, IM={feat['d_IM']:.2f}, "
                  f"ang_TI={feat['ang_TI']:.1f}, ang_IM={feat['ang_IM_dir']:.1f}, "
                  f"idx_tip_y={feat['idx_tip_y']:.2f}, mid_tip_y={feat['mid_tip_y']:.2f}, "
                  f"idx_horiz={feat['idx_horiz_ratio']:.2f}, mid_horiz={feat['mid_horiz_ratio']:.2f}, "
                  f"idx_dx={feat['idx_dx']:.2f}, straight=(I:{feat['idx_straight']:.2f}, M:{feat['mid_straight']:.2f})")

