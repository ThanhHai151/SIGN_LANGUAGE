import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

for letter in ['O', 'S', 'P', 'Q']:
    with open(f'backend/data/alphabet_v2/static_landmarks/{letter}.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'============================== LETTER {letter} ==============================')
    for s in d['samples']:
        if s.get('phase') == 'stable_pose' and s.get('landmarks'):
            lm = s['landmarks']
            feat = clf.extract_features(lm)
            print(f"Frame {s.get('frame_index') or s.get('frame_idx')}:")
            print(f"  thumb_ext={feat.get('thumb_ext')}, index_ext={feat.get('index_ext')}, mid_ext={feat.get('middle_ext')}, ring_ext={feat.get('ring_ext')}, pinky_ext={feat.get('pinky_ext')}")
            print(f"  d_TI={feat.get('d_TI')}, d_TM={feat.get('d_TM')}, d_TW={feat.get('d_TW')}, d_IM={feat.get('d_IM')}")
            print(f"  d_IW={feat.get('d_IW')}, d_MW={feat.get('d_MW')}, d_RW={feat.get('d_RW')}, d_PW={feat.get('d_PW')}")
            print(f"  idx_tip_y={feat.get('idx_tip_y')}, mid_tip_y={feat.get('mid_tip_y')}, idx_horiz={feat.get('idx_horiz_ratio')}, mid_horiz={feat.get('mid_horiz_ratio')}")
            print(f"  idx_dx={feat.get('idx_dx')}, idx_straight={feat.get('idx_straight')}, mid_straight={feat.get('mid_straight')}")
            print(f"  ang_TI={feat.get('ang_TI')}, ang_IM_dir={feat.get('ang_IM_dir')}, thumb_lat={feat.get('thumb_lat')}")
            break

