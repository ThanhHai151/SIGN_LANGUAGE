import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

for letter in ['K', 'D', 'Q', 'P']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f'=== {letter}.json (total_samples: {len(samples)}) ===')
    for s in samples:
        f_idx = s.get('frame_index') or s.get('frame_idx')
        phase = s.get('phase')
        lms = s.get('landmarks')
        if lms:
            feat = clf.extract_features(lms)
            tip_y = feat.get('idx_tip_y') if feat else None
            mid_straight = feat.get('mid_straight') if feat else None
            d_tm = feat.get('d_TM') if feat else None
            if letter in ['K', 'D'] and tip_y is not None and tip_y > 0:
                print(f'  Frame {f_idx} (phase: {phase}): idx_tip_y = {tip_y:.3f}, d_IW = {feat.get("d_IW"):.3f}, d_MW = {feat.get("d_MW"):.3f}, idx_dy = {feat.get("idx_dy"):.3f}')
            elif letter == 'Q' and ((mid_straight is not None and mid_straight >= 0.80) or (d_tm is not None and d_tm <= 0.90)):
                print(f'  Frame {f_idx} (phase: {phase}): mid_straight = {mid_straight:.3f}, d_TM = {d_tm:.3f}, idx_tip_y = {tip_y:.3f}')
