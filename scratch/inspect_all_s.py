import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/S.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for s in d['samples']:
    if s.get('landmarks'):
        f_idx = s.get('frame_index')
        feat = clf.extract_features(s['landmarks'])
        res = clf.classify(s['landmarks'])
        print(f"Frame {f_idx}: res={res}, d_TW={feat['d_TW']}, d_IW={feat['d_IW']}, d_MW={feat['d_MW']}, thumb_lat={feat['thumb_lat']}")
