import sys
sys.path.insert(0, ".")
import json
from backend.vision.alphabet_classifier import alphabet_classifier

data = json.load(open('backend/data/alphabet_v2/static_landmarks/Y.json', encoding='utf-8'))
samples = [s for s in data['samples'] if s['phase'] == 'stable_pose' and s['landmarks'] is not None]
for s in samples:
    res = alphabet_classifier.classify_detailed(s['landmarks'])
    feat = res.get('features', {})
    d_pw = feat.get('d_PW')
    d_tw = feat.get('d_TW')
    d_ti = feat.get('d_TI')
    d_tm = feat.get('d_TM')
    t_curl = feat.get('thumb_curl')
    t_ext = feat.get('thumb_ext')
    print(f"f{s['frame_index']:02d}: d_PW={d_pw} d_TW={d_tw} d_TI={d_ti} d_TM={d_tm} thumb_ext={t_ext} thumb_curl={t_curl} cand={res.get('matched_candidates')}")
