import sys
sys.path.insert(0, ".")
import json
from backend.vision.alphabet_classifier import alphabet_classifier

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples = data.get('samples', [])
print(f"Total samples: {len(samples)}")
for s in samples:
    idx = s['frame_index']
    phase = s['phase']
    lm = s['landmarks']
    res = alphabet_classifier.classify_detailed(lm)
    status = res.get('status')
    letter = res.get('letter')
    conf = res.get('confidence')
    reason = res.get('reason')
    candidates = res.get('matched_candidates', [])
    feat = res.get('features')
    if feat:
        d_pw = feat.get('d_PW', 0.0)
        d_iw = feat.get('d_IW', 0.0)
        d_mw = feat.get('d_MW', 0.0)
        d_tw = feat.get('d_TW', 0.0)
        d_rw = feat.get('d_RW', 0.0)
        d_ti = feat.get('d_TI', 0.0)
        p_curl = feat.get('pinky_curl', 0.0)
        p_straight = feat.get('pinky_straight', 0.0)
        p_horiz = feat.get('pinky_horiz_ratio', 0.0)
        ext = f"p={feat.get('pinky_ext')} t={feat.get('thumb_ext')} i={feat.get('index_ext')} m={feat.get('middle_ext')} r={feat.get('ring_ext')}"
        dists = f"d_PW={d_pw:.2f} d_IW={d_iw:.2f} d_MW={d_mw:.2f} d_RW={d_rw:.2f} d_TW={d_tw:.2f} d_TI={d_ti:.2f}"
        print(f"f{idx:02d} ({phase:11s}): status={status:8s} letter={letter} conf={conf} | {ext} | {dists} | cand={candidates}")
    else:
        print(f"f{idx:02d} ({phase:11s}): status={status:8s} letter={letter} conf={conf} reason={reason}")
