import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, json
from backend.vision.recognizer import VSLRecognizer
import backend.vision.alphabet_classifier as ac
import backend.vision.recognizer as rc

# Apply candidate patches in memory
rc.ALLOWED_FALLBACK_BASE_LETTERS = {'A', 'D', 'E', 'O'}
rc.DEFAULT_SUPPLEMENTAL_THRESHOLDS['N'] = 0.70
ac.H_HORIZ_ANG_IM_MAX = 35.0

orig_classify = ac.alphabet_classifier.classify_detailed
def patched_classify(landmarks_raw, *args, **kwargs):
    res = orig_classify(landmarks_raw, *args, **kwargs)
    if res.get('letter') == 'V' and res.get('features'):
        f = res['features']
        dx = abs(f.get('idx_dx', 0.0))
        dy = abs(f.get('idx_dy', 0.0))
        if dx > dy:
            cands = [m for m in res.get('matched_candidates', []) if m[0] != 'V']
            if not cands:
                res['status'] = 'NO_MATCH'
                res['letter'] = 'UNKNOWN'
                res['confidence'] = 0.0
                res['reason'] = 'V-guard: fingers are horizontal (|dx| > |dy|)'
            else:
                res['letter'] = cands[0][0]
                res['confidence'] = cands[0][2]
                res['status'] = 'MATCH'
                res['matched_candidates'] = cands
    return res

ac.alphabet_classifier.classify_detailed = patched_classify

rec = VSLRecognizer()

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(npz['classes'])
landmarks = npz['landmarks']
y = npz['y']

corr = 0
per_class = {c: {'corr': 0, 'tot': 0, 'preds': {}} for c in classes}
for i in range(len(y)):
    target = classes[y[i]]
    lm = landmarks[i].tolist()
    pred = rec.predict_landmarks(lm, mode='letters', session_id=f'cand_eval_{i}')['sign']
    per_class[target]['tot'] += 1
    per_class[target]['preds'][pred] = per_class[target]['preds'].get(pred, 0) + 1
    if pred == target:
        corr += 1
        per_class[target]['corr'] += 1

print(f"=== CANDIDATE COMBO ON LEGACY 551: {corr}/551 ({corr/551*100:.2f}%) ===")
for c in classes:
    st = per_class[c]
    p_str = ', '.join(f"{k}:{v}" for k, v in st['preds'].items() if k != c)
    pct = st['corr'] / st['tot'] * 100
    print(f"{c:<2}: {st['corr']}/{st['tot']} ({pct:.1f}%) | other preds: {p_str}")

# Also test on Canonical Suites
canonical_files = {
    "H": "backend/data/alphabet_v2/static_landmarks/D0500N.json",
    "O": "backend/data/alphabet_v2/static_landmarks/O.json",
    "Q": "backend/data/alphabet_v2/static_landmarks/Q.json",
    "R": "backend/data/alphabet_v2/static_landmarks/R.json",
    "M": "backend/data/alphabet_v2/static_landmarks/M.json",
    "N": "backend/data/alphabet_v2/static_landmarks/N.json",
    "V": "backend/data/alphabet_v2/static_landmarks/V.json",
}
print("\n=== CANDIDATE COMBO ON CANONICAL SUITES ===")
for letter, fpath in canonical_files.items():
    with open(fpath, "r", encoding="utf-8") as f:
        samples = [s for s in json.load(f).get("samples", []) if s.get("landmarks") and s.get("quality_status") == "ok"]
    c_corr = 0
    c_preds = {}
    for s in samples:
        p = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"canon_cand_{letter}")["sign"]
        c_preds[p] = c_preds.get(p, 0) + 1
        if p == letter:
            c_corr += 1
    p_str = ', '.join(f"{k}:{v}" for k, v in c_preds.items() if k != letter)
    pct = c_corr / len(samples) * 100
    print(f"{letter}: {c_corr}/{len(samples)} ({pct:.1f}%) | other preds: {p_str}")
