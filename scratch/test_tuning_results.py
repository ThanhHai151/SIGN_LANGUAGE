import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np

# Load user's webcam S frames
webcam_file = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(webcam_file, 'r', encoding='utf-8') as f:
    webcam_lines = [json.loads(l.strip()) for l in f if l.strip()]

webcam_s = [
    line['geometric']['features'] 
    for line in webcam_lines 
    if line.get('expected_sign') == 'S' and 'features' in line.get('geometric', {})
]

# Load dataset
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']

from backend.vision.alphabet_classifier import AlphabetClassifier
clf = AlphabetClassifier()

dataset_samples = []
for i in range(len(y)):
    lm = lms[i].tolist()
    feat = clf.extract_features(lm)
    dataset_samples.append((classes[y[i]], feat))

def eval_s(d_tw_max, d_iw_max, d_mw_max, d_rw_max, d_pw_max, d_ti_min, d_ti_max, d_tm_min, d_tm_max, thumb_lat_min):
    w_hits = 0
    for f in webcam_s:
        i_ext = f.get('index_ext', False)
        m_ext = f.get('middle_ext', False)
        r_ext = f.get('ring_ext', False)
        p_ext = f.get('pinky_ext', False)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            d_tw = f.get('d_TW', 99)
            d_iw = f.get('d_IW', 99)
            d_mw = f.get('d_MW', 99)
            d_rw = f.get('d_RW', 99)
            d_pw = f.get('d_PW', 99)
            d_ti = f.get('d_TI', 99)
            d_tm = f.get('d_TM', 99)
            th_lat = f.get('thumb_lat', 0)
            if (d_tw <= d_tw_max and d_ti_min <= d_ti <= d_ti_max and
                d_tm_min <= d_tm <= d_tm_max and d_iw <= d_iw_max and
                d_mw <= d_mw_max and d_rw <= d_rw_max and d_pw <= d_pw_max and
                th_lat >= thumb_lat_min):
                w_hits += 1

    d_s_hits = 0
    d_fp = 0
    fp_details = {}
    for lab, f in dataset_samples:
        if f is None: continue
        i_ext = f.get('i_ext', False)
        m_ext = f.get('m_ext', False)
        r_ext = f.get('r_ext', False)
        p_ext = f.get('p_ext', False)
        is_s = False
        if not i_ext and not m_ext and not r_ext and not p_ext:
            d_tw = f.get('d_TW', 99)
            d_iw = f.get('d_IW', 99)
            d_mw = f.get('d_MW', 99)
            d_rw = f.get('d_RW', 99)
            d_pw = f.get('d_PW', 99)
            d_ti = f.get('d_TI', 99)
            d_tm = f.get('d_TM', 99)
            th_lat = f.get('thumb_lat', 0)
            if (d_tw <= d_tw_max and d_ti_min <= d_ti <= d_ti_max and
                d_tm_min <= d_tm <= d_tm_max and d_iw <= d_iw_max and
                d_mw <= d_mw_max and d_rw <= d_rw_max and d_pw <= d_pw_max and
                th_lat >= thumb_lat_min):
                is_s = True
        if lab == 'S':
            if is_s: d_s_hits += 1
        else:
            if is_s:
                d_fp += 1
                fp_details[lab] = fp_details.get(lab, 0) + 1

    return w_hits, d_s_hits, d_fp, fp_details

# Let's test combinations
print("Testing tuned thresholds...")
for d_tm_max in [0.72, 0.74, 0.76, 0.78]:
    for th_lat_min in [0.55, 0.60]:
        for d_ti_max in [0.45, 0.48]:
            w, s, fp, fpc = eval_s(
                d_tw_max=1.40,
                d_iw_max=1.35,
                d_mw_max=0.85,
                d_rw_max=0.75,
                d_pw_max=0.85,
                d_ti_min=0.10,
                d_ti_max=d_ti_max,
                d_tm_min=0.35,
                d_tm_max=d_tm_max,
                thumb_lat_min=th_lat_min
            )
            print(f"d_tm_max={d_tm_max}, th_lat_min={th_lat_min}, d_ti_max={d_ti_max} => Webcam={w}/60, Dataset_S={s}/15, FP={fp} {fpc}")

