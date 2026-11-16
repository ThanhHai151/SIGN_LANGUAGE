import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.alphabet_classifier import alphabet_classifier, O_D_TI_MAX, O_D_TM_MAX, O_D_IW_MIN, O_D_MW_MIN

# 1. Inspect O.json failures
with open('backend/data/alphabet_v2/static_landmarks/O.json', 'r', encoding='utf-8') as f:
    o_json = json.load(f)

print("=== O.json Failures ===")
for idx, s in enumerate(o_json['samples']):
    if s.get('phase') != 'stable_pose': continue
    lm = s.get('landmarks')
    if not lm or len(lm) != 21:
        print(f"Sample {idx}: invalid landmarks")
        continue
    feat = alphabet_classifier.extract_features(lm)
    res = alphabet_classifier.classify_detailed(lm)
    if res['letter'] != 'O':
        d_ti = feat['d_TI']
        d_tm = feat['d_TM']
        d_iw = feat['d_IW']
        d_mw = feat['d_MW']
        d_rw = feat['d_RW']
        d_pw = feat['d_PW']
        i_ext = feat['index_ext']
        m_ext = feat['middle_ext']
        r_ext = feat['ring_ext']
        p_ext = feat['pinky_ext']
        print(f"O.json Sample {idx:2d}: status={res['status']} letter={res['letter']} | d_ti={d_ti:.3f} d_tm={d_tm:.3f} d_iw={d_iw:.3f} d_mw={d_mw:.3f} d_rw={d_rw:.3f} d_pw={d_pw:.3f} | ext=({i_ext},{m_ext},{r_ext},{p_ext})")

# 2. Inspect dataset v2 O failures
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes'].tolist()
y = npz['y']
lms = npz['landmarks']
o_idx = classes.index('O')

print("\n=== Dataset v2 O Failures ===")
for idx in range(len(y)):
    if y[idx] != o_idx: continue
    lm = lms[idx].tolist()
    res = alphabet_classifier.classify_detailed(lm)
    if res['letter'] != 'O':
        feat = alphabet_classifier.extract_features(lm)
        d_ti = feat['d_TI']
        d_tm = feat['d_TM']
        d_iw = feat['d_IW']
        d_mw = feat['d_MW']
        d_rw = feat['d_RW']
        d_pw = feat['d_PW']
        i_ext = feat['index_ext']
        m_ext = feat['middle_ext']
        r_ext = feat['ring_ext']
        p_ext = feat['pinky_ext']
        print(f"NPZ Sample {idx:2d}: status={res['status']} letter={res['letter']} | d_ti={d_ti:.3f} d_tm={d_tm:.3f} d_iw={d_iw:.3f} d_mw={d_mw:.3f} d_rw={d_rw:.3f} d_pw={d_pw:.3f} | ext=({i_ext},{m_ext},{r_ext},{p_ext})")

