import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.alphabet_classifier import alphabet_classifier

with open('backend/data/alphabet_v2/static_landmarks/O.json', 'r', encoding='utf-8') as f:
    o_data = json.load(f)

stable_o = [s for s in o_data['samples'] if s.get('phase') == 'stable_pose']
print(f"Total stable O samples: {len(stable_o)}")

for idx, s in enumerate(stable_o):
    lm = s['landmarks']
    feat = alphabet_classifier.extract_features(lm)
    res = alphabet_classifier.classify_detailed(lm)
    
    # Check current rule O conditions
    i_ext = feat['index_ext']
    m_ext = feat['middle_ext']
    r_ext = feat['ring_ext']
    p_ext = feat['pinky_ext']
    d_ti = feat['d_TI']
    d_tm = feat['d_TM']
    d_iw = feat['d_IW']
    d_mw = feat['d_MW']
    
    cond_no_ext = (not i_ext and not m_ext and not r_ext and not p_ext)
    cond_d_ti = (d_ti <= 0.22)
    cond_d_tm = (d_tm <= 0.20)
    cond_d_iw = (d_iw >= 0.95)
    cond_d_mw = (d_mw >= 0.90)
    
    match = res['letter'] == 'O'
    print(f"Sample {idx:2d}: Match={match} | no_ext={cond_no_ext} (i={i_ext},m={m_ext},r={r_ext},p={p_ext}) | d_ti={d_ti:.3f} (<=0.22:{cond_d_ti}) | d_tm={d_tm:.3f} (<=0.20:{cond_d_tm}) | d_iw={d_iw:.3f} (>=0.95:{cond_d_iw}) | d_mw={d_mw:.3f} (>=0.90:{cond_d_mw})")

