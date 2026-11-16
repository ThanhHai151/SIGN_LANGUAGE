import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, compute_finger_states

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for s in d['samples']:
    if s.get('letter') in ['t', 'u', 's'] and s.get('landmarks'):
        p = s['landmarks']
        sh, ps = _shift_and_scale(p)
        states = compute_finger_states(sh, ps)
        d_ti = calculate_distance(sh[4], sh[8]) / ps
        d_tm = calculate_distance(sh[4], sh[12]) / ps
        d_im = calculate_distance(sh[8], sh[12]) / ps
        d_iw = calculate_distance(sh[8], sh[0]) / ps
        d_mw = calculate_distance(sh[12], sh[0]) / ps
        d_rw = calculate_distance(sh[16], sh[0]) / ps
        d_pw = calculate_distance(sh[20], sh[0]) / ps
        print(f"Fixture {s.get('letter')}: phase={s.get('phase')}, vid={s.get('video_id')}, d_TI={d_ti:.2f}, d_IM={d_im:.2f}, d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_RW={d_rw:.2f}, d_PW={d_pw:.2f}, states={[k for k, v in states.items() if 'ext' in k and v]}")

