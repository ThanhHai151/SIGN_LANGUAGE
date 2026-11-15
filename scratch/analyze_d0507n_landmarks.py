import os
import sys
import json

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.diacritic_state_machine import DiacriticStateMachine
from backend.vision.feature_extractor import calculate_distance, calculate_angle
from backend.vision.recognizer import VSLRecognizer

with open('scratch/d0507n_extracted_raw.json', encoding='utf-8') as f:
    d = json.load(f)

fsm = DiacriticStateMachine()
rec = VSLRecognizer()

print(f"{'Idx':<5} | {'t':<6} | {'d(8,12)':<8} | {'idx_pip':<8} | {'idx_curl':<8} | {'d_pw':<8} | {'d_pi':<8} | {'pred_sign':<10}")
print("-" * 75)

for s in d:
    if s.get('quality_status') != 'ok':
        continue
    idx = s['frame_index']
    t = s['timestamp_sec']
    lms = s['landmarks']
    mf = fsm._extract_frame_features(lms, t)
    sh = mf.shifted_landmarks
    ps = mf.palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    d_pi = mf.d_pinky_index_norm

    rec_res = rec.predict_landmarks(lms, mode='letters', timestamp=t, expected_sign='Ô')
    sign = rec_res.get('sign')
    tok = rec_res.get('new_token')
    rep = rec_res.get('replaced_token')

    token_str = f"{sign}"
    if tok:
        token_str += f" [NEW:{tok}]"
    if rep:
        token_str += f" [REP:{rep}]"

    print(f"{idx:<5} | {t:<6.3f} | {mf.d_index_middle_norm:<8.3f} | {mf.index_pip_angle:<8.1f} | {mf.index_curl:<8.3f} | {d_pw:<8.3f} | {d_pi:<8.3f} | {token_str:<25}")

