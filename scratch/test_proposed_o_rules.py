import json
import sys
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine, BaselineSnapshot
from backend.vision.feature_extractor import calculate_distance

fsm = DiacriticStateMachine()

def test_eval(base, lms, baseline_mf, curr_mf):
    sh = curr_mf.shifted_landmarks
    ps = curr_mf.palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    base_pw = calculate_distance(baseline_mf.shifted_landmarks[20], baseline_mf.shifted_landmarks[0]) / baseline_mf.palm_scale
    d_im = curr_mf.d_index_middle_norm
    d_pi = curr_mf.d_pinky_index_norm
    pip = curr_mf.index_pip_angle
    
    # Ơ rule:
    # 1. Index and Middle stay in O loop: d_im <= 0.18
    # 2. Pinky extends: d_pw >= 1.25 and (d_pw - base_pw) >= 0.25 and d_pi >= 0.50
    is_o_horn = (d_im <= 0.18 and d_pw >= 1.25 and (d_pw - base_pw) >= 0.25 and d_pi >= 0.50)
    
    # Ô rule:
    # 1. Index and Middle spread to form circumflex: d_im >= 0.20
    # 2. Index bent at PIP: pip <= 110.0 and curr_mf.index_curl >= 0.35
    # 3. Pinky folded in: d_pw <= 0.85
    is_o_hat = (d_im >= 0.20 and pip <= 110.0 and curr_mf.index_curl >= 0.35 and d_pw <= 0.85)
    
    if is_o_hat:
        return "Ô", f"Tạo nón trỏ-giữa (d_IM={d_im:.2f}>=0.20, PIP={pip:.1f}<=110.0)"
    if is_o_horn:
        return "Ơ", f"Bật râu ngón út (d_PW={d_pw:.2f}>=1.25, d_PI={d_pi:.2f}>=0.50)"
    return None

print("Testing proposed rules on Ô.json:")
with open('backend/data/alphabet_v2/motion_sequences/Ô.json', encoding='utf-8') as f:
    d = json.load(f)
seq = d['sequence']
base_mf = fsm._extract_frame_features(seq[20]['landmarks'], seq[20]['timestamp_sec'])
for item in seq:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec')
    lms = item.get('landmarks')
    if not lms or idx < 25 or idx > 55:
        continue
    mf = fsm._extract_frame_features(lms, t)
    res = test_eval("O", lms, base_mf, mf)
    if res:
        print(f"  Frame {idx:02d} (t={t:.2f}): {res}")

print("\nTesting proposed rules on Ơ.json:")
with open('backend/data/alphabet_v2/motion_sequences/Ơ.json', encoding='utf-8') as f:
    d = json.load(f)
seq = d['sequence']
base_mf = fsm._extract_frame_features(seq[36]['landmarks'], seq[36]['timestamp_sec'])
for item in seq:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec')
    lms = item.get('landmarks')
    if not lms or idx < 36 or idx > 60:
        continue
    mf = fsm._extract_frame_features(lms, t)
    res = test_eval("O", lms, base_mf, mf)
    if res:
        print(f"  Frame {idx:02d} (t={t:.2f}): {res}")

