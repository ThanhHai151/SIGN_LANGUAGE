import json
import sys
sys.path.insert(0, ".")

import backend.vision.diacritic_state_machine as dsm
from backend.vision.feature_extractor import calculate_distance

# Patch COMPETING_RULES_MAP
dsm.COMPETING_RULES_MAP["O"] = ("I", "E")

# Add d_pinky_wrist_norm to _extract_frame_features
orig_extract = dsm.DiacriticStateMachine._extract_frame_features
def patched_extract(self, landmarks_raw, timestamp):
    mf = orig_extract(self, landmarks_raw, timestamp)
    sh = mf.shifted_landmarks
    ps = mf.palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    mf.d_pinky_wrist_norm = d_pw
    return mf
dsm.DiacriticStateMachine._extract_frame_features = patched_extract

# Patch _evaluate_diacritic_for_group
orig_eval = dsm.DiacriticStateMachine._evaluate_diacritic_for_group
def patched_eval(self, base, curr_frame, baseline):
    if base == "O":
        d_pw = getattr(curr_frame, 'd_pinky_wrist_norm', 1.0)
        base_pw = getattr(baseline, 'd_pinky_wrist_norm', 1.0)
        # 1. Ô: Index and middle spread to form circumflex, bent at PIP, pinky folded
        if (curr_frame.d_index_middle_norm >= 0.20 and 
            curr_frame.index_pip_angle <= 110.0 and 
            curr_frame.index_curl >= 0.35 and 
            d_pw <= 0.85):
            return "Ô", f"Tạo nón trỏ-giữa (d_IM={curr_frame.d_index_middle_norm:.2f}>=0.20, PIP={curr_frame.index_pip_angle:.1f}<=110.0)"
        
        # 2. Ơ: Index and middle stay in O loop, pinky extends
        d_pw_delta = d_pw - base_pw
        if (curr_frame.d_index_middle_norm <= 0.18 and 
            d_pw >= 1.25 and 
            (d_pw_delta >= 0.20 or curr_frame.d_pinky_index_norm >= self.o_horn_dist_threshold)):
            return "Ơ", f"Bật râu ngón út (d_PW={d_pw:.2f}>=1.25, delta={d_pw_delta:+.2f})"
        return None
    return orig_eval(self, base, curr_frame, baseline)
dsm.DiacriticStateMachine._evaluate_diacritic_for_group = patched_eval

# Patch is_competing_label to support tuple
def patched_is_competing(self, label: str) -> bool:
    if not self.current_base:
        return False
    competing = dsm.COMPETING_RULES_MAP.get(self.current_base)
    if isinstance(competing, (tuple, list, set)):
        return label in competing
    return competing == label
dsm.DiacriticStateMachine.is_competing_label = patched_is_competing

from backend.vision.recognizer import VSLRecognizer
rec = VSLRecognizer()

with open('backend/data/alphabet_v2/motion_sequences/Ô.json', encoding='utf-8') as f:
    d = json.load(f)

seq = d['sequence']
session_id = "test_o_sim_patched"
rec.reset()
rec.reset_diacritic_fsm(session_id)

tokens = []
diac_frames = []

for item in seq:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec')
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=session_id, frame_idx=idx, timestamp=t, expected_sign='Ô')
    sign = res.get('sign')
    tok = res.get('new_token')
    src = res.get('source')
    diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
    if tok:
        tokens.append((idx, round(t, 2), tok, src))
    if sign == 'Ô' or diag.get('diacritic_detected') == 'Ô':
        diac_frames.append((idx, round(t, 2), sign, diag.get('state'), diag.get('reason')))

print("Patched Test on Ô.json:")
print("Tokens emitted:", tokens)
print("Diacritic Ô frames count:", len(diac_frames))
if diac_frames:
    print("First 3 Ô frames:", diac_frames[:3])

