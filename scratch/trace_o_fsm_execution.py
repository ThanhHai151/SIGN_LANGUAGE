import json
import sys
sys.path.insert(0, ".")

import backend.vision.diacritic_state_machine as dsm
from backend.vision.feature_extractor import calculate_distance

dsm.COMPETING_RULES_MAP["O"] = ("I", "E")

orig_extract = dsm.DiacriticStateMachine._extract_frame_features
def patched_extract(self, landmarks_raw, timestamp):
    mf = orig_extract(self, landmarks_raw, timestamp)
    sh = mf.shifted_landmarks
    ps = mf.palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    mf.d_pinky_wrist_norm = d_pw
    return mf
dsm.DiacriticStateMachine._extract_frame_features = patched_extract

orig_eval = dsm.DiacriticStateMachine._evaluate_diacritic_for_group
def patched_eval(self, base, curr_frame, baseline):
    if base == "O":
        d_pw = getattr(curr_frame, 'd_pinky_wrist_norm', 1.0)
        base_pw = getattr(baseline, 'd_pinky_wrist_norm', 1.0)
        if (curr_frame.d_index_middle_norm >= 0.20 and 
            curr_frame.index_pip_angle <= 110.0 and 
            curr_frame.index_curl >= 0.35 and 
            d_pw <= 0.85):
            return "Ô", f"Tạo nón trỏ-giữa (d_IM={curr_frame.d_index_middle_norm:.2f}>=0.20, PIP={curr_frame.index_pip_angle:.1f}<=110.0)"
        
        d_pw_delta = d_pw - base_pw
        if (curr_frame.d_index_middle_norm <= 0.18 and 
            d_pw >= 1.25 and 
            (d_pw_delta >= 0.20 or curr_frame.d_pinky_index_norm >= self.o_horn_dist_threshold)):
            return "Ơ", f"Bật râu ngón út (d_PW={d_pw:.2f}>=1.25, delta={d_pw_delta:+.2f})"
        return None
    return orig_eval(self, base, curr_frame, baseline)
dsm.DiacriticStateMachine._evaluate_diacritic_for_group = patched_eval

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
session_id = "test_o_trace"
rec.reset()
rec.reset_diacritic_fsm(session_id)

for item in seq:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec')
    lms = item.get('landmarks')
    if idx < 30 or idx > 45:
        continue
    res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=session_id, frame_idx=idx, timestamp=t, expected_sign='Ô')
    fsm_key = (str(session_id), str(session_id), 'letters')
    fsm = rec.diacritic_fsms.get(fsm_key)
    fsm_st = fsm.state if fsm else None
    fsm_base = fsm.current_base if fsm else None
    diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
    print(f"Frame {idx:02d}: sign={res['sign']:<4} src={res['source']:<18} fsm_st={fsm_st:<20} fsm_base={fsm_base} diac={diag.get('diacritic_detected')}")

