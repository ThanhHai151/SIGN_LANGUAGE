import json
import sys
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine
from backend.vision.feature_extractor import calculate_distance, calculate_angle

fsm = DiacriticStateMachine()

def analyze(filename, name):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n==================== {name} ({filename}) ====================")
    print(f"{'Idx':<4} | {'t':<5} | {'d(8,12)':<7} | {'d(20,0)':<7} | {'d(20,8)':<7} | {'idx_curl':<8} | {'mid_curl':<8} | {'pky_curl':<8} | {'idx_pip':<7} | {'mid_pip':<7}")
    print("-" * 88)
    for item in data['sequence']:
        idx = item.get('frame_index', 0)
        t = item.get('timestamp_sec', 0.0)
        lms = item.get('landmarks')
        if not lms:
            continue
        mf = fsm._extract_frame_features(lms, t)
        sh = mf.shifted_landmarks
        ps = mf.palm_scale
        d_pw = calculate_distance(sh[20], sh[0]) / ps
        # middle pip angle: 9, 10, 11
        mid_pip = calculate_angle(sh[9], sh[10], sh[11])
        # ring curl, mid curl from fstates
        fstates = mf.finger_states
        mid_curl = fstates.get('middle_curl', 0.0)
        pky_curl = fstates.get('pinky_curl', 0.0)
        idx_curl = fstates.get('index_curl', 0.0)
        
        # print selected frames
        if idx % 2 == 0 and idx >= 10 and idx <= 60:
            print(f"{idx:<4} | {t:<5.2f} | {mf.d_index_middle_norm:<7.3f} | {d_pw:<7.3f} | {mf.d_pinky_index_norm:<7.3f} | {idx_curl:<8.3f} | {mid_curl:<8.3f} | {pky_curl:<8.3f} | {mf.index_pip_angle:<7.1f} | {mid_pip:<7.1f}")

analyze('backend/data/alphabet_v2/motion_sequences/Ô.json', 'Ô (D0507B)')
analyze('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'Ơ (D0508N)')

