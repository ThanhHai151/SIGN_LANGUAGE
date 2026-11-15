import json
import sys
sys.path.insert(0, ".")

with open('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

from backend.vision.diacritic_state_machine import DiacriticStateMachine
from backend.vision.feature_extractor import calculate_distance

fsm = DiacriticStateMachine()

print(f"{'Idx':<5} | {'pinky_curl':<10} | {'index_curl':<10} | {'d(8,12)':<10} | {'d(20,8)':<10} | {'d(20,palm)':<10} | {'index_pip':<10}")
print("-" * 80)

for item in data['sequence']:
    idx = item.get('frame_index', 0)
    lms = item.get('landmarks')
    t = item.get('timestamp_sec', 0.0)
    if not lms or idx not in [20, 25, 30, 33, 37, 40, 45, 50, 55]:
        continue
    mf = fsm._extract_frame_features(lms, t)
    shifted = mf.shifted_landmarks
    d_pinky_wrist = calculate_distance(shifted[20], shifted[0]) / mf.palm_scale
    print(f"{idx:<5} | {mf.pinky_curl:<10.3f} | {mf.index_curl:<10.3f} | {mf.d_index_middle_norm:<10.3f} | {mf.d_pinky_index_norm:<10.3f} | {d_pinky_wrist:<10.3f} | {mf.index_pip_angle:<10.1f}")

