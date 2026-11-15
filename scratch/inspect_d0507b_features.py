import json
import sys
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine

with open('backend/data/alphabet_v2/motion_sequences/Ô.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

seq = data.get('sequence', [])
fsm = DiacriticStateMachine(circumflex_lift_threshold=0.020, circumflex_vy_threshold=0.05)

print(f"{'Idx':<5} | {'Time':<6} | {'Wrist Y':<8} | {'Lift':<8} | {'Vy':<8} | {'d_PI':<8} | {'Eval Match'}")
print("-" * 75)

for item in seq:
    lms = item.get('landmarks')
    t = item.get('timestamp_sec', 0.0)
    idx = item.get('frame_index', 0)
    if not lms:
        continue
    
    frame = fsm._extract_frame_features(lms, t)
    if fsm.baseline is None and idx >= 17:
        # simulate baseline at frame 17
        fsm.baseline = fsm._extract_frame_features(lms, t)
        # BaselineSnapshot
        from backend.vision.diacritic_state_machine import BaselineSnapshot
        fsm.baseline = BaselineSnapshot(
            base_label="O",
            start_time=t,
            stable_time=t,
            wrist_y=frame.wrist_y,
            palm_center_y=frame.palm_center_y,
            palm_scale=frame.palm_scale,
            d_index_middle_norm=frame.d_index_middle_norm,
            d_pinky_index_norm=frame.d_pinky_index_norm,
            pinky_curl=frame.pinky_curl,
            index_pip_angle=frame.index_pip_angle,
            index_curl=frame.index_curl
        )
    
    lift = (fsm.baseline.wrist_y - frame.wrist_y) if fsm.baseline else 0.0
    match = fsm._evaluate_diacritic_for_group("O", frame, fsm.baseline) if fsm.baseline else None
    
    if idx in range(14, 40):
        print(f"{idx:<5} | {t:<6.3f} | {frame.wrist_y:<8.4f} | {lift:<8.4f} | {frame.vy:<8.4f} | {frame.d_pinky_index_norm:<8.4f} | {match}")

