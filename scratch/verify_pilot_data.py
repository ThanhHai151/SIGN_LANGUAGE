import json
import math

with open("backend/tests/fixtures/pilot_real_letter_a_landmarks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data['samples'])}")
for idx, s in enumerate(data['samples']):
    status = s['quality_status']
    lms = s['landmarks']
    print(f"[{idx}] frame={s['frame_idx']} ts={s['timestamp_sec']}s conf={s['confidence']} status={status}")
    if status == 'ok':
        assert len(lms) == 21, f"Frame {s['frame_idx']} does not have 21 points"
        for p_idx, p in enumerate(lms):
            assert len(p) == 3, f"Point {p_idx} does not have 3 coords"
            x, y, z = p
            assert not math.isnan(x) and not math.isinf(x), f"x is NaN/Inf at {p_idx}"
            assert not math.isnan(y) and not math.isinf(y), f"y is NaN/Inf at {p_idx}"
            assert not math.isnan(z) and not math.isinf(z), f"z is NaN/Inf at {p_idx}"
    elif status == 'no_hand_detected':
        assert lms is None, f"Landmarks should be None for no_hand_detected"

print("Pilot verification: ALL CHECKS PASSED.")

