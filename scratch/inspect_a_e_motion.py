import json
import sys
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine

fsm = DiacriticStateMachine()

for target, filename in [('Â', 'backend/data/alphabet_v2/motion_sequences/Â.json'),
                         ('Ê', 'backend/data/alphabet_v2/motion_sequences/Ê.json'),
                         ('Ă', 'backend/data/alphabet_v2/motion_sequences/Ă.json')]:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n=== Inspection for {target} ===")
    seq = data['sequence']
    base_wrist_y = None
    max_lift = -999.0
    for item in seq:
        idx = item.get('frame_index', 0)
        lms = item.get('landmarks')
        t = item.get('timestamp_sec', 0.0)
        if not lms:
            continue
        mf = fsm._extract_frame_features(lms, t)
        if base_wrist_y is None and idx > 15:
            base_wrist_y = mf.wrist_y
        if base_wrist_y is not None:
            lift = base_wrist_y - mf.wrist_y
            if lift > max_lift:
                max_lift = lift
        if idx in [20, 30, 40, 50, 60]:
            print(f"  Frame {idx:02d}: wrist_y={mf.wrist_y:.3f}, vy={mf.vy:.3f}, d_im={mf.d_index_middle_norm:.3f}, d_pi={mf.d_pinky_index_norm:.3f}")
    print(f"  Max wrist lift: {max_lift:.4f}")

