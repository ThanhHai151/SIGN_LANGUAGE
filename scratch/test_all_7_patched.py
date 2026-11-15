import sys
sys.path.insert(0, ".")
from scratch.test_o_patch_simulation2 import rec
import json

VIDEOS = {
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A'),
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A'),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E'),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D'),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O'),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O'),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U'),
}

print("\n=== TESTING ALL 7 LETTERS WITH PATCHED FSM ===")
for target, (path, exp_base) in VIDEOS.items():
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    seq = d['sequence']
    session_id = f"test_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(session_id)
    tokens = []
    for item in seq:
        idx = item.get('frame_index')
        t = item.get('timestamp_sec')
        lms = item.get('landmarks')
        res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=session_id, frame_idx=idx, timestamp=t, expected_sign=target)
        tok = res.get('new_token')
        if tok:
            tokens.append((idx, round(t, 2), tok, res.get('source')))
    print(f"{target} (base {exp_base}): tokens={tokens}")
