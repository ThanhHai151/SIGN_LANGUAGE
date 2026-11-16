import json
import os
import sys

sys.path.insert(0, ".")
from backend.vision.diacritic_state_machine import DiacriticStateMachine

seq_files = {
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A'),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E'),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O'),
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A'),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D'),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O'),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U'),
}

# Pre-load sequences
loaded_seqs = {}
for letter, (fpath, expected_base) in seq_files.items():
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    loaded_seqs[letter] = {
        'seq': data.get('sequence', []),
        'expected_base': expected_base,
        'video_id': data.get('video_id', '')
    }

lift_candidates = [0.035, 0.028, 0.025, 0.020]
search_sec_candidates = [0.65, 0.80, 0.85, 1.00]

print("=" * 110)
print(f"{'Lift':<6} | {'SearchSec':<10} | {'Â':<14} | {'Ê':<14} | {'Ô':<14} | {'Ă':<10} | {'Đ':<10} | {'Ơ':<10} | {'Ư':<10}")
print("=" * 110)

results_grid = []

for lift in lift_candidates:
    for search_sec in search_sec_candidates:
        row = {'lift': lift, 'search_sec': search_sec, 'replays': {}}
        row_str = f"{lift:<6.3f} | {search_sec:<10.2f} | "

        for target in ['Â', 'Ê', 'Ô', 'Ă', 'Đ', 'Ơ', 'Ư']:
            seq = loaded_seqs[target]['seq']
            exp_base = loaded_seqs[target]['expected_base']

            fsm = DiacriticStateMachine(
                circumflex_lift_threshold=lift,
                max_diacritic_search_sec=search_sec
            )

            confirmed = False
            first_frame = None
            first_time = None
            detected_diac = None

            for item in seq:
                lms = item.get('landmarks')
                t = item.get('timestamp_sec', 0.0)
                idx = item.get('frame_index', 0)

                # Feed expected base when hand is detected
                if lms:
                    res = fsm.update(
                        base_label=exp_base,
                        base_confidence=0.90,
                        landmarks_raw=lms,
                        timestamp=t
                    )
                else:
                    res = fsm.update_transient(landmarks_raw=None, timestamp=t)

                if res.get('state') == 'DIACRITIC_CONFIRMED' and not confirmed:
                    confirmed = True
                    first_frame = idx
                    first_time = t
                    detected_diac = res.get('diacritic_detected')

            status_str = f"f{first_frame} ({detected_diac})" if confirmed else "FAIL"
            row['replays'][target] = {
                'confirmed': confirmed,
                'frame': first_frame,
                'time': first_time,
                'detected': detected_diac
            }
            if target in ['Â', 'Ê', 'Ô']:
                row_str += f"{status_str:<14} | "
            else:
                row_str += f"{status_str:<10} | "

        print(row_str)
        results_grid.append(row)
