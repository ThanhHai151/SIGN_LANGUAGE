import json
import glob

print("=== CHECKING FIXTURE COORDINATE CONVENTIONS ===")
for letter in ['K', 'P', 'Q', 'D']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for s in data['samples']:
        if s.get('landmarks'):
            lms = s['landmarks']
            f_idx = s.get('frame_index') or s.get('frame_idx')
            phase = s.get('phase')
            print(f"{letter} Frame {f_idx} (phase={phase}):")
            print(f"   LM0 (wrist): x={lms[0][0]:.4f}, y={lms[0][1]:.4f}, z={lms[0][2]:.4f}")
            print(f"   LM8 (index): x={lms[8][0]:.4f}, y={lms[8][1]:.4f}, z={lms[8][2]:.4f}")
            print(f"   Y_tip - Y_wrist = {lms[8][1] - lms[0][1]:.4f}")
            break

print("\n=== CHECKING WEBCAM JSONL CONVENTIONS ===")
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep01' in r.get('trial_id', ''):
            feat = r['geometric']['features']
            print("Webcam P rep01 frame:", r.get('frame_idx'))
            print("   idx_tip_y in log:", feat.get('idx_tip_y'))
            print("   mid_tip_y in log:", feat.get('mid_tip_y'))
            print("   idx_dy in log:", feat.get('idx_dy'))
            print("   d_iw in log:", feat.get('d_IW'))
            break

