import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# Load X frame 144
with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

x144 = [s for s in x_data['samples'] if (s.get('frame_index') or s.get('frame_idx')) == 144][0]
feat_x144 = clf.extract_features(x144['landmarks'])

# Load Webcam P 90 frames
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
webcam_p = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            webcam_p.append(r['geometric']['features'])

print("=== COMPARISON: X FRAME 144 VS WEBCAM P (90 FRAMES) ===")
print(f"{'Feature':18s} | {'X frame 144':12s} | {'Webcam P Min':14s} | {'Webcam P Max':14s} | {'Webcam P Mean':14s}")
print("-" * 80)

for k in sorted(feat_x144.keys()):
    vx = feat_x144[k]
    if isinstance(vx, (int, float)):
        v_p = [f[k] for f in webcam_p if isinstance(f.get(k), (int, float))]
        if v_p:
            min_p, max_p, mean_p = min(v_p), max(v_p), sum(v_p) / len(v_p)
            diff_mark = " <-- OUTSIDE RANGE" if (vx < min_p or vx > max_p) else ""
            print(f"{k:18s} | {vx:12.4f} | {min_p:14.4f} | {max_p:14.4f} | {mean_p:14.4f}{diff_mark}")
        else:
            print(f"{k:18s} | {vx:12.4f} | {'N/A':14s} | {'N/A':14s} | {'N/A':14s}")
    else:
        print(f"{k:18s} | {str(vx):12s} | {'N/A':14s} | {'N/A':14s} | {'N/A':14s}")

