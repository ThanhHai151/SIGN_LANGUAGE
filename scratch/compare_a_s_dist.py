import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clf = AlphabetClassifier()

a_lms = npz['landmarks'][npz['y'] == 0]
meta = json.loads(str(npz['metadata_json']))
a_meta = [m for m in meta if m.get('label') == 'A']

print(f"Total A samples: {len(a_lms)}")
a_ang_tp = []
a_th_lat = []
a_tm_dist = []
for i in range(len(a_lms)):
    f = clf.extract_features(a_lms[i].tolist())
    if f is None: continue
    a_ang_tp.append(f['ang_TP'])
    a_th_lat.append(f['thumb_lat'])
    a_tm_dist.append(f['thumb_mid_dist'])

print(f"A ang_TP: min={min(a_ang_tp):.1f}, mean={np.mean(a_ang_tp):.1f}, max={max(a_ang_tp):.1f}")
print(f"A thumb_lat: min={min(a_th_lat):.2f}, mean={np.mean(a_th_lat):.2f}, max={max(a_th_lat):.2f}")
print(f"A thumb_mid_dist: min={min(a_tm_dist):.2f}, mean={np.mean(a_tm_dist):.2f}, max={max(a_tm_dist):.2f}")

# Check S samples in dataset
s_lms = npz['landmarks'][npz['y'] == 16]
s_ang_tp = []
s_th_lat = []
s_tm_dist = []
for i in range(len(s_lms)):
    f = clf.extract_features(s_lms[i].tolist())
    if f is None: continue
    s_ang_tp.append(f['ang_TP'])
    s_th_lat.append(f['thumb_lat'])
    s_tm_dist.append(f['thumb_mid_dist'])

print(f"\nS ang_TP: min={min(s_ang_tp):.1f}, mean={np.mean(s_ang_tp):.1f}, max={max(s_ang_tp):.1f}")
print(f"S thumb_lat: min={min(s_th_lat):.2f}, mean={np.mean(s_th_lat):.2f}, max={max(s_th_lat):.2f}")
print(f"S thumb_mid_dist: min={min(s_tm_dist):.2f}, mean={np.mean(s_tm_dist):.2f}, max={max(s_tm_dist):.2f}")

# Check User's webcam S
webcam_file = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(webcam_file, 'r', encoding='utf-8') as f:
    webcam_lines = [json.loads(l.strip()) for l in f if l.strip()]

webcam_s = [
    line['geometric']['features'] 
    for line in webcam_lines 
    if line.get('expected_sign') == 'S' and 'features' in line.get('geometric', {})
]
w_ang_tp = [f['ang_TP'] for f in webcam_s]
w_th_lat = [f['thumb_lat'] for f in webcam_s]
w_tm_dist = [f['thumb_mid_dist'] for f in webcam_s]
print(f"\nWebcam S ang_TP: min={min(w_ang_tp):.1f}, mean={np.mean(w_ang_tp):.1f}, max={max(w_ang_tp):.1f}")
print(f"Webcam S thumb_lat: min={min(w_th_lat):.2f}, mean={np.mean(w_th_lat):.2f}, max={max(w_th_lat):.2f}")
print(f"Webcam S thumb_mid_dist: min={min(w_tm_dist):.2f}, mean={np.mean(w_tm_dist):.2f}, max={max(w_tm_dist):.2f}")

