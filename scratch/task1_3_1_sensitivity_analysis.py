import sys
sys.path.insert(0, '.')
import json
import numpy as np

# Load webcam P features and fixture features
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
webcam_p = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            webcam_p.append(r['geometric']['features'])

from backend.vision.alphabet_classifier import VSLAlphabetClassifier
clf = VSLAlphabetClassifier()

def get_fixture_feats(letter):
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [clf.extract_features(s['landmarks']) for s in data.get('samples', []) if s.get('phase') == 'stable_pose' and s.get('landmarks')]

k_stable = get_fixture_feats('K')
q_stable = get_fixture_feats('Q')
d_stable = get_fixture_feats('D')

print('=== SENSITIVITY ANALYSIS ===\n')

# Check sensitivity of individual thresholds
experiments = [
    ('idx_tip_y', [0.40, 0.50, 0.60, 0.70, 0.80, 0.90], '>=', 'c3_idx_tip_y_ge_0_60'),
    ('idx_dy', [0.40, 0.50, 0.60, 0.70, 0.80], '>=', 'c4_idx_dy_ge_0_60'),
    ('d_MW', [1.10, 1.25, 1.35, 1.50, 1.65], '>=', 'c6_d_mw_ge_1_35'),
    ('mid_straight', [0.70, 0.75, 0.80, 0.85, 0.90], '>=', 'c7_mid_straight_ge_0_80'),
    ('d_TM', [0.75, 0.85, 0.90, 0.95, 1.05], '<=', 'c9_d_tm_le_0_90'),
    ('d_RW', [1.40, 1.50, 1.60, 1.80, 2.00, 2.15], '<=', 'c13_d_rw_le_1_60'),
    ('d_PW', [1.30, 1.40, 1.50, 1.65, 1.80], '<=', 'c14_d_pw_le_1_50'),
]

for feat, thresholds, op, cond_name in experiments:
    print(f"--- Threshold variations for {feat} (Current baseline op: {op}) ---")
    for th in thresholds:
        if op == '>=':
            p_pass = sum(1 for f in webcam_p if f.get(feat, 0) >= th)
            k_pass = sum(1 for f in k_stable if f.get(feat, 0) >= th)
            q_pass = sum(1 for f in q_stable if f.get(feat, 0) >= th)
            d_pass = sum(1 for f in d_stable if f.get(feat, 0) >= th)
        else:
            p_pass = sum(1 for f in webcam_p if f.get(feat, 99) <= th)
            k_pass = sum(1 for f in k_stable if f.get(feat, 99) <= th)
            q_pass = sum(1 for f in q_stable if f.get(feat, 99) <= th)
            d_pass = sum(1 for f in d_stable if f.get(feat, 99) <= th)
        print(f"  Th = {th:5.2f} | P_pass: {p_pass:2d}/90 ({p_pass/90*100:5.1f}%) | K_pass: {k_pass:2d}/17 | Q_pass: {q_pass:2d}/19 | D_pass: {d_pass:2d}/20")
    print()

