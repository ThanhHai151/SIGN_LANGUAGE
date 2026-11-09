import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
from scratch.task1_3_1_eval_candidate import evaluate_p_candidate_conditions

clf = VSLAlphabetClassifier()

# Simulate classification with 2-branch P
def simulate_classifier_detailed(landmarks):
    # Standard classification
    res = clf.classify_detailed(landmarks)
    
    # Check candidate P-down
    feat = clf.extract_features(landmarks)
    p_down_res = evaluate_p_candidate_conditions(feat)
    
    # If standard matched P, it's P (North)
    if res['status'] == 'MATCH' and res['letter'] == 'P':
        return 'P'
    
    # If candidate P-down matched:
    if p_down_res and p_down_res['matched']:
        # If standard also matched something else, check conflict
        # P removes K, V, D, L in standard logic
        return 'P'
        
    if res['status'] == 'MATCH':
        return res['letter']
    else:
        return 'NEUTRAL'

# Load datasets
datasets = {
    'P_down (Webcam 90 frames)': [],
    'P_D0509B (20 stable frames)': [],
    'K (17 stable frames)': [],
    'Q (19 stable frames)': [],
    'D (20 stable frames)': []
}

# 1. Webcam P
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            # Since landmarks aren't directly in log, use features for P-down simulation
            feat = r['geometric']['features']
            p_res = evaluate_p_candidate_conditions(feat)
            # Old rule was 100% NO_MATCH
            pred = 'P' if (p_res and p_res['matched']) else 'NEUTRAL'
            datasets['P_down (Webcam 90 frames)'].append(pred)

# 2. Fixtures
for letter, dname in [('P', 'P_D0509B (20 stable frames)'), ('K', 'K (17 stable frames)'), ('Q', 'Q (19 stable frames)'), ('D', 'D (20 stable frames)')]:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for s in data.get('samples', []):
        if s.get('phase') == 'stable_pose' and s.get('landmarks'):
            pred = simulate_classifier_detailed(s['landmarks'])
            datasets[dname].append(pred)

print('=== EMPIRICAL CONFUSION MATRIX ===\n')
header = f"{'Actual Pose':30s} | {'Total':5s} | {'P':5s} | {'K':5s} | {'Q':5s} | {'D':5s} | {'Neutral':7s} | {'Other':6s}"
print(header)
print('-' * len(header))

for dname, preds in datasets.items():
    total = len(preds)
    cnt_p = preds.count('P')
    cnt_k = preds.count('K')
    cnt_q = preds.count('Q')
    cnt_d = preds.count('D')
    cnt_neutral = preds.count('NEUTRAL')
    cnt_other = total - (cnt_p + cnt_k + cnt_q + cnt_d + cnt_neutral)
    print(f"{dname:30s} | {total:5d} | {cnt_p:5d} | {cnt_k:5d} | {cnt_q:5d} | {cnt_d:5d} | {cnt_neutral:7d} | {cnt_other:6d}")

