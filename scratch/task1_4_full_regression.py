import sys
sys.path.insert(0, '.')
import os
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# 29 letters of VSL
ALL_29_LETTERS = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 
    'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Ô', 'Ơ', 'P', 
    'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
]

fixtures_dir = 'backend/data/alphabet_v2/static_landmarks'

print("=== REGRESSION AUDIT ACROSS ALL 29 VSL LETTERS ===")
header = f"{'Letter':6s} | {'Fixture Status':25s} | {'Total Frames':12s} | {'Pred P (FP)':12s} | {'Match Expected':14s} | {'Neutral':9s} | {'Conflict':9s} | {'Other Letter':12s}"
print(header)
print("-" * len(header))

summary_table = []

for let in ALL_29_LETTERS:
    fixture_file = f"{let}.json"
    fixture_path = os.path.join(fixtures_dir, fixture_file)
    
    if not os.path.exists(fixture_path):
        row = {
            'letter': let,
            'status': 'NOT TESTED (No fixture)',
            'total': 0,
            'p_count': 0,
            'expected_count': 0,
            'neutral_count': 0,
            'conflict_count': 0,
            'other_count': 0,
            'details': {}
        }
        summary_table.append(row)
        print(f"{let:6s} | {'NOT TESTED (No fixture)':25s} | {'0':12s} | {'0':12s} | {'0':14s} | {'0':9s} | {'0':9s} | {'0':12s}")
        continue
        
    with open(fixture_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    samples = data.get('samples', [])
    valid_samples = [s for s in samples if s.get('landmarks')]
    
    tot = len(valid_samples)
    p_cnt = 0
    exp_cnt = 0
    neu_cnt = 0
    conf_cnt = 0
    oth_cnt = 0
    pred_dist = {}
    
    for s in valid_samples:
        lms = s['landmarks']
        res = clf.classify_detailed(lms)
        pred = res['letter']
        st = res['status']
        
        pred_dist[pred] = pred_dist.get(pred, 0) + 1
        
        if pred == 'P' and let != 'P':
            p_cnt += 1
        if pred == let:
            exp_cnt += 1
        if pred == 'UNKNOWN' or st in ('NO_MATCH', 'INVALID_INPUT'):
            neu_cnt += 1
        if st == 'CONFLICT':
            conf_cnt += 1
        if pred not in (let, 'UNKNOWN', 'P'):
            oth_cnt += 1
            
    status_str = f"Found ({tot} frames)"
    row = {
        'letter': let,
        'status': status_str,
        'total': tot,
        'p_count': p_cnt,
        'expected_count': exp_cnt,
        'neutral_count': neu_cnt,
        'conflict_count': conf_cnt,
        'other_count': oth_cnt,
        'details': pred_dist
    }
    summary_table.append(row)
    print(f"{let:6s} | {status_str:25s} | {tot:12d} | {p_cnt:12d} | {exp_cnt:14d} | {neu_cnt:9d} | {conf_cnt:9d} | {oth_cnt:12d}")

# Save detailed results for report
with open('scratch/task1_4_regression_results.json', 'w', encoding='utf-8') as f:
    json.dump(summary_table, f, indent=2, ensure_ascii=False)

