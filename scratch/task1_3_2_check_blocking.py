import sys
sys.path.insert(0, '.')
import json
from scratch.task1_3_2_production_replay import fixtures, evaluate_p_candidate_conditions

print("=== BLOCKING CONDITIONS FOR K, Q, D FIXTURES ===")
for letter in ['K', 'Q', 'D']:
    print(f"\n--- Fixture {letter} ---")
    all_f = fixtures[letter]['all']
    blocking_counts = {}
    for item in all_f:
        res = evaluate_p_candidate_conditions(item['features'])
        if not res['matched']:
            for fcond in res['failed']:
                blocking_counts[fcond] = blocking_counts.get(fcond, 0) + 1
        else:
            print(f"ALERT: Frame {item['id']} MATCHED P!")
    print(f"Total frames: {len(all_f)}")
    print("Conditions that blocked P:")
    for c, cnt in sorted(blocking_counts.items(), key=lambda x: -x[1]):
        print(f"   {c:30s}: failed in {cnt:2d}/{len(all_f)} frames ({cnt/len(all_f)*100:.1f}%)")

