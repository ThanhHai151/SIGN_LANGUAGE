import sys
sys.path.insert(0, '.')
import json
from scratch.task1_3_2_production_replay import fixtures, simulate_production_pipeline

print("=== INSPECTING INDIVIDUAL PREDICTIONS ON FIXTURES ===")
for letter in ['P', 'K', 'Q', 'D']:
    print(f"\n--- {letter} True Stable Frames ---")
    for item in fixtures[letter]['stable']:
        res = simulate_production_pipeline(item['landmarks'], features=item['features'])
        print(f"  Frame {item['f_idx']:2d}: pred={res['final_sign']:7s} | status={res['status']:8s} | matches={[m[0] for m in res['matches']]}")

    if fixtures[letter]['exit']:
        print(f"\n--- {letter} Exit Frames ---")
        for item in fixtures[letter]['exit']:
            res = simulate_production_pipeline(item['landmarks'], features=item['features'])
            print(f"  Frame {item['f_idx']:2d}: pred={res['final_sign']:7s} | status={res['status']:8s} | matches={[m[0] for m in res['matches']]}")

