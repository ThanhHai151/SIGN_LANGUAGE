import sys
sys.path.insert(0, '.')
import json
from scratch.task1_3_1_eval_candidate import evaluate_p_candidate_conditions

log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            feat = r['geometric']['features']
            res = evaluate_p_candidate_conditions(feat)
            if not res['matched']:
                trial = r['trial_id']
                fidx = r['frame_idx']
                print(f"{trial}_f{fidx}: failed={res['failed']}, d_RW={feat.get('d_RW')}, d_PW={feat.get('d_PW')}")

