import sys
sys.path.insert(0, '.')
import json
from scratch.task1_3_1_eval_candidate import evaluate_p_candidate_conditions

log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
trials = {}
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            tid = r['trial_id']
            if tid not in trials: trials[tid] = []
            feat = r['geometric']['features']
            matched = evaluate_p_candidate_conditions(feat)['matched']
            trials[tid].append(matched)

print('=== TRIAL-BY-TRIAL BREAKDOWN FOR WEBCAM P ===\n')
pass_trials = 0
for tid, flags in sorted(trials.items()):
    p_cnt = sum(1 for x in flags if x)
    tot = len(flags)
    acc = p_cnt / tot * 100
    is_trial_pass = (p_cnt >= tot / 2.0) and (p_cnt > 0)
    if is_trial_pass: pass_trials += 1
    status_str = "PASS" if is_trial_pass else "FAIL"
    print(f'{tid:35s}: P = {p_cnt:2d}/{tot:2d} ({acc:5.1f}%) | Majority Vote = {status_str}')

print(f'\nTrial Accuracy: {pass_trials}/{len(trials)} ({pass_trials/len(trials)*100:.1f}%)')

