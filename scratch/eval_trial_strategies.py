import json
import collections

trials = collections.OrderedDict()

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        tid = r.get('trial_id')
        exp = r.get('expected_sign')
        if tid and exp:
            if tid not in trials:
                att = 'att02' if '_att02' in tid else 'att01'
                trials[tid] = {
                    'trial_id': tid,
                    'expected_sign': exp,
                    'attempt': att,
                    'frames': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'neutral': 0,
                    'confirmed': 0,
                    'incorrect_signs': collections.Counter()
                }
            t = trials[tid]
            t['frames'] += 1
            fs = r.get('final_sign')
            if fs == exp:
                t['correct'] += 1
            elif fs == 'NEUTRAL':
                t['neutral'] += 1
            else:
                t['incorrect'] += 1
                t['incorrect_signs'][fs] += 1
            if r.get('confirmed'):
                t['confirmed'] += 1

def adjudicate(t):
    c = t['correct']
    i = t['incorrect']
    nn = c + i
    if nn == 0:
        return 'NEUTRAL'
    if c > i:
        return 'CORRECT'
    elif i > c:
        return 'INCORRECT'
    else:
        return 'MIXED'

# Strategy 1: All 224 trials (220 att01 + 4 att02)
all_counts = collections.Counter()
att01_counts = collections.Counter()
att02_counts = collections.Counter()

for tid, t in trials.items():
    res = adjudicate(t)
    t['adjudicated_result'] = res
    all_counts[res] += 1
    if t['attempt'] == 'att01':
        att01_counts[res] += 1
    else:
        att02_counts[res] += 1

# Strategy 2: Resolved 220 trials (att02 replaces att01 for A_rep03, K_rep01, P_rep01, S_rep01)
resolved_trials = {}
for tid, t in trials.items():
    # base key: e.g. A_rep03
    parts = tid.split('_')
    # parts: ['pilot', '20261107', '2141', 'LETTER', 'repXX', 'attYY']
    letter = parts[3]
    rep = parts[4]
    base_key = f"{letter}_{rep}"
    if base_key not in resolved_trials or t['attempt'] == 'att02':
        resolved_trials[base_key] = t

resolved_counts = collections.Counter([t['adjudicated_result'] for t in resolved_trials.values()])

print("=== ALL 224 TRIALS ===")
print(dict(all_counts))
print("=== ONLY 220 ATT01 TRIALS ===")
print(dict(att01_counts))
print("=== ONLY 4 ATT02 TRIALS ===")
print(dict(att02_counts))
print("=== RESOLVED 220 TRIALS (att02 replaces att01) ===")
print(dict(resolved_counts))

