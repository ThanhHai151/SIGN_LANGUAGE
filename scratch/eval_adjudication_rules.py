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

def adjudicate(t, min_nn=1):
    c = t['correct']
    i = t['incorrect']
    nn = c + i
    if nn < min_nn:
        return 'NEUTRAL'
    if c > i and (c / nn) > 0.5:
        return 'CORRECT'
    elif i > c and (i / nn) > 0.5:
        return 'INCORRECT'
    else:
        return 'MIXED'

for min_nn in [1, 3, 5]:
    counts_all = collections.Counter()
    counts_att01 = collections.Counter()
    counts_att02 = collections.Counter()
    for tid, t in trials.items():
        res = adjudicate(t, min_nn=min_nn)
        counts_all[res] += 1
        if t['attempt'] == 'att01':
            counts_att01[res] += 1
        else:
            counts_att02[res] += 1
    print(f"=== Min Non-Neutral Frames = {min_nn} ===")
    print(f"ALL 224: {dict(counts_all)}")
    print(f"att01 (220): {dict(counts_att01)}")
    print(f"att02 (4): {dict(counts_att02)}")

