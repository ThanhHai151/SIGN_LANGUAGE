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
                # determine attempt
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

print(f"Total trials collected: {len(trials)}")

# Let's inspect trials where NN == 0, where C > 0 and I == 0, where C == 0 and I > 0, and where both C > 0 and I > 0
all_zero_nn = []
c_only = []
i_only = []
both = []

for tid, t in trials.items():
    nn = t['correct'] + t['incorrect']
    if nn == 0:
        all_zero_nn.append(t)
    elif t['correct'] > 0 and t['incorrect'] == 0:
        c_only.append(t)
    elif t['correct'] == 0 and t['incorrect'] > 0:
        i_only.append(t)
    else:
        both.append(t)

print(f"Trials with NN == 0 (100% neutral): {len(all_zero_nn)}")
print(f"Trials with C > 0 and I == 0 (pure correct when active): {len(c_only)}")
print(f"Trials with C == 0 and I > 0 (pure incorrect when active): {len(i_only)}")
print(f"Trials with both C > 0 and I > 0 (mixed non-neutral): {len(both)}")

print("\n--- TRIALS WITH BOTH C > 0 and I > 0 ---")
for t in both:
    nn = t['correct'] + t['incorrect']
    print(f"{t['trial_id']}: exp={t['expected_sign']}, total={t['frames']}, C={t['correct']}, I={t['incorrect']}, U={t['neutral']}, NN={nn}, C/NN={t['correct']/nn*100:.1f}%, conf={t['confirmed']}, incor_signs={dict(t['incorrect_signs'])}")

print("\n--- TRIALS WITH C > 0 and I == 0 BUT FEW FRAMES (C <= 3) ---")
for t in c_only:
    if t['correct'] <= 3:
        print(f"{t['trial_id']}: exp={t['expected_sign']}, total={t['frames']}, C={t['correct']}, U={t['neutral']}, conf={t['confirmed']}")

