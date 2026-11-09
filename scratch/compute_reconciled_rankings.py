import json
import collections

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if json.loads(line).get('trial_id') and json.loads(line).get('expected_sign')]

# Group by letter and trial
letter_stats = {}
trials = collections.OrderedDict()

for r in records:
    tid = r['trial_id']
    exp = r['expected_sign']
    att = 'att02' if '_att02' in tid else 'att01'
    
    if exp not in letter_stats:
        letter_stats[exp] = {
            'letter': exp,
            'total_frames': 0,
            'correct_frames': 0,
            'incorrect_frames': 0,
            'neutral_frames': 0,
            'trials_all': set(),
            'trials_att01': set(),
            'trials_att02': set(),
            'confirmed_frames': 0,
        }
    ls = letter_stats[exp]
    ls['total_frames'] += 1
    ls['trials_all'].add(tid)
    if att == 'att01':
        ls['trials_att01'].add(tid)
    else:
        ls['trials_att02'].add(tid)
        
    fs = r['final_sign']
    if fs == exp:
        ls['correct_frames'] += 1
    elif fs == 'NEUTRAL':
        ls['neutral_frames'] += 1
    else:
        ls['incorrect_frames'] += 1
        
    if r.get('confirmed'):
        ls['confirmed_frames'] += 1
        
    if tid not in trials:
        trials[tid] = {
            'trial_id': tid,
            'expected_sign': exp,
            'attempt': att,
            'total_frames': 0,
            'correct_frames': 0,
            'incorrect_frames': 0,
            'neutral_frames': 0,
            'confirmed_count': 0
        }
    t = trials[tid]
    t['total_frames'] += 1
    if fs == exp:
        t['correct_frames'] += 1
    elif fs == 'NEUTRAL':
        t['neutral_frames'] += 1
    else:
        t['incorrect_frames'] += 1
    if r.get('confirmed'):
        t['confirmed_count'] += 1

# Adjudicate trials
for tid, t in trials.items():
    c = t['correct_frames']
    i = t['incorrect_frames']
    nn = c + i
    if nn == 0:
        t['adjudicated_result'] = 'NEUTRAL'
    elif c > i:
        t['adjudicated_result'] = 'CORRECT'
    elif i > c:
        t['adjudicated_result'] = 'INCORRECT'
    else:
        t['adjudicated_result'] = 'MIXED'
    t['non_neutral_frames'] = nn
    t['correct_rate_among_non_neutral'] = round(c / nn * 100, 2) if nn > 0 else 0.0

# Per-letter summary
summary = []
for exp, ls in letter_stats.items():
    exp_trials = [t for t in trials.values() if t['expected_sign'] == exp]
    total_tr_all = len(exp_trials)
    cor_tr_all = sum(1 for t in exp_trials if t['adjudicated_result'] == 'CORRECT')
    tr_acc_all = round(cor_tr_all / total_tr_all * 100, 2)
    
    # Resolved 10 trials
    resolved = {}
    for t in exp_trials:
        parts = t['trial_id'].split('_')
        base_rep = f"{parts[3]}_{parts[4]}"
        if base_rep not in resolved or t['attempt'] == 'att02':
            resolved[base_rep] = t
    total_tr_res = len(resolved)
    cor_tr_res = sum(1 for t in resolved.values() if t['adjudicated_result'] == 'CORRECT')
    tr_acc_res = round(cor_tr_res / total_tr_res * 100, 2)
    
    # Confirmed trials
    conf_tr_all = sum(1 for t in exp_trials if t['confirmed_count'] > 0)
    conf_tr_rate_all = round(conf_tr_all / total_tr_all * 100, 2)
    
    conf_tr_res = sum(1 for t in resolved.values() if t['confirmed_count'] > 0)
    conf_tr_rate_res = round(conf_tr_res / total_tr_res * 100, 2)
    
    tf = ls['total_frames']
    cf = ls['correct_frames']
    nf = ls['neutral_frames']
    inf = ls['incorrect_frames']
    
    frame_acc = round(cf / tf * 100, 2)
    neutral_rate = round(nf / tf * 100, 2)
    fp_rate = round(inf / tf * 100, 2)
    
    summary.append({
        'letter': exp,
        'total_trials_all': total_tr_all,
        'correct_trials_all': cor_tr_all,
        'trial_acc_all': tr_acc_all,
        'total_trials_res': total_tr_res,
        'correct_trials_res': cor_tr_res,
        'trial_acc_res': tr_acc_res,
        'total_frames': tf,
        'correct_frames': cf,
        'frame_acc': frame_acc,
        'neutral_frames': nf,
        'neutral_rate': neutral_rate,
        'incorrect_frames': inf,
        'fp_rate': fp_rate,
        'confirmed_trials_all': conf_tr_all,
        'confirmed_rate_all': conf_tr_rate_all,
        'confirmed_trials_res': conf_tr_res,
        'confirmed_rate_res': conf_tr_rate_res
    })

# Save to json
with open('scratch/reconciled_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({'letter_summary': summary, 'all_trials': list(trials.values())}, f, indent=2, ensure_ascii=False)

print("Computed reconciled metrics successfully!")

