import json
import numpy as np

jsonl_path = 'backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl'

# Load records
all_rule_v_records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        r = json.loads(line)
        if r.get('trial_id') and r.get('expected_sign'):
            fs = r.get('final_sign')
            src = r.get('prediction_source')
            geo = r.get('geometric', {})
            if fs == 'V' and src == 'geometric_rule' and geo.get('status') == 'MATCH':
                r['line_number'] = line_idx + 1
                all_rule_v_records.append(r)

print(f"Total Rule V MATCH records: {len(all_rule_v_records)}")

# Group by expected_sign
by_sign = {}
for r in all_rule_v_records:
    exp = r.get('expected_sign')
    by_sign.setdefault(exp, []).append(r)

print("Distribution by expected_sign:")
for sign, recs in sorted(by_sign.items()):
    print(f"  {sign}: {len(recs)} frames")

h_recs = by_sign.get('H', [])
v_recs = by_sign.get('V', [])
k_recs = by_sign.get('K', [])

# 1. Candidate simulation
# Condition A: mid_horiz_ratio <= 0.50
# Condition B: ang_im_dir <= 38.0
# Condition A AND B: mid_horiz_ratio <= 0.50 and ang_im_dir <= 38.0

def eval_condition(r, cond):
    feats = r.get('geometric', {}).get('features', {})
    mid = feats.get('mid_horiz_ratio')
    ang = feats.get('ang_IM_dir')
    if cond == 'A':
        return (mid is not None) and (mid <= 0.50)
    elif cond == 'B':
        return (ang is not None) and (ang <= 38.0)
    elif cond == 'A_AND_B':
        return (mid is not None) and (mid <= 0.50) and (ang is not None) and (ang <= 38.0)
    elif cond == 'A_OR_B':
        return ((mid is not None) and (mid <= 0.50)) or ((ang is not None) and (ang <= 38.0))
    return False

sim_results = {}
for cond_name, label in [('A', 'Condition A (mid_horiz_ratio <= 0.50)'),
                         ('B', 'Condition B (ang_im_dir <= 38.0)'),
                         ('A_AND_B', 'Condition A AND B (mid <= 0.50 & ang <= 38.0)')]:
    # H -> V: we want to eliminate (pass == False means eliminated)
    h_pass = [r for r in h_recs if eval_condition(r, cond_name)]
    h_elim = [r for r in h_recs if not eval_condition(r, cond_name)]
    
    # V -> V: we want to retain (pass == True means retained)
    v_pass = [r for r in v_recs if eval_condition(r, cond_name)]
    v_elim = [r for r in v_recs if not eval_condition(r, cond_name)]
    
    # K -> V:
    k_pass = [r for r in k_recs if eval_condition(r, cond_name)]
    k_elim = [r for r in k_recs if not eval_condition(r, cond_name)]

    sim_results[cond_name] = {
        'label': label,
        'H_total': len(h_recs),
        'H_eliminated': len(h_elim),
        'H_eliminated_pct': len(h_elim) / len(h_recs) * 100 if h_recs else 0,
        'H_leaked': len(h_pass),
        'H_leaked_pct': len(h_pass) / len(h_recs) * 100 if h_recs else 0,
        'V_total': len(v_recs),
        'V_retained': len(v_pass),
        'V_retained_pct': len(v_pass) / len(v_recs) * 100 if v_recs else 0,
        'V_eliminated': len(v_elim),
        'V_eliminated_pct': len(v_elim) / len(v_recs) * 100 if v_recs else 0,
        'K_total': len(k_recs),
        'K_eliminated': len(k_elim),
        'K_leaked': len(k_pass),
    }

print("\nCandidate simulation results:")
for k, v in sim_results.items():
    print(f"[{k}] {v['label']}:")
    print(f"  H->V: Eliminated {v['H_eliminated']}/{v['H_total']} ({v['H_eliminated_pct']:.2f}%), Leaked: {v['H_leaked']}")
    print(f"  V->V: Retained {v['V_retained']}/{v['V_total']} ({v['V_retained_pct']:.2f}%), Eliminated: {v['V_eliminated']}")
    print(f"  K->V: Eliminated {v['K_eliminated']}/{v['K_total']}, Leaked: {v['K_leaked']}")

# 2. Sensitivity sweep
# mid_horiz_ratio: [0.40, 0.45, 0.50, 0.55, 0.60]
# ang_im_dir: [28.0, 32.0, 36.0, 38.0, 40.0, 44.0]
mid_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
ang_thresholds = [28.0, 32.0, 36.0, 38.0, 40.0, 44.0]

print("\n--- Sensitivity Sweep: mid_horiz_ratio alone ---")
for t in mid_thresholds:
    h_pass = sum(1 for r in h_recs if r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= t)
    v_pass = sum(1 for r in v_recs if r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= t)
    k_pass = sum(1 for r in k_recs if r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= t)
    print(f"mid <= {t:.2f} | H elim: {len(h_recs)-h_pass}/{len(h_recs)} ({(len(h_recs)-h_pass)/len(h_recs)*100:.1f}%) | V retained: {v_pass}/{len(v_recs)} ({v_pass/len(v_recs)*100:.1f}%) | K elim: {len(k_recs)-k_pass}/{len(k_recs)}")

print("\n--- Sensitivity Sweep: ang_im_dir alone ---")
for t in ang_thresholds:
    h_pass = sum(1 for r in h_recs if r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= t)
    v_pass = sum(1 for r in v_recs if r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= t)
    k_pass = sum(1 for r in k_recs if r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= t)
    print(f"ang <= {t:.1f} | H elim: {len(h_recs)-h_pass}/{len(h_recs)} ({(len(h_recs)-h_pass)/len(h_recs)*100:.1f}%) | V retained: {v_pass}/{len(v_recs)} ({v_pass/len(v_recs)*100:.1f}%) | K elim: {len(k_recs)-k_pass}/{len(k_recs)}")

print("\n--- Sensitivity Sweep: Joint Grid (mid <= M AND ang <= A) ---")
grid_results = []
for m in mid_thresholds:
    for a in ang_thresholds:
        h_pass = sum(1 for r in h_recs if (r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= m) and (r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= a))
        v_pass = sum(1 for r in v_recs if (r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= m) and (r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= a))
        k_pass = sum(1 for r in k_recs if (r.get('geometric', {}).get('features', {}).get('mid_horiz_ratio', 999) <= m) and (r.get('geometric', {}).get('features', {}).get('ang_IM_dir', 999) <= a))
        grid_results.append({
            'mid_thresh': m,
            'ang_thresh': a,
            'h_elim': len(h_recs) - h_pass,
            'h_elim_pct': (len(h_recs) - h_pass) / len(h_recs) * 100,
            'v_ret': v_pass,
            'v_ret_pct': v_pass / len(v_recs) * 100,
            'k_elim': len(k_recs) - k_pass
        })

# 3. 2x2 Interaction Matrix
# mid <= 0.50 vs mid > 0.50
# ang <= 38.0 vs ang > 38.0
def get_2x2_matrix(records):
    q11 = 0 # mid <= 0.50 and ang <= 38
    q12 = 0 # mid <= 0.50 and ang > 38
    q21 = 0 # mid > 0.50 and ang <= 38
    q22 = 0 # mid > 0.50 and ang > 38
    for r in records:
        feats = r.get('geometric', {}).get('features', {})
        mid = feats.get('mid_horiz_ratio', 999)
        ang = feats.get('ang_IM_dir', 999)
        if mid <= 0.50 and ang <= 38.0:
            q11 += 1
        elif mid <= 0.50 and ang > 38.0:
            q12 += 1
        elif mid > 0.50 and ang <= 38.0:
            q21 += 1
        else:
            q22 += 1
    return {'mid<=0.50 & ang<=38': q11,
            'mid<=0.50 & ang>38': q12,
            'mid>0.50 & ang<=38': q21,
            'mid>0.50 & ang>38': q22}

matrix_h = get_2x2_matrix(h_recs)
matrix_v = get_2x2_matrix(v_recs)
matrix_k = get_2x2_matrix(k_recs)

print("\n2x2 Interaction Matrix:")
print("H->V (140 frames):", matrix_h)
print("V->V (362 frames):", matrix_v)
print("K->V (1 frame):", matrix_k)

# 4. Per-trial breakdown
# We need to know all trials for H and V in the dataset
trial_stats = {}
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        t_id = r.get('trial_id')
        exp = r.get('expected_sign')
        if t_id and exp in ('H', 'V'):
            if t_id not in trial_stats:
                trial_stats[t_id] = {
                    'expected': exp,
                    'total_frames': 0,
                    'rule_v_match': 0,
                    'pass_A': 0,
                    'pass_B': 0,
                    'pass_A_and_B': 0
                }
            trial_stats[t_id]['total_frames'] += 1
            
            fs = r.get('final_sign')
            src = r.get('prediction_source')
            geo = r.get('geometric', {})
            if fs == 'V' and src == 'geometric_rule' and geo.get('status') == 'MATCH':
                trial_stats[t_id]['rule_v_match'] += 1
                if eval_condition(r, 'A'):
                    trial_stats[t_id]['pass_A'] += 1
                if eval_condition(r, 'B'):
                    trial_stats[t_id]['pass_B'] += 1
                if eval_condition(r, 'A_AND_B'):
                    trial_stats[t_id]['pass_A_and_B'] += 1

print("\nPer-trial statistics:")
for t_id in sorted(trial_stats.keys()):
    st = trial_stats[t_id]
    print(f"Trial {t_id} (exp={st['expected']}): total={st['total_frames']}, RuleV={st['rule_v_match']}, passA={st['pass_A']}, passB={st['pass_B']}, passAB={st['pass_A_and_B']}")

# Save simulation data
sim_data = {
    'candidate_simulation': sim_results,
    'sensitivity_grid': grid_results,
    'interaction_matrix': {
        'H': matrix_h,
        'V': matrix_v,
        'K': matrix_k
    },
    'per_trial': trial_stats
}

with open('scratch/counterfactual_simulation_results.json', 'w', encoding='utf-8') as f:
    json.dump(sim_data, f, indent=2)

print("\nSaved scratch/counterfactual_simulation_results.json successfully!")

