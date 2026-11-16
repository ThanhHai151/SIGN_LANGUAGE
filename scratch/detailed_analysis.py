import json
import numpy as np

def extract_features(filepath, filter_fn=None):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            if filter_fn and not filter_fn(row):
                continue
            geom = row.get('geometric', {})
            feat = geom.get('features', {})
            pred = row.get('final_sign') or row.get('predicted_sign') or 'NEUTRAL'
            data.append({
                'file': filepath,
                'idx': idx,
                'trial_id': row.get('trial_id'),
                'frame_idx': row.get('frame_idx'),
                'pred': pred,
                'conf': row.get('final_confidence', 0.0),
                'confirmed': row.get('confirmed', False),
                'geom_letter': geom.get('letter'),
                'i_diag': geom.get('i_rule_diagnostics'),
                'feat': feat
            })
    return data

d_2024 = extract_features('backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl')
d_2032 = extract_features('backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl')

# Group 1: Correct I
g_i = [d for d in d_2032 if d['pred'] == 'I']

# Group 2: Misclassified Y
g_y = [d for d in d_2024 if d['pred'] == 'Y']

# Group 3a: NEUTRAL in 2032 rep01 and continuation
g_neutral_2032 = [d for d in d_2032 if d['pred'] == 'NEUTRAL']
g_neutral_2032_rep01 = [d for d in g_neutral_2032 if d['trial_id'] == 'pilot_20261105_2032_I_rep01']
g_neutral_2032_unlabelled = [d for d in g_neutral_2032 if d['trial_id'] is None]

# Group 3b: NEUTRAL in 2024 rep02
g_neutral_2024_rep02 = [d for d in d_2024 if d['trial_id'] == 'pilot_20261105_2024_I_rep02' and d['pred'] == 'NEUTRAL']

print(f"Count Correct I: {len(g_i)}")
print(f"Count Misclassified Y: {len(g_y)}")
print(f"Count NEUTRAL 2032: {len(g_neutral_2032)} (rep01: {len(g_neutral_2032_rep01)}, unlabelled: {len(g_neutral_2032_unlabelled)})")
print(f"Count NEUTRAL 2024 rep02: {len(g_neutral_2024_rep02)}")

keys = ['d_PW', 'd_IW', 'd_MW', 'd_RW', 'd_TW', 'd_TI', 'd_TM', 'thumb_lat', 'ang_TP', 'pinky_curl', 'index_curl']

def summarize_group(name, items):
    print(f"\n--- Summary for {name} ({len(items)} frames) ---")
    for k in keys:
        vals = [d['feat'].get(k) for d in items if d['feat'].get(k) is not None]
        if vals:
            print(f"  {k:12s}: min={min(vals):.4f}, mean={np.mean(vals):.4f}, max={max(vals):.4f}, std={np.std(vals):.4f}")

summarize_group('Group I (Correct)', g_i)
summarize_group('Group Y (Misclassified)', g_y)
summarize_group('Group NEUTRAL 2032 (rep01 + continuation)', g_neutral_2032)
summarize_group('Group NEUTRAL 2024 rep02', g_neutral_2024_rep02)

