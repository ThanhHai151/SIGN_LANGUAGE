import json
import numpy as np

features_to_audit = [
    'ang_IM_dir',
    'mid_horiz_ratio',
    'idx_horiz_ratio',
    'd_IW',
    'd_MW',
    'd_IM',
    'diff_x_IM',
    'idx_tip_y',
    'mcp_width_2d',
    'thumb_ring_dist'
]

h_to_v_records = []
v_to_v_records = []
k_to_v_records = []

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('trial_id') and r.get('expected_sign'):
            fs = r.get('final_sign')
            src = r.get('prediction_source')
            geo = r.get('geometric', {})
            if fs == 'V' and src == 'geometric_rule' and geo.get('status') == 'MATCH':
                exp = r.get('expected_sign')
                if exp == 'H':
                    h_to_v_records.append(r)
                elif exp == 'V':
                    v_to_v_records.append(r)
                elif exp == 'K':
                    k_to_v_records.append(r)

print(f"Loaded: H->V = {len(h_to_v_records)}, V->V = {len(v_to_v_records)}, K->V = {len(k_to_v_records)}")

def compute_detailed_stats(records, group_name):
    group_stats = {}
    for feat_key in features_to_audit:
        vals = []
        missing = 0
        for r in records:
            f_dict = r.get('geometric', {}).get('features', {})
            val = f_dict.get(feat_key)
            if val is not None:
                vals.append(float(val))
            else:
                missing += 1
        
        arr = np.array(vals)
        count_valid = len(arr)
        if count_valid > 0:
            stats = {
                'group': group_name,
                'feature': feat_key,
                'count_valid': count_valid,
                'count_missing': missing,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'p5': float(np.percentile(arr, 5)),
                'p10': float(np.percentile(arr, 10)),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'p95': float(np.percentile(arr, 95)),
            }
        else:
            stats = {
                'group': group_name,
                'feature': feat_key,
                'count_valid': 0,
                'count_missing': missing,
            }
        group_stats[feat_key] = stats
    return group_stats

h_stats = compute_detailed_stats(h_to_v_records, 'H->V')
v_stats = compute_detailed_stats(v_to_v_records, 'V->V')

with open('scratch/detailed_distribution_stats.json', 'w', encoding='utf-8') as f:
    json.dump({'H_to_V': h_stats, 'V_to_V': v_stats}, f, indent=2)

print("\nComputed detailed feature distribution stats successfully!")

