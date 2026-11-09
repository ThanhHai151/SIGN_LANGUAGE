import json
import numpy as np

def extract_features(filter_fn):
    feats = []
    with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if filter_fn(r):
                geo = r.get('geometric', {})
                f_dict = geo.get('features')
                if f_dict and isinstance(f_dict, dict):
                    feats.append(f_dict)
    return feats

h_to_v_feats = extract_features(lambda r: r.get('expected_sign') == 'H' and r.get('final_sign') == 'V')
h_to_neu_feats = extract_features(lambda r: r.get('expected_sign') == 'H' and r.get('final_sign') == 'NEUTRAL')
v_to_v_feats = extract_features(lambda r: r.get('expected_sign') == 'V' and r.get('final_sign') == 'V')

print(f"Counts: H->V = {len(h_to_v_feats)}, H->NEUTRAL = {len(h_to_neu_feats)}, V->V = {len(v_to_v_feats)}")

feature_keys = [
    'd_IW', 'd_MW', 'd_IM', 'diff_x_IM', 'ang_IM_dir',
    'idx_horiz_ratio', 'mid_horiz_ratio', 'idx_tip_y', 'mid_tip_y',
    'mcp_width_2d', 'thumb_ring_dist', 'thumb_mid_dist',
    'idx_straight', 'mid_straight', 'index_curl', 'middle_curl',
    'ring_curl', 'pinky_curl'
]

print("\n" + "="*110)
print(f"{'Feature':<18} | {'H -> V (Mean ± Std [Med])':<30} | {'H -> NEUTRAL (Mean ± Std [Med])':<30} | {'V -> V (Mean ± Std [Med])':<30}")
print("="*110)

stats_table = {}

for k in feature_keys:
    def get_stats(f_list):
        vals = [float(f[k]) for f in f_list if f.get(k) is not None]
        if not vals:
            return None
        return {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'median': np.median(vals),
            'min': np.min(vals),
            'max': np.max(vals)
        }
    
    s_hv = get_stats(h_to_v_feats)
    s_hn = get_stats(h_to_neu_feats)
    s_vv = get_stats(v_to_v_feats)
    
    stats_table[k] = {'H_to_V': s_hv, 'H_to_NEUTRAL': s_hn, 'V_to_V': s_vv}
    
    str_hv = f"{s_hv['mean']:.3f} ± {s_hv['std']:.3f} [{s_hv['median']:.3f}]" if s_hv else "N/A"
    str_hn = f"{s_hn['mean']:.3f} ± {s_hn['std']:.3f} [{s_hn['median']:.3f}]" if s_hn else "N/A"
    str_vv = f"{s_vv['mean']:.3f} ± {s_vv['std']:.3f} [{s_vv['median']:.3f}]" if s_vv else "N/A"
    
    print(f"{k:<18} | {str_hv:<30} | {str_hn:<30} | {str_vv:<30}")

with open('scratch/h_v_comparison_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats_table, f, indent=2)

print("\nSaved comparison stats to scratch/h_v_comparison_stats.json")

