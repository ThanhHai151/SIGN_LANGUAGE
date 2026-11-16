import json

data = json.load(open('scratch/detailed_distribution_stats.json', 'r', encoding='utf-8'))
h_stats = data['H_to_V']
v_stats = data['V_to_V']

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

print(f"{'Feature':<16} | {'Grp':<4} | {'Val/Mis':<7} | {'Min':<6} | {'P5':<6} | {'P10':<6} | {'P25':<6} | {'Med':<6} | {'Mean ± Std':<15} | {'P75':<6} | {'P90':<6} | {'P95':<6} | {'Max':<6}")
print("="*135)

for f in features_to_audit:
    hs = h_stats[f]
    vs = v_stats[f]
    
    print(f"{f:<16} | H->V | {hs['count_valid']}/{hs['count_missing']} | {hs['min']:.2f} | {hs['p5']:.2f} | {hs['p10']:.2f} | {hs['p25']:.2f} | {hs['median']:.2f} | {hs['mean']:.2f}±{hs['std']:.2f} | {hs['p75']:.2f} | {hs['p90']:.2f} | {hs['p95']:.2f} | {hs['max']:.2f}")
    print(f"{'':<16} | V->V | {vs['count_valid']}/{vs['count_missing']} | {vs['min']:.2f} | {vs['p5']:.2f} | {vs['p10']:.2f} | {vs['p25']:.2f} | {vs['median']:.2f} | {vs['mean']:.2f}±{vs['std']:.2f} | {vs['p75']:.2f} | {vs['p90']:.2f} | {vs['p95']:.2f} | {vs['max']:.2f}")
    print("-"*135)

