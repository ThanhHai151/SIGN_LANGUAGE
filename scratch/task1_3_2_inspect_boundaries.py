import sys
sys.path.insert(0, '.')
import json
from scratch.task1_3_2_production_replay import fixtures, webcam_p_active, evaluate_p_candidate_conditions

print("=== 3. BOUNDARY ANALYSIS OF CANDIDATE THRESHOLDS ===")

# Look at Webcam P 90 frames: min and max for each feature vs threshold
feats = [item['features'] for item in webcam_p_active]

keys = [
    ('d_IW', '>=', 1.70),
    ('idx_straight', '>=', 0.90),
    ('idx_tip_y', '>=', 0.60),
    ('idx_dy', '>=', 0.60),
    ('idx_horiz_ratio', '<=', 0.70),
    ('d_MW', '>=', 1.35),
    ('mid_straight', '>=', 0.80),
    ('mid_tip_y', '>=', 0.60),
    ('d_TM', '<=', 1.10),
    ('d_TI', '<=', 1.05),
    ('d_TW', '>=', 1.20),
    ('ang_IM_dir', 'range', (15.0, 50.0)),
    ('d_RW', '<=', 1.60),
    ('d_PW', '<=', 1.50),
]

print(f"{'Feature':16s} | {'Op':5s} | {'Thresh':12s} | {'Webcam Min':10s} | {'Webcam Max':10s} | {'Margin to Boundary':20s}")
print("-" * 80)
for k, op, th in keys:
    vals = [f[k] for f in feats]
    vmin, vmax = min(vals), max(vals)
    if op == '>=':
        margin = vmin - th
        margin_str = f"min {vmin:.3f} >= {th:.2f} (margin: +{margin:.3f})"
    elif op == '<=':
        margin = th - vmax
        margin_str = f"max {vmax:.3f} <= {th:.2f} (margin: {margin:+.3f})"
    elif op == 'range':
        low, high = th
        margin_low = vmin - low
        margin_high = high - vmax
        margin_str = f"low +{margin_low:.1f}°, high -{margin_high:.1f}°"
    print(f"{k:16s} | {op:5s} | {str(th):12s} | {vmin:10.4f} | {vmax:10.4f} | {margin_str}")

# Look at the 4 rejected frames in Webcam P
print("\n--- DETAILED LOOK AT THE 4 REJECTED WEBCAM P FRAMES ---")
for item in webcam_p_active:
    eval_res = evaluate_p_candidate_conditions(item['features'])
    if not eval_res['matched']:
        print(f"Frame {item['id']}: failed={eval_res['failed']}")
        for fkey in eval_res['failed']:
            print(f"   {fkey}: val={item['features'].get(fkey.split('_')[1].upper(), item['features'].get('d_RW' if 'rw' in fkey else 'd_PW'))}")

