import json
import numpy as np

latest = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(latest, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

s_frames = []
for idx, l in enumerate(lines):
    data = json.loads(l)
    if data.get('expected_sign') == 'S':
        geom = data.get('geometric', {})
        feats = geom.get('features', {})
        s_diag = geom.get('s_rule_diagnostics', {})
        s_frames.append({
            'frame_idx': data.get('frame_idx'),
            'feats': feats,
            'failed': s_diag.get('failed_conditions') if s_diag else None
        })

print(f"Total S frames recorded on webcam: {len(s_frames)}")
if s_frames:
    keys = ['d_TW', 'd_IW', 'd_MW', 'd_RW', 'd_PW', 'd_TI', 'd_TM', 'thumb_lat', 'palm_scale']
    print(f"{'F_idx':5s} | {'d_TW':5s} {'d_IW':5s} {'d_MW':5s} {'d_RW':5s} {'d_PW':5s} | {'d_TI':5s} {'d_TM':5s} {'th_lat':6s} | Failed conditions")
    print("-" * 95)
    for sf in s_frames[:35]:
        f = sf['feats']
        dtw = f.get('d_TW', 0)
        diw = f.get('d_IW', 0)
        dmw = f.get('d_MW', 0)
        drw = f.get('d_RW', 0)
        dpw = f.get('d_PW', 0)
        dti = f.get('d_TI', 0)
        dtm = f.get('d_TM', 0)
        th_lat = f.get('thumb_lat', 0)
        failed = sf['failed']
        print(f"{sf['frame_idx']:<5d} | {dtw:5.2f} {diw:5.2f} {dmw:5.2f} {drw:5.2f} {dpw:5.2f} | {dti:5.2f} {dtm:5.2f} {th_lat:6.2f} | {failed}")

    # Compute min, median, max of user's live S frames
    print("\n--- STATISTICAL SUMMARY OF USER'S WEBCAM S FRAMES ---")
    for k in keys:
        vals = [sf['feats'].get(k) for sf in s_frames if sf['feats'].get(k) is not None]
        if vals:
            print(f"{k:<12s}: min={np.min(vals):.3f}, 25%={np.percentile(vals, 25):.3f}, median={np.median(vals):.3f}, 75%={np.percentile(vals, 75):.3f}, max={np.max(vals):.3f}")

