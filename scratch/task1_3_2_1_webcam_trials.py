import sys
sys.path.insert(0, '.')
import json

log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
trials = {}
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        tid = r.get('trial_id')
        if r.get('expected_sign') == 'P' and tid and 'P_rep' in tid:
            if tid not in trials:
                trials[tid] = []
            trials[tid].append(r)

print("=== WEBCAM P 10 TRIALS STRUCTURE ===")
for tid, frames in trials.items():
    rep_name = tid.split('_')[-1]
    f_indices = [f['frame_idx'] for f in frames]
    t_start = frames[0]['timestamp']
    t_end = frames[-1]['timestamp']
    dur = t_end - t_start
    print(f"Trial {rep_name:5s}: {len(frames):2d} frames (idx: {min(f_indices)} to {max(f_indices)}), duration: {dur:.2f}s")
    for f in frames:
        f_idx = f['frame_idx']
        feat = f['geometric']['features']
        d_rw = feat.get('d_RW')
        d_pw = feat.get('d_PW')
        idx_tip_y = feat.get('idx_tip_y')
        ang_im = feat.get('ang_IM_dir')
        is_fail = (d_rw > 1.60 or d_pw > 1.50)
        mark = "FAIL (NEUTRAL)" if is_fail else "MATCH P"
        if is_fail or f_idx in (1, 2, max(f_indices)):
            print(f"   Frame {f_idx:2d} (t={f['timestamp'] - t_start:.2f}s): {mark:14s} | d_RW={d_rw:.3f}, d_PW={d_pw:.3f}, idx_tip_y={idx_tip_y:+.3f}, ang_IM={ang_im:4.1f}°")

