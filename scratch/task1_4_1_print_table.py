import sys
sys.path.insert(0, '.')
import json

with open('scratch/task1_4_1_replay_all_frames.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

focus = [r for r in data if r['frame_index'] and 134 <= r['frame_index'] <= 148]

print(f"{'Frame':5s} | {'Time':6s} | {'Phase':11s} | {'Geometric Result':18s} | {'Pre-Smooth (Source)':22s} | {'History Window (10)':30s} | {'Votes':18s} | {'Confirmed?':10s} | {'Last Conf'}")
print("-" * 145)

for r in focus:
    f_idx = r['frame_index']
    t_sec = f"{r['timestamp']:.3f}s" if r['timestamp'] else "N/A"
    phase = r['phase']
    geom = f"{r['geom_letter'] or 'NONE'} ({r['geom_status']})"
    pre = f"{r['runtime_sign']} {r['runtime_conf']:.2f} ({r['runtime_source']})"
    hist = "[" + ", ".join([f"'{s}'" for s, c in r['history_after']]) + "]"
    votes = str(r['vote_counts'])
    conf = "CONFIRMED" if r['runtime_confirmed'] else "False"
    last = f"{r['last_confirmed_sign']} (t={r['last_confirmed_time']:.3f}s)"
    print(f"{f_idx:5d} | {t_sec:6s} | {phase:11s} | {geom:18s} | {pre:22s} | {hist:46s} | {votes:18s} | {conf:10s} | {last}")
