import json
import glob
import os

files = [
    "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_013304.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_022547.jsonl"
]

for pf in files:
    fname = os.path.basename(pf)
    if not os.path.exists(pf):
        print(f"{fname}: NOT FOUND")
        continue
    trials = {}
    with open(pf, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            t_id = rec.get("trial_id")
            if t_id:
                if t_id not in trials: trials[t_id] = []
                trials[t_id].append(rec)
    print(f"\n=== {fname} (Total Trials: {len(trials)}) ===")
    durs = []
    fps_list = []
    has_sync_or_phase = False
    for t_id, f_list in sorted(trials.items()):
        ts = [f.get("timestamp") for f in f_list if f.get("timestamp") is not None]
        dur = (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
        durs.append(dur)
        fps = len(f_list) / dur if dur > 0 else 0
        fps_list.append(fps)
        # Check if record has explicit phase field
        if any(f.get("phase") or f.get("kinematic_phase") for f in f_list):
            has_sync_or_phase = True
    print(f"  Duration Range: min={min(durs):.2f}s, max={max(durs):.2f}s, mean={sum(durs)/len(durs):.2f}s")
    print(f"  Frames per trial: min={min(len(v) for v in trials.values())}, max={max(len(v) for v in trials.values())}, mean={sum(len(v) for v in trials.values())/len(trials):.1f}")
    print(f"  Estimated FPS: mean={sum(fps_list)/len(fps_list):.1f}")
    print(f"  Has explicit phase field: {has_sync_or_phase}")

