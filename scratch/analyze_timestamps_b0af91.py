import json
import numpy as np

fpath = 'backend/data/diagnostics/pilot_eval_29_pilot_20261111_050708_b0af91_20261111_051044_frames.jsonl'
records = []
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total frames: {len(records)}")

# Group by trial reps using timestamps
# Rep 1: ts ~ 206-210
# Rep 2: ts ~ 221-225
# Rep 3: ts ~ 237-241
# Rep 4: ts ~ 257-262
# Rep 5: ts ~ 277-281

reps = [
    ("Rep 1", records[0:9]),
    ("Rep 2", records[9:19]),
    ("Rep 3", records[19:29]),
    ("Rep 4", records[29:40]),
    ("Rep 5", records[40:50]),
]

for name, recs in reps:
    ts_list = [r['timestamp'] for r in recs]
    deltas = np.diff(ts_list)
    duration = ts_list[-1] - ts_list[0] if len(ts_list) > 1 else 0
    fps = (len(recs) - 1) / duration if duration > 0 else 0
    preds = {}
    for r in recs:
        p = r['predicted_sign']
        preds[p] = preds.get(p, 0) + 1
    avg_delta = np.mean(deltas) if len(deltas) > 0 else 0
    min_delta = np.min(deltas) if len(deltas) > 0 else 0
    max_delta = np.max(deltas) if len(deltas) > 0 else 0
    print(f"\n{name} ({len(recs)} frames, duration={duration:.2f}s, effective_fps={fps:.2f} Hz):")
    print(f"  Timestamps: [{ts_list[0]:.2f}s -> {ts_list[-1]:.2f}s]")
    print(f"  Frame interval (dt): mean={avg_delta*1000:.1f}ms, min={min_delta*1000:.1f}ms, max={max_delta*1000:.1f}ms")
    print(f"  Predictions: {preds}")

# Total stats
total_frames = len(records)
match_q = sum(1 for r in records if r['predicted_sign'] == 'Q')
mismatch_p = sum(1 for r in records if r['predicted_sign'] == 'P')
neutral = sum(1 for r in records if r['predicted_sign'] == 'NEUTRAL')

print("\n=======================================================")
print("TOTAL SESSION FRAME METRICS (50 frames total):")
print(f"  Total frames recorded across all 5 reps: {total_frames}")
print(f"  Q match frames   : {match_q} ({match_q/total_frames*100:.1f}%)")
print(f"  P mismatch frames: {mismatch_p} ({mismatch_p/total_frames*100:.1f}%)")
print(f"  NEUTRAL frames   : {neutral} ({neutral/total_frames*100:.1f}%)")
print(f"  Active frames (Q + P): {match_q + mismatch_p} ({ (match_q+mismatch_p)/total_frames*100:.1f}% coverage)")
print(f"  Accuracy on active frames (18 frames): {match_q}/{match_q+mismatch_p} ({match_q/(match_q+mismatch_p)*100:.1f}%)")
print(f"  Overall coverage on entire 50-frame session: {match_q}/50 ({match_q/50*100:.1f}%)")

