import json

fpath = "backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl"
with open(fpath, "r", encoding="utf-8") as f:
    frames = [json.loads(line) for line in f]

print(f"Total frames: {len(frames)}")

# Group by elapsed_sec resets or check timestamps
deltas = []
for i in range(1, len(frames)):
    dt = frames[i]["timestamp"] - frames[i-1]["timestamp"]
    if 0 < dt < 2.0: # within same rep
        deltas.append(dt)

print(f"Intra-rep frame intervals count: {len(deltas)}")
if deltas:
    import statistics
    print(f"Mean delta: {statistics.mean(deltas):.3f}s -> {1.0/statistics.mean(deltas):.2f} FPS")
    print(f"Median delta: {statistics.median(deltas):.3f}s -> {1.0/statistics.median(deltas):.2f} FPS")
    print(f"Min delta: {min(deltas):.3f}s ({1.0/min(deltas):.2f} FPS)")
    print(f"Max delta: {max(deltas):.3f}s ({1.0/max(deltas):.2f} FPS)")

# Also print the actual frames for rep 1
print("\nFirst 10 frames:")
for idx, f in enumerate(frames[:10]):
    print(f"  Frame {idx+1}: ts={f['timestamp']:.3f}, elapsed={f.get('elapsed_sec')}, phase={f.get('phase')}, sign={f.get('predicted_sign')}")

