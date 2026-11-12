import json

with open('backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl', 'r') as f:
    frames = [json.loads(line) for line in f]

for i, f in enumerate(frames):
    if f.get('hand_detected') is None:
        print(f"Frame {i+1:2d} | rep_frame_idx={f.get('frame_idx')} | elapsed={f.get('elapsed_sec'):.3f}s | phase={f.get('phase')} | pred={f.get('predicted_sign')} ({f.get('confidence')}) | confirmed={f.get('confirmed')}")

