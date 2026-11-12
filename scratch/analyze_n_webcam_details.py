import json

lines = [json.loads(l) for l in open('backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl', encoding='utf-8')]

print("=== N WEBCAM FRAMES DETAILS ===")
print(f"Total lines: {len(lines)}")
for i, f in enumerate(lines):
    pred = f.get('predicted_sign')
    conf = f.get('confidence')
    h_conf = f.get('head_confidence')
    gate = f.get('gating_status')
    rej = f.get('reject_reason')
    hand = f.get('hand_detected')
    status = f.get('frame_status')
    if pred != 'NEUTRAL' or i < 15:
        print(f"Frame {i:03d} (rep {f.get('elapsed_sec', 0):.2f}s): pred={pred}, conf={conf}, head_conf={h_conf}, gate={gate}, rej={rej}, hand={hand}")

# Summary of reject_reason
reasons = {}
preds = {}
for f in lines:
    r = f.get('reject_reason')
    p = f.get('predicted_sign')
    reasons[r] = reasons.get(r, 0) + 1
    preds[p] = preds.get(p, 0) + 1

print("\nPreds distribution:", preds)
print("Reject reasons distribution:", reasons)

