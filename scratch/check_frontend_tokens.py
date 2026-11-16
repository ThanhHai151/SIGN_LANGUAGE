import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

confirmed_events = [r for r in records if r.get("confirmed")]
print(f"Total confirmed events in log: {len(confirmed_events)}")

# Backend condition:
# Whenever confirmed == True, backend recognizer does self.sentence_tokens.append(label_text)
backend_appends = len(confirmed_events)

# Frontend condition (Line 881 of CameraSignRecognizer.jsx):
# data.confirmed && data.new_token && data.sign !== 'NEUTRAL'
frontend_emitted = [r for r in confirmed_events if r.get("final_sign") != "NEUTRAL"]
frontend_dropped = [r for r in confirmed_events if r.get("final_sign") == "NEUTRAL"]

print(f"Backend sentence_tokens appends: {backend_appends}")
print(f"Frontend UI accepted tokens (data.sign !== 'NEUTRAL'): {len(frontend_emitted)}")
print(f"Frontend UI dropped tokens (data.sign === 'NEUTRAL'): {len(frontend_dropped)}")

print("\nDetails of Frontend dropped events:")
for r in frontend_dropped:
    print(f"  Frame {r.get('frame_idx')}, ts={r.get('timestamp')}, sign={r.get('final_sign')}, trial={r.get('trial_id')}")

