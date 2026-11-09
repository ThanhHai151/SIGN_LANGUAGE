import sys
import os
import json
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

active_frames = [r for r in lines if r.get("expected_sign") == "P"]

# Check confirmed events
confirmed_events = []
for idx, r in enumerate(active_frames):
    if r.get("confirmed"):
        confirmed_events.append({
            "idx": idx,
            "trial_id": r.get("trial_id"),
            "frame_idx": r.get("frame_idx"),
            "timestamp": r.get("timestamp"),
            "final_sign": r.get("final_sign"),
            "final_confidence": r.get("final_confidence"),
            "new_token": r.get("new_token"),
            "sentence": r.get("sentence"),
        })

print(f"Total confirmed events in active frames: {len(confirmed_events)}")
for ev in confirmed_events:
    print(f"  Trial: {ev['trial_id']}, FrameIdx: {ev['frame_idx']}, Sign: {ev['final_sign']}, Conf: {ev['final_confidence']}, NewToken: {ev['new_token']}, Sentence: '{ev['sentence']}'")

# Check all new_token occurrences in entire file (including idle frames)
all_tokens = [r.get("new_token") for r in lines if r.get("new_token")]
print(f"\nAll new_tokens across entire file (including idle): {collections.Counter(all_tokens)}")

# Check last sentence
print(f"Final sentence at end of file: '{lines[-1].get('sentence')}'")

