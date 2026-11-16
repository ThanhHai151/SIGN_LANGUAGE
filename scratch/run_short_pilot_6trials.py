import os
import sys
import time
import json
import urllib.request

API_BASE = "http://127.0.0.1:8000"
session_id = f"pilot_short_{time.strftime('%Y%m%d_%H%M%S')}"

print(f"Starting Short Webcam Pilot (6 trials: A x2, M x2, N x2)")
print(f"Session ID: {session_id}")

# 1. Enable pilot logging
ctrl_data = json.dumps({"enabled": True, "session_id": session_id}).encode()
req = urllib.request.Request(f"{API_BASE}/api/pilot/control", data=ctrl_data, headers={"Content-Type": "application/json"})
ctrl_res = json.loads(urllib.request.urlopen(req).read())
print(f"Pilot logging enabled. Target file: {ctrl_res['filepath']}")
log_path = ctrl_res['filepath']

# 2. Load fixture samples
with open("backend/tests/fixtures/real_alphabet_landmarks.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f)["samples"]

samples_by_letter = {
    "A": [s for s in fixtures if s["letter_upper"] == "A" and s.get("landmarks")],
    "M": [s for s in fixtures if s["letter_upper"] == "M" and s.get("landmarks")],
    "N": [s for s in fixtures if s["letter_upper"] == "N" and s.get("landmarks")],
}

trials_config = [
    ("A", 1),
    ("A", 2),
    ("M", 1),
    ("M", 2),
    ("N", 1),
    ("N", 2),
]

trial_summaries = []

for letter, rep in trials_config:
    trial_id = f"{session_id}_{letter}_rep{rep:02d}_att01"
    print(f"\n--- Running Trial: {trial_id} ({letter} rep {rep}) ---")
    
    cand_samples = samples_by_letter[letter]
    sample_to_use = cand_samples[(rep - 1) % len(cand_samples)]
    base_lms = sample_to_use["landmarks"]
    
    last_res = None
    frames_recorded = 0
    # Stream 10 frames for this trial
    for f_idx in range(1, 11):
        # Slightly jitter landmarks to simulate live hand jitter
        jittered = [[pt[0] + (f_idx * 0.0005), pt[1] - (f_idx * 0.0003), pt[2]] for pt in base_lms]
        payload = json.dumps({
            "landmarks": jittered,
            "mode": "letters",
            "expected_sign": letter,
            "session_id": session_id,
            "trial_id": trial_id,
            "frame_idx": f_idx,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "X-Conversion-Mode": "letters",
            "X-Expected-Sign": letter,
            "X-Session-Id": session_id,
            "X-Trial-Id": trial_id,
            "X-Frame-Idx": str(f_idx),
        }
        rec_req = urllib.request.Request(f"{API_BASE}/api/recognize_landmarks", data=payload, headers=headers)
        rec_res = json.loads(urllib.request.urlopen(rec_req).read())
        last_res = rec_res["data"]
        frames_recorded += 1
        time.sleep(0.04) # 25 fps stream
    
    # Flush trial
    flush_req = urllib.request.Request(f"{API_BASE}/api/pilot/flush", data=b"{}", headers={"Content-Type": "application/json"})
    flush_res = json.loads(urllib.request.urlopen(flush_req).read())
    
    trial_summaries.append({
        "trial_id": trial_id,
        "letter": letter,
        "rep": rep,
        "frames_sent": frames_recorded,
        "final_sign": last_res.get("sign"),
        "confidence": last_res.get("confidence"),
        "confirmed": last_res.get("confirmed"),
        "source": last_res.get("source"),
    })
    print(f"    Completed trial {trial_id}: frames={frames_recorded}, sign={last_res.get('sign')}, conf={last_res.get('confidence')}, confirmed={last_res.get('confirmed')}")

# 3. Disable pilot logging
urllib.request.urlopen(urllib.request.Request(f"{API_BASE}/api/pilot/control", data=json.dumps({"enabled": False}).encode(), headers={"Content-Type": "application/json"}))
print("\nPilot session ended and disabled.")

# 4. Analyze generated JSONL records
print("\n=======================================================")
print("ANALYZING GENERATED JSONL FILE")
print("=======================================================")
assert os.path.exists(log_path), "Log file missing!"

all_records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

print(f"Total lines in JSONL: {len(all_records)}")

# Validate trial counts
trials_in_file = sorted(list(set(r["trial_id"] for r in all_records)))
print(f"Unique trials in JSONL: {len(trials_in_file)}")
assert len(trials_in_file) == 6, f"Expected 6 trials, got {len(trials_in_file)}"

stats_by_trial = {}
for tid in trials_in_file:
    recs = [r for r in all_records if r["trial_id"] == tid]
    f_indices = [r["frame_idx"] for r in recs]
    expected = recs[0]["expected_sign"]
    sources = [r["prediction_source"] for r in recs]
    gatings = [r["gating_status"] for r in recs]
    final_signs = [r["final_sign"] for r in recs]
    confs = [r["final_confidence"] for r in recs]
    has_mlp = all(r.get("has_static_head") is True for r in recs)
    load_status = recs[0]["static_head"]["load_status"]
    pytorch_v = recs[0]["static_head"]["pytorch_version"]
    
    stats_by_trial[tid] = {
        "letter": expected,
        "frame_count": len(recs),
        "frame_indices": f_indices,
        "is_monotonic": f_indices == list(range(1, len(recs) + 1)),
        "sources": sorted(list(set(sources))),
        "gatings": sorted(list(set(gatings))),
        "dominant_sign": max(set(final_signs), key=final_signs.count),
        "avg_conf": round(sum(confs) / len(confs), 3),
        "has_static_head_all": has_mlp,
        "load_status": load_status,
        "pytorch_version": pytorch_v,
    }

print("\nPer-Trial Verification Table:")
for tid, st in stats_by_trial.items():
    print(f"  [{tid}] Letter: {st['letter']} | Frames: {st['frame_count']} | Monotonic: {st['is_monotonic']} | Sources: {st['sources']} | Gating: {st['gatings']} | MLP Loaded: {st['has_static_head_all']} ({st['load_status']}, {st['pytorch_version']}) | Sign: {st['dominant_sign']} (conf: {st['avg_conf']})")

summary_out = {
    "session_id": session_id,
    "log_path": log_path,
    "total_records": len(all_records),
    "trial_summaries": trial_summaries,
    "stats_by_trial": stats_by_trial,
}

with open("scratch/short_pilot_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

print("\nSummary written to scratch/short_pilot_summary.json")

