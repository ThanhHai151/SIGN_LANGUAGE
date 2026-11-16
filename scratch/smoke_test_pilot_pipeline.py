import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath("."))

from backend.app import (
    app,
    recognizer,
    control_pilot_logging,
    flush_pilot_log,
    recognize_landmarks,
    PilotControlRequest,
    LandmarkRequest,
)

class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}

print("=== SMOKE TEST PILOT PIPELINE ===")

# Test 1: Check Static Head status & recognizer
print("1. Recognizer ready. Has static head:", getattr(recognizer, "has_static_head", False))

# Test 2: Enable Pilot Logging with session_id via control_pilot_logging
session_id = f"smoke_test_{int(time.time())}"
ctrl_req = PilotControlRequest(enabled=True, session_id=session_id)
res_control = control_pilot_logging(ctrl_req)
print("2. Pilot control enable:", res_control)
assert res_control["success"] is True
filepath = res_control["filepath"]
print("   Target log filepath:", filepath)
assert os.path.exists(filepath), f"File {filepath} was not created!"

# Test 3: Load fixture sample for 'A'
fixture_path = os.path.join("backend", "tests", "fixtures", "real_alphabet_landmarks.json")
with open(fixture_path, "r", encoding="utf-8") as f:
    fixture_data = json.load(f)
a_sample = next(s for s in fixture_data["samples"] if s["letter_upper"] == "A" and s.get("quality_status") == "ok")

# Test 4: Simulate Trial A/rep01/att01 (Phase 1: 5 frames)
trial_id_att01 = f"{session_id}_A_rep01_att01"
print(f"3. Simulating Trial {trial_id_att01} (Phase 1: 5 frames)")

for idx in range(1, 6):
    req = LandmarkRequest(
        landmarks=a_sample["landmarks"],
        mode="letters",
        expected_sign="A",
        session_id=session_id,
        trial_id=trial_id_att01,
        frame_idx=idx,
    )
    headers = {
        "x-conversion-mode": "letters",
        "x-expected-sign": "A",
        "x-session-id": session_id,
        "x-trial-id": trial_id_att01,
        "x-frame-idx": str(idx),
    }
    res_lm = recognize_landmarks(req, MockRequest(headers))
    assert res_lm["success"] is True

# Test 5: Pause simulation (zero frames sent for trial)
print("4. Simulating Pause: no frames sent for trial...")
time.sleep(0.05)

# Test 6: Resume simulation (Phase 2: frames 6 to 10 with SAME trial_id and continuous frame_idx)
print(f"5. Simulating Resume: sending frames 6 to 10 with same {trial_id_att01}...")
for idx in range(6, 11):
    req = LandmarkRequest(
        landmarks=a_sample["landmarks"],
        mode="letters",
        expected_sign="A",
        session_id=session_id,
        trial_id=trial_id_att01,
        frame_idx=idx,
    )
    headers = {
        "x-conversion-mode": "letters",
        "x-expected-sign": "A",
        "x-session-id": session_id,
        "x-trial-id": trial_id_att01,
        "x-frame-idx": str(idx),
    }
    res_lm = recognize_landmarks(req, MockRequest(headers))
    assert res_lm["success"] is True

# Test 7: Flush after att01
print("6. Calling /api/pilot/flush...")
res_flush = flush_pilot_log()
print("   Flush response:", res_flush)
assert res_flush["success"] is True
assert res_flush["stats"]["logged_count"] >= 10

# Test 8: Read and verify JSONL records for att01
print("7. Verifying JSONL records on disk...")
with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

att01_records = [r for r in lines if r.get("trial_id") == trial_id_att01]
print(f"   Found {len(att01_records)} records for {trial_id_att01}")
assert len(att01_records) == 10, f"Expected 10 frames, got {len(att01_records)}"

frame_indices = [r["frame_idx"] for r in att01_records]
print("   Frame indices:", frame_indices)
assert frame_indices == list(range(1, 11)), f"Frame indices non-monotonic: {frame_indices}"

print("   Record keys:", list(att01_records[0].keys()))
for r in att01_records:
    assert r["session_id"] == session_id
    assert r["trial_id"] == trial_id_att01
    assert r["expected_sign"] == "A"
    assert "geometric" in r
    assert "static_head" in r
    assert "final_sign" in r
    assert "prediction_source" in r
    assert "confirmed" in r

# Test 9: Simulate Retest -> att02
trial_id_att02 = f"{session_id}_A_rep01_att02"
print(f"8. Simulating Retest Trial {trial_id_att02} (frames 1 to 5)...")
for idx in range(1, 6):
    req = LandmarkRequest(
        landmarks=a_sample["landmarks"],
        mode="letters",
        expected_sign="A",
        session_id=session_id,
        trial_id=trial_id_att02,
        frame_idx=idx,
    )
    headers = {
        "x-conversion-mode": "letters",
        "x-expected-sign": "A",
        "x-session-id": session_id,
        "x-trial-id": trial_id_att02,
        "x-frame-idx": str(idx),
    }
    recognize_landmarks(req, MockRequest(headers))

flush_pilot_log()

with open(filepath, "r", encoding="utf-8") as f:
    all_lines = [json.loads(line) for line in f if line.strip()]

att02_records = [r for r in all_lines if r.get("trial_id") == trial_id_att02]
print(f"   Found {len(att02_records)} records for {trial_id_att02}")
assert len(att02_records) == 5

# Verify att01 is intact (not overwritten)
att01_still = [r for r in all_lines if r.get("trial_id") == trial_id_att01]
assert len(att01_still) == 10, "Retest corrupted previous attempt records!"
print("   Retest verification passed: att01 and att02 coexist cleanly.")

# Test 10: Close session
print("9. Disabling pilot logging session...")
res_disable = control_pilot_logging(PilotControlRequest(enabled=False))
print("   Disabled:", res_disable)

# Test 11: Safety check on duplicate session_id
print("10. Testing duplicate session ID handling...")
res_reopen = control_pilot_logging(PilotControlRequest(enabled=True, session_id=session_id))
new_filepath = res_reopen["filepath"]
print(f"    Original filepath: {filepath}")
print(f"    New filepath:      {new_filepath}")
assert new_filepath != filepath, "Expected safe collision-proof filename, but got identical filepath!"
assert os.path.exists(new_filepath)
control_pilot_logging(PilotControlRequest(enabled=False))

print("\n>>> ALL SMOKE TEST ASSERTIONS PASSED SUCCESSFULLY! <<<")
