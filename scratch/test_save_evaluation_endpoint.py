# scratch/test_save_evaluation_endpoint.py
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import tempfile
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

mock_report = {
    "session_id": "test_telemetry_patch_001",
    "total_trials": 1,
    "pass_rate": 100.0,
    "summary": {"tested": "Q"}
}

mock_frames = [
    {
        "frame_idx": 1,
        "expected_sign": "Q",
        "predicted_sign": "Q",
        "confidence": 0.78,
        "timestamp": 1234.56,
        "is_match": True,
        "landmarks": None,
        "hand_detected": True,
        "prediction_source": "static_head",
        "gating_status": "passed_supplemental",
        "classifier_status": "MATCH",
        "reject_reason": None,
        "head_confidence": 0.78,
        "confirmed": False,
        "new_token": None,
        "phase": "STABLE_HOLD",
        "elapsed_sec": 2.5
    },
    {
        "frame_idx": 2,
        "expected_sign": "Q",
        "predicted_sign": "NEUTRAL",
        "confidence": 0.0,
        "timestamp": 1234.76,
        "is_match": False,
        "landmarks": None,
        "hand_detected": False,
        "prediction_source": None,
        "gating_status": None,
        "classifier_status": None,
        "reject_reason": None,
        "head_confidence": None,
        "confirmed": False,
        "new_token": None,
        "phase": "STABLE_HOLD",
        "elapsed_sec": 2.7
    }
]

resp = client.post("/api/pilot/save_evaluation", json={
    "session_id": "test_telemetry_patch_001",
    "report": mock_report,
    "frames": mock_frames
})

assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
data = resp.json()
assert data.get("success") is True, f"Expected success=True, got {data}"
json_file = data.get("filename")
assert json_file is not None, "Filename must be returned"

diag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "data", "diagnostics")
json_path = os.path.join(diag_dir, json_file)
assert os.path.exists(json_path), f"JSON file must exist at {json_path}"

# Find the generated JSONL
jsonl_file = json_file.replace(".json", "_frames.jsonl")
jsonl_path = os.path.join(diag_dir, jsonl_file)
assert os.path.exists(jsonl_path), f"JSONL file must exist at {jsonl_path}"

# Read JSONL lines
with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

assert len(lines) == 2, f"Expected 2 frame lines, got {len(lines)}"
f1 = lines[0]
assert f1["hand_detected"] is True
assert f1["prediction_source"] == "static_head"
assert f1["classifier_status"] == "MATCH"
assert f1["head_confidence"] == 0.78
assert f1["phase"] == "STABLE_HOLD"

f2 = lines[1]
assert f2["hand_detected"] is False
assert f2["prediction_source"] is None
assert f2["phase"] == "STABLE_HOLD"

# Clean up test files
os.remove(json_path)
os.remove(jsonl_path)
print("SUCCESS: /api/pilot/save_evaluation end-to-end verified with new telemetry fields!")
