import os
import sys
import time
import json
import statistics

# Add workspace to path
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.app import app, recognizer

print("=== STARTING INFERENCE THROUGHPUT BENCHMARK ===")

# Create sample 21 landmarks
sample_landmarks = [[0.5 + 0.01 * (i % 3), 0.5 + 0.01 * (i // 3), 0.0] for i in range(21)]

# 1. Benchmark Pure Python In-Process Recognizer
print("\n--- 1. In-Process Recognizer (predict_landmarks) ---")
in_proc_times = []
for i in range(100):
    t0 = time.perf_counter()
    res = recognizer.predict_landmarks(
        landmarks_list=sample_landmarks,
        mode="letters",
        session_id="bench_sess",
        trial_id="bench_trial_1",
        frame_idx=i,
        timestamp=time.time()
    )
    t1 = time.perf_counter()
    in_proc_times.append((t1 - t0) * 1000)

print(f"Count: {len(in_proc_times)}")
print(f"Mean Latency: {statistics.mean(in_proc_times):.3f} ms")
print(f"p50: {statistics.median(in_proc_times):.3f} ms")
print(f"p95: {sorted(in_proc_times)[int(len(in_proc_times)*0.95)]:.3f} ms")
print(f"Effective In-Process FPS: {1000.0 / statistics.mean(in_proc_times):.1f} FPS")

# 2. Benchmark FastAPI TestClient (/api/recognize_landmarks)
print("\n--- 2. HTTP Endpoint via TestClient (/api/recognize_landmarks) ---")
client = TestClient(app)

payload = {
    "landmarks": sample_landmarks,
    "multi_hands": [{"handedness": "Right", "landmarks": sample_landmarks}],
    "handedness": "Right",
    "mode": "letters",
    "expected_sign": "Â",
    "session_id": "bench_http_sess",
    "trial_id": "bench_trial_1",
    "frame_idx": 1,
    "timestamp": time.time(),
    "phase": "STABLE_HOLD"
}

# Warmup 5 requests
for _ in range(5):
    client.post("/api/recognize_landmarks", json=payload)

http_times = []
payload_bytes = len(json.dumps(payload).encode("utf-8"))
response_bytes_list = []

for i in range(100):
    payload["frame_idx"] = i
    payload["timestamp"] = time.time()
    t0 = time.perf_counter()
    resp = client.post("/api/recognize_landmarks", json=payload)
    t1 = time.perf_counter()
    http_times.append((t1 - t0) * 1000)
    response_bytes_list.append(len(resp.content))

print(f"Count: {len(http_times)}")
print(f"Request Payload Size: {payload_bytes} bytes")
print(f"Response Payload Size: {statistics.mean(response_bytes_list):.1f} bytes")
print(f"Mean HTTP Latency: {statistics.mean(http_times):.3f} ms")
print(f"p50: {statistics.median(http_times):.3f} ms")
print(f"p95: {sorted(http_times)[int(len(http_times)*0.95)]:.3f} ms")
print(f"Throughput: {1000.0 / statistics.mean(http_times):.1f} requests/sec")
