import json

frames_path = "backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl"
with open(frames_path, "r", encoding="utf-8") as f:
    frames = [json.loads(line) for line in f]

print(f"Total frames: {len(frames)}")

for i, f in enumerate(frames):
    t = f.get("temporal", {})
    trial_id = f.get("trial_id")
    session_id = f.get("session_id")
    confirmed = f.get("confirmed")
    final_sign = f.get("final_sign")
    final_conf = f.get("final_confidence")
    elapsed = f.get("elapsed_sec") or f.get("sync", {}).get("client_capture_ts")
    hist = t.get("recent_predictions_10", [])
    last_sign = t.get("last_confirmed_sign")
    vote_cnt = t.get("vote_count")
    # print all confirmed or selected frames around rep 6, 7, 8, 9, 10
    if confirmed or i in [0, 8, 9, 19, 20, 28, 29, 38, 39, 49, 50, 58, 59, 67, 68, 69, 70, 71, 77, 78, 79, 87, 88]:
        print(f"Frame {i+1:2d} | trial: {trial_id} | sign: {final_sign} ({final_conf}) | conf: {confirmed} | vote: {vote_cnt} | last_conf: {last_sign} | hist_len: {len(hist)}")

