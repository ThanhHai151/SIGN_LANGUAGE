import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

letters = ["Ă", "Â", "Ê", "Đ", "Ơ", "Ư", "Ô"]
fps_targets = [None, 20, 15, 10, 5, 2] # None = native

results = {}

for let in letters:
    fpath = f"backend/data/alphabet_v2/motion_sequences/{let}.json"
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    seq = data["sequence"]
    
    results[let] = {}
    
    for fps in fps_targets:
        fps_label = "native" if fps is None else f"{fps}fps"
        
        # Subsample by timestamp
        if fps is None:
            sub_seq = seq
        else:
            min_dt = 1.0 / fps
            sub_seq = []
            last_t = -1.0
            for item in seq:
                t = float(item.get("timestamp_sec", 0.0))
                if last_t < 0 or (t - last_t) >= (min_dt * 0.90): # small tolerance
                    sub_seq.append(item)
                    last_t = t
        
        session_id = f"replay_{let}_{fps_label}"
        rec.reset_diacritic_fsm(session_id)
        
        base_stable = None
        candidate = None
        confirmed = None
        confirmed_token = None
        confirmed_ts = None
        reject_reasons = []
        preds = []
        
        for item in sub_seq:
            lms = item.get("landmarks")
            t = float(item.get("timestamp_sec", 0.0))
            frame_idx = item.get("frame_index")
            
            res = rec.predict_landmarks(lms, mode="letters", session_id=session_id, timestamp=t)
            sign = res.get("sign")
            token = res.get("new_token")
            is_conf = res.get("confirmed")
            src = res.get("source")
            
            fsm = rec.diacritic_fsms.get(session_id)
            if fsm:
                if base_stable is None and fsm.state in ("BASE_STABLE", "DIACRITIC_CANDIDATE", "COOLDOWN"):
                    base_stable = (frame_idx, fsm.current_base, t)
                if candidate is None and fsm.state == "DIACRITIC_CANDIDATE":
                    candidate = (frame_idx, fsm.diacritic_candidate, t)
                if confirmed is None and (fsm.state == "COOLDOWN" or sign == let or token == let):
                    confirmed = (frame_idx, sign, t, src)
                    confirmed_token = token
                    confirmed_ts = t
            
            if sign != "NEUTRAL":
                preds.append(sign)
        
        results[let][fps_label] = {
            "total_frames": len(sub_seq),
            "base_stable": base_stable,
            "candidate": candidate,
            "confirmed": confirmed,
            "confirmed_token": confirmed_token,
            "confirmed_ts": confirmed_ts,
            "is_success": confirmed is not None and (confirmed[1] == let or confirmed_token == let),
            "predictions_summary": {p: preds.count(p) for p in set(preds)}
        }

print("=" * 100)
print(f"{'Letter':<6} | {'FPS':<8} | {'Frames':<8} | {'Base Stable':<18} | {'Candidate':<15} | {'Confirmed':<15} | {'Status'}")
print("=" * 100)

for let in letters:
    for fps in fps_targets:
        fps_label = "native" if fps is None else f"{fps}fps"
        r = results[let][fps_label]
        b_str = f"f{r['base_stable'][0]}:{r['base_stable'][1]}" if r['base_stable'] else "None"
        c_str = f"f{r['candidate'][0]}:{r['candidate'][1]}" if r['candidate'] else "None"
        cf_str = f"f{r['confirmed'][0]}:{r['confirmed'][1]}" if r['confirmed'] else "None"
        status = "CONFIRMED" if r['is_success'] else "FAIL"
        print(f"{let:<6} | {fps_label:<8} | {r['total_frames']:<8} | {b_str:<18} | {c_str:<15} | {cf_str:<15} | {status}")
    print("-" * 100)

