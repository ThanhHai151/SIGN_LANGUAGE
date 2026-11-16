"""
Simulate exact recognizer smoothing buffer step by step across all 142 records
"""

import json
from collections import deque, Counter

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

smoothing_window = 10
confidence_threshold = 0.70
cooldown_sec = 1.2

history = deque(maxlen=smoothing_window)
last_confirmed_sign = None
last_confirmed_time = 0.0

print(f"{'Line':<5} | {'TrialID':<30} | {'InstantSign':<12} | {'Conf':<6} | {'LoggedConf':<11} | {'BufferVotes':<25} | {'MostCommon':<12} | {'CooldownOK':<11} | {'SimConf':<8}")
print("-" * 135)

for i, r in enumerate(records):
    ts = r.get("timestamp")
    trial_id = r.get("trial_id") or "IDLE"
    sign = r.get("final_sign")
    conf = r.get("final_confidence")
    logged_conf = r.get("confirmed")
    
    # Recognizer appends to history before or after smoothing?
    # Let's check recognizer.py: lines 1577: self.prediction_history.append((final_sign, final_conf))
    history.append((sign, conf))
    
    recent_signs = [s for s, c in history if c >= confidence_threshold]
    sim_confirmed = False
    most_common = None
    cooldown_ok = False
    
    if len(recent_signs) >= smoothing_window // 2:
        mc, count = Counter(recent_signs).most_common(1)[0]
        most_common = f"{mc}:{count}"
        if count >= (smoothing_window // 2) and mc != "NEUTRAL":
            if mc != last_confirmed_sign or (ts - last_confirmed_time > cooldown_sec):
                cooldown_ok = True
                sim_confirmed = True
                last_confirmed_sign = mc
                last_confirmed_time = ts
                
    if logged_conf or sim_confirmed or "NEUTRAL" in sign or i in (0, 1, 9, 10, 11, 12, 136, 137, 138, 139, 140, 141):
        votes = dict(Counter(recent_signs))
        print(f"{i:<5} | {trial_id:<30} | {sign:<12} | {conf:<6.2f} | {str(logged_conf):<11} | {str(votes):<25} | {str(most_common):<12} | {str(cooldown_ok):<11} | {str(sim_confirmed):<8}")

