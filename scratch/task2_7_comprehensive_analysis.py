"""
Comprehensive Analysis Script for TASK 2.7 Webcam Holdout Verification
"""

import sys
import os
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np

FILE_PATH = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"

# Candidate 2B Thresholds
THRESHOLDS = {
    "d_ti_le_0.25": lambda f: f.get("d_TI") <= 0.25,
    "d_iw_le_1.55": lambda f: f.get("d_IW") <= 1.55,
    "i_curl_ge_0.15": lambda f: f.get("index_curl") >= 0.15,
    "idx_straight_ge_0.40": lambda f: f.get("idx_straight") >= 0.40,
    "idx_straight_le_0.85": lambda f: f.get("idx_straight") <= 0.85,
    "d_rw_le_0.70": lambda f: f.get("d_RW") <= 0.70,
    "d_pw_le_0.65": lambda f: f.get("d_PW") <= 0.65,
    "d_tw_ge_0.70": lambda f: f.get("d_TW") >= 0.70,
    "d_tw_le_1.30": lambda f: f.get("d_TW") <= 1.30,
    "idx_tip_y_le_-0.20": lambda f: f.get("idx_tip_y") <= -0.20,
}

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    file_size = os.path.getsize(FILE_PATH)
    file_sha256 = sha256_file(FILE_PATH)

    print("=" * 80)
    print("TASK 2.7 COMPREHENSIVE DATA & RECOGNITION AUDIT")
    print("=" * 80)
    print(f"File: {FILE_PATH}")
    print(f"Size: {file_size:,} bytes")
    print(f"SHA256: {file_sha256}")
    print(f"Total Records: {len(records)}")

    # 1. Classify records
    active_records = []
    idle_records = []
    for r in records:
        if r.get("trial_id") is not None and r.get("expected_sign") == "X":
            active_records.append(r)
        else:
            idle_records.append(r)

    print(f"\n1. RECORD BREAKDOWN:")
    print(f"- Active Trial Records: {len(active_records)}")
    print(f"- Idle Records: {len(idle_records)}")

    # Check timestamps and continuity
    timestamps = [r["timestamp"] for r in records]
    dt_list = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    print(f"- Start timestamp: {timestamps[0]:.4f}")
    print(f"- End timestamp: {timestamps[-1]:.4f}")
    print(f"- Total session duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"- Min dt: {min(dt_list):.4f}s, Max dt: {max(dt_list):.4f}s, Mean dt: {np.mean(dt_list):.4f}s")

    # Group by trial
    trials = defaultdict(list)
    for r in active_records:
        trials[r["trial_id"]].append(r)

    print(f"\n2. TRIAL ANALYSIS (10 Trials):")
    # Kinematics partitioning:
    # In each trial, define stable hold vs transition.
    # If trial has >= 9 frames:
    # boundary: frame 0 is entry, last frame is exit; or check motion stability.
    # In Task 2.2, first 3 and last 3 of 28-30 frames trials were transition (6/28-30 = ~20%).
    # Here trials have 9-12 frames (recorded at ~2-3 FPS over 4-5s duration).
    # Let's inspect per-frame speed and features.
    
    trial_stats = []
    all_stable_frames = []
    all_trans_frames = []

    for tid in sorted(trials.keys()):
        t_frames = trials[tid]
        n_f = len(t_frames)
        t_start = t_frames[0]["timestamp"]
        t_end = t_frames[-1]["timestamp"]
        t_dur = t_end - t_start
        
        # Determine transition vs stable
        # For a ~10-frame trial: frame 0 is entry transition (1 frame), frame -1 is exit transition (1 frame)
        # Or frames 1..n-2 are stable hold.
        # Let's check boundary frames:
        trans_indices = {0, n_f - 1} if n_f >= 5 else set()
        
        stable_t = []
        trans_t = []
        first_conf_time = None
        
        for idx_in_trial, f in enumerate(t_frames):
            if f.get("confirmed") and first_conf_time is None:
                first_conf_time = f["timestamp"] - t_start
                
            if idx_in_trial in trans_indices:
                trans_t.append(f)
                all_trans_frames.append(f)
            else:
                stable_t.append(f)
                all_stable_frames.append(f)

        s_correct = sum(1 for f in stable_t if f["final_sign"] == "X")
        s_acc = (s_correct / len(stable_t) * 100) if stable_t else 0.0
        
        t_correct = sum(1 for f in t_frames if f["final_sign"] == "X")
        t_acc = (t_correct / n_f * 100)
        
        conf_count = sum(1 for f in t_frames if f.get("confirmed"))
        
        trial_stats.append({
            "trial_id": tid,
            "total_frames": n_f,
            "duration": t_dur,
            "stable_frames": len(stable_t),
            "trans_frames": len(trans_t),
            "stable_correct": s_correct,
            "stable_acc": s_acc,
            "active_correct": t_correct,
            "active_acc": t_acc,
            "confirmed_count": conf_count,
            "first_conf_time": first_conf_time,
            "preds": Counter(f["final_sign"] for f in t_frames)
        })

    print(f"{'Trial ID':<32} | {'Dur(s)':<6} | {'Tot':<4} | {'Stb':<4} | {'Trn':<4} | {'Stb Acc':<8} | {'Act Acc':<8} | {'Conf':<5} | {'1st Conf(s)':<11}")
    print("-" * 95)
    for ts in trial_stats:
        first_c = f"{ts['first_conf_time']:.2f}s" if ts['first_conf_time'] is not None else "None"
        print(f"{ts['trial_id']:<32} | {ts['duration']:<6.2f} | {ts['total_frames']:<4} | {ts['stable_frames']:<4} | {ts['trans_frames']:<4} | {ts['stable_acc']:<7.1f}% | {ts['active_acc']:<7.1f}% | {ts['confirmed_count']:<5} | {first_c:<11}")

    print("\n3. AGGREGATE METRICS:")
    print(f"- Total Active Frames: {len(active_records)}")
    print(f"  * X correct: {sum(1 for f in active_records if f['final_sign'] == 'X')} ({sum(1 for f in active_records if f['final_sign'] == 'X')/len(active_records)*100:.2f}%)")
    print(f"  * Neutral: {sum(1 for f in active_records if f['final_sign'] == 'NEUTRAL')}")
    print(f"  * False Positives to other letters: {sum(1 for f in active_records if f['final_sign'] not in ('X', 'NEUTRAL'))}")
    
    print(f"\n- Stable-Hold Frames: {len(all_stable_frames)}")
    print(f"  * X correct: {sum(1 for f in all_stable_frames if f['final_sign'] == 'X')} ({sum(1 for f in all_stable_frames if f['final_sign'] == 'X')/len(all_stable_frames)*100:.2f}%)")
    print(f"  * Neutral: {sum(1 for f in all_stable_frames if f['final_sign'] == 'NEUTRAL')}")
    print(f"  * False Positives: {sum(1 for f in all_stable_frames if f['final_sign'] not in ('X', 'NEUTRAL'))}")
    
    print(f"\n- Transition Frames: {len(all_trans_frames)}")
    print(f"  * X correct: {sum(1 for f in all_trans_frames if f['final_sign'] == 'X')} ({sum(1 for f in all_trans_frames if f['final_sign'] == 'X')/len(all_trans_frames)*100:.2f}%)")
    print(f"  * Neutral: {sum(1 for f in all_trans_frames if f['final_sign'] == 'NEUTRAL')}")
    print(f"  * False Positives: {sum(1 for f in all_trans_frames if f['final_sign'] not in ('X', 'NEUTRAL'))}")

    # Trial pass rate
    trials_pass_70 = sum(1 for ts in trial_stats if ts["active_acc"] >= 70.0)
    trials_confirmed = sum(1 for ts in trial_stats if ts["confirmed_count"] > 0)
    print(f"\n- Trial Pass Rate (>=70% Acc): {trials_pass_70} / {len(trial_stats)} ({trials_pass_70/len(trial_stats)*100:.1f}%)")
    print(f"- Trial Confirmation Rate: {trials_confirmed} / {len(trial_stats)} ({trials_confirmed/len(trial_stats)*100:.1f}%)")

    # 4. Feature distribution of Candidate 2B on holdout active frames
    print("\n4. CANDIDATE 2B FEATURE DISTRIBUTIONS ON HOLDOUT ACTIVE FRAMES (108 frames):")
    feat_names = ["d_TI", "d_IW", "index_curl", "idx_straight", "d_RW", "d_PW", "d_TW", "idx_tip_y"]
    print(f"{'Feature':<14} | {'Min':<8} | {'p25':<8} | {'Median':<8} | {'Mean':<8} | {'p75':<8} | {'Max':<8} | {'Threshold':<20} | {'Margin to Limit'}")
    print("-" * 115)
    for fn in feat_names:
        vals = [f["geometric"]["features"][fn] for f in active_records if fn in f["geometric"]["features"]]
        min_v = np.min(vals)
        p25_v = np.percentile(vals, 25)
        med_v = np.median(vals)
        mean_v = np.mean(vals)
        p75_v = np.percentile(vals, 75)
        max_v = np.max(vals)
        
        # Compare to candidate 2B limits
        if fn == "d_TI":
            th_str = "<= 0.25"
            margin = f"{0.25 - max_v:+.4f} (safe)"
        elif fn == "d_IW":
            th_str = "<= 1.55"
            margin = f"{1.55 - max_v:+.4f} (safe)"
        elif fn == "index_curl":
            th_str = ">= 0.15"
            margin = f"{min_v - 0.15:+.4f} (safe)"
        elif fn == "idx_straight":
            th_str = "[0.40, 0.85]"
            margin = f"[{min_v - 0.40:+.4f}, {0.85 - max_v:+.4f}] (safe)"
        elif fn == "d_RW":
            th_str = "<= 0.70"
            margin = f"{0.70 - max_v:+.4f} (safe)"
        elif fn == "d_PW":
            th_str = "<= 0.65"
            margin = f"{0.65 - max_v:+.4f} (safe)"
        elif fn == "d_TW":
            th_str = "[0.70, 1.30]"
            margin = f"[{min_v - 0.70:+.4f}, {1.30 - max_v:+.4f}] (safe)"
        elif fn == "idx_tip_y":
            th_str = "<= -0.20"
            margin = f"{-0.20 - max_v:+.4f} (safe)"
        else:
            th_str = "-"
            margin = "-"
            
        print(f"{fn:<14} | {min_v:<8.4f} | {p25_v:<8.4f} | {med_v:<8.4f} | {mean_v:<8.4f} | {p75_v:<8.4f} | {max_v:<8.4f} | {th_str:<20} | {margin}")

    # Check Idle frames features and predictions
    print("\n5. IDLE FRAMES AUDIT (34 frames):")
    idle_preds = Counter(f["final_sign"] for f in idle_records)
    print(f"- Predictions during idle/inter-trial: {dict(idle_preds)}")
    idle_conf = sum(1 for f in idle_records if f.get("confirmed"))
    print(f"- Tokens confirmed during idle: {idle_conf}")

if __name__ == "__main__":
    main()

