import json

with open("scratch/task3_8_root_cause_summary.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== 1. SESSIONS COMPARISON ===")
for k, v in d["sessions_stats"].items():
    s_idx = v["idx_straight"]
    s_dti = v["d_TI"]
    print(f"{k}:")
    print(f"   idx_straight: mean={s_idx['mean']}, med={s_idx['median']}, [p25={s_idx['p25']}, p75={s_idx['p75']}], min={s_idx['min']}, max={s_idx['max']}, std={s_idx['std']}")
    print(f"   d_TI        : mean={s_dti['mean']}, med={s_dti['median']}, [p25={s_dti['p25']}, p75={s_dti['p75']}], min={s_dti['min']}, max={s_dti['max']}, std={s_dti['std']}")

print("\n=== 2. TRIALS STATS (pilot_20261109_004312) ===")
for tid, v in d["trials_stats_004312"].items():
    s_idx = v["idx_straight"]
    s_dti = v["d_TI"]
    print(f"{tid}: N={v['total_frames']}, Matched={v['matched_frames']} ({v['match_rate']*100:.1f}%)")
    print(f"   idx_straight: mean={s_idx['mean']}, med={s_idx['median']}, [p25={s_idx['p25']}, p75={s_idx['p75']}], min={s_idx['min']}, max={s_idx['max']}")
    print(f"   d_TI        : mean={s_dti['mean']}, med={s_dti['median']}, [p25={s_dti['p25']}, p75={s_dti['p75']}], min={s_dti['min']}, max={s_dti['max']}")

print("\n=== 3. CONTRAST CLASSES (NPZ) ===")
for k, v in d["contrast_classes_npz"].items():
    s_idx = v["idx_straight"]
    s_dti = v["d_TI"]
    s_dtm = v["d_TM"]
    s_dtw = v["d_TW"]
    print(f"{k} (N={s_idx['N']}):")
    print(f"   idx_straight: mean={s_idx['mean']}, med={s_idx['median']}, min={s_idx['min']}, max={s_idx['max']}")
    print(f"   d_TI        : mean={s_dti['mean']}, med={s_dti['median']}, min={s_dti['min']}, max={s_dti['max']}")
    print(f"   d_TM        : mean={s_dtm['mean']}, med={s_dtm['median']}, min={s_dtm['min']}, max={s_dtm['max']}")
    print(f"   d_TW        : mean={s_dtw['mean']}, med={s_dtw['median']}, min={s_dtw['min']}, max={s_dtw['max']}")

print("\n=== 4. B FRAME 54 OF D0492 ===")
b54 = d["b_frame_54"]
print(f"Video {b54['video_id']}, Frame {b54['frame_index']}, Time {b54['timestamp_sec']}s, Label: {b54['true_label']}")
print(f"Features: {json.dumps(b54['features'], indent=2)}")
print(f"Rule O Eval: {json.dumps(b54['rule_o_evaluation'], indent=2)}")

