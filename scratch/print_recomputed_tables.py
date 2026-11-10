import json

d = json.load(open("scratch/task12_deep_recomputation_summary.json", encoding="utf-8"))
stats = d["letter_frame_stats"]

print("=========================================================================================================================")
print(f"{'Letter':<6}{'Trials':^8}{'TrCor':^7}{'TrAcc%':^9}{'Att01%':^9}{'Att02%':^9}{'Frames':^8}{'FrCor':^7}{'FrAcc%':^9}{'Neutral':^9}{'Incorrect':^11}{'MeanCor':^9}{'MedCor':^9}{'TopConfusion':<20}")
print("=========================================================================================================================")
for l, s in stats.items():
    top_c = ", ".join(f"{k}({v})" for k, v in list(s["top_confusions"].items())[:2]) or "-"
    print(f"{l:<6}{s['trials_total']:^8}{s['trials_correct']:^7}{s['trial_accuracy_pct']:^9.1f}{s['att01_accuracy_pct']:^9.1f}{s['att02_accuracy_pct']:^9.1f}{s['total_frames']:^8}{s['correct_frames']:^7}{s['frame_accuracy_pct']:^9.1f}{s['neutral_frames']:^9}{s['incorrect_frames']:^11}{s['conf_mean_correct']:^9.3f}{s['conf_median_correct']:^9.3f}{top_c:<20}")

print("\n--- RANKINGS BY TRIAL ACCURACY ---")
ranked_trial = sorted(stats.items(), key=lambda x: (x[1]["trial_accuracy_pct"], x[1]["frame_accuracy_pct"]), reverse=True)
for rank, (l, s) in enumerate(ranked_trial, 1):
    print(f"{rank:>2}. Letter {l:<2}: Trial Acc = {s['trial_accuracy_pct']:>5.1f}% ({s['trials_correct']}/{s['trials_total']}) | Frame Acc = {s['frame_accuracy_pct']:>5.1f}% ({s['correct_frames']}/{s['total_frames']}) | Confirmed Trials = {s['confirmed_trials']}/{s['trials_total']}")

print("\n--- RANKINGS BY FRAME ACCURACY ---")
ranked_frame = sorted(stats.items(), key=lambda x: (x[1]["frame_accuracy_pct"], x[1]["trial_accuracy_pct"]), reverse=True)
for rank, (l, s) in enumerate(ranked_frame, 1):
    print(f"{rank:>2}. Letter {l:<2}: Frame Acc = {s['frame_accuracy_pct']:>5.1f}% ({s['correct_frames']}/{s['total_frames']}) | Trial Acc = {s['trial_accuracy_pct']:>5.1f}% ({s['trials_correct']}/{s['trials_total']})")

