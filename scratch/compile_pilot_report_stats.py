import json
from collections import Counter, defaultdict

new_file = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"
old_file = "backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl"

with open(new_file, "r", encoding="utf-8") as f:
    new_lines = [json.loads(line) for line in f]

controlled_frames = [d for d in new_lines if d.get("session_id") is not None]
idle_frames = [d for d in new_lines if d.get("session_id") is None]

print("=" * 70)
print("1. SUMMARY OF NEW PILOT TELEMETRY (pilot_pilot_20261108_0321.jsonl)")
print("=" * 70)
print(f"Total lines: {len(new_lines)}")
print(f"Controlled frames: {len(controlled_frames)}")
print(f"Idle frames: {len(idle_frames)}")

# Per-class stats in controlled frames
for cls in ["H", "K", "V"]:
    c_frames = [d for d in controlled_frames if d.get("expected_sign") == cls]
    total = len(c_frames)
    correct = sum(1 for d in c_frames if d.get("final_sign") == cls)
    neutral = sum(1 for d in c_frames if d.get("final_sign") == "NEUTRAL")
    wrong = total - correct - neutral
    wrong_dist = Counter(d.get("final_sign") for d in c_frames if d.get("final_sign") not in [cls, "NEUTRAL"])
    src_dist = Counter(d.get("prediction_source") for d in c_frames)
    acc = (correct / total) * 100 if total else 0
    cov = ((total - neutral) / total) * 100 if total else 0
    print(f"\n--- Class {cls} ---")
    print(f"  Total frames: {total}")
    print(f"  Correct ({cls}): {correct} ({acc:.2f}%)")
    print(f"  NEUTRAL: {neutral} ({neutral/total*100:.2f}%)")
    print(f"  Incorrect: {wrong} ({wrong/total*100:.2f}%) -> {dict(wrong_dist)}")
    print(f"  Coverage: {cov:.2f}%")
    print(f"  Sources: {dict(src_dist)}")

# Per-trial stats
print("\n" + "=" * 70)
print("2. PER-TRIAL BREAKDOWN (CONTROLLED)")
print("=" * 70)
trials = defaultdict(list)
for d in controlled_frames:
    trials[d.get("trial_id")].append(d)

print(f"{'Trial ID':<35} | {'Exp':<3} | {'Total':<5} | {'Correct':<7} | {'NEUTRAL':<7} | {'Wrong':<5} | {'Wrong Sign':<12}")
print("-" * 85)
for tr_id, fr_list in sorted(trials.items()):
    exp = fr_list[0].get("expected_sign")
    tot = len(fr_list)
    cor = sum(1 for d in fr_list if d.get("final_sign") == exp)
    neu = sum(1 for d in fr_list if d.get("final_sign") == "NEUTRAL")
    wro = tot - cor - neu
    wro_signs = Counter(d.get("final_sign") for d in fr_list if d.get("final_sign") not in [exp, "NEUTRAL"])
    wro_str = ", ".join(f"{s}:{c}" for s, c in wro_signs.items()) if wro_signs else "-"
    print(f"{tr_id:<35} | {exp:<3} | {tot:<5} | {cor:<7} | {neu:<7} | {wro:<5} | {wro_str:<12}")

# Idle stats
print("\n" + "=" * 70)
print("3. IDLE FRAMES ANALYSIS (session_id = None)")
print("=" * 70)
idle_signs = Counter(d.get("final_sign") for d in idle_frames)
idle_src = Counter(d.get("prediction_source") for d in idle_frames)
print(f"Total idle frames: {len(idle_frames)}")
print(f"Idle signs distribution: {dict(idle_signs)}")
print(f"Idle prediction sources: {dict(idle_src)}")

# Check H reason in idle frames
h_idle_reasons = Counter(d.get("geometric", {}).get("reason") for d in idle_frames if d.get("final_sign") == "H")
print(f"Idle H reasons: {dict(h_idle_reasons)}")

