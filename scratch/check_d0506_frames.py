import json
import sys
import numpy as np

sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()
data = json.load(open("backend/data/alphabet_v2/static_landmarks/O.json", "r", encoding="utf-8"))
samples = [s for s in data["samples"] if s.get("landmarks")]

print(f"Total samples with landmarks: {len(samples)}")
results = []
for i, s in enumerate(samples):
    lm = s["landmarks"]
    f = clf.extract_features(lm)
    res = clf.classify_detailed(lm)
    matches = clf.evaluate_rules(f)
    is_o = any(m[0] == "O" for m in matches)
    r_letter, r_conf = clf.classify(lm)
    f_idx = s["frame_index"]
    ts = s.get("timestamp_sec", 0.0)
    
    rec = {
        "index": i,
        "frame_index": f_idx,
        "timestamp_sec": ts,
        "phase": s.get("phase"),
        "idx_straight": f["idx_straight"],
        "d_TI": f["d_TI"],
        "palm_facing_camera": f["palm_facing_camera"],
        "idx_upward": f["idx_upward"],
        "mid_upward": f["mid_upward"],
        "ring_upward": f["ring_upward"],
        "pinky_upward": f["pinky_upward"],
        "wrist_dist": f["wrist_dist"],
        "matches_o": is_o,
        "matches_all": matches,
        "classified_letter": r_letter,
        "confidence": r_conf
    }
    results.append(rec)
    print(f"[{i:02d}] frame={f_idx:02d} ts={ts:.3f}s phase={rec['phase']} | idx_str={f['idx_straight']:.4f} d_TI={f['d_TI']:.4f} | palm={rec['palm_facing_camera']} idx_up={rec['idx_upward']} mid_up={rec['mid_upward']} ring_up={rec['ring_upward']} pnk_up={rec['pinky_upward']} wrist_dist={rec['wrist_dist']:.4f} | match_O={is_o} class={r_letter} conf={r_conf}")

hold_indices = [22, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
hold_frames = [r for r in results if r["frame_index"] in hold_indices]
non_hold_frames = [r for r in results if r["frame_index"] not in hold_indices]

print("\n--- 12 HOLD FRAMES STATS ---")
print(f"Count: {len(hold_frames)}")
hold_idx_str = [r["idx_straight"] for r in hold_frames]
hold_d_ti = [r["d_TI"] for r in hold_frames]
print(f"idx_straight: min={np.min(hold_idx_str):.4f}, p25={np.percentile(hold_idx_str, 25):.4f}, med={np.median(hold_idx_str):.4f}, mean={np.mean(hold_idx_str):.4f}, p75={np.percentile(hold_idx_str, 75):.4f}, max={np.max(hold_idx_str):.4f}")
print(f"d_TI:         min={np.min(hold_d_ti):.4f}, p25={np.percentile(hold_d_ti, 25):.4f}, med={np.median(hold_d_ti):.4f}, mean={np.mean(hold_d_ti):.4f}, p75={np.percentile(hold_d_ti, 75):.4f}, max={np.max(hold_d_ti):.4f}")
print(f"Matches O: {sum(r['matches_o'] for r in hold_frames)}/{len(hold_frames)}")

print("\n--- 8 NON-HOLD FRAMES STATS ---")
print(f"Count: {len(non_hold_frames)}")
for r in non_hold_frames:
    print(f"frame={r['frame_index']} ts={r['timestamp_sec']:.3f} idx_str={r['idx_straight']:.4f} d_TI={r['d_TI']:.4f} matches_o={r['matches_o']} matches_all={r['matches_all']}")

print("\n--- ALL 20 FRAMES STATS ---")
all_idx_str = [r["idx_straight"] for r in results]
all_d_ti = [r["d_TI"] for r in results]
print(f"idx_straight: min={np.min(all_idx_str):.4f}, p25={np.percentile(all_idx_str, 25):.4f}, med={np.median(all_idx_str):.4f}, mean={np.mean(all_idx_str):.4f}, p75={np.percentile(all_idx_str, 75):.4f}, max={np.max(all_idx_str):.4f}")
print(f"d_TI:         min={np.min(all_d_ti):.4f}, p25={np.percentile(all_d_ti, 25):.4f}, med={np.median(all_d_ti):.4f}, mean={np.mean(all_d_ti):.4f}, p75={np.percentile(all_d_ti, 75):.4f}, max={np.max(all_d_ti):.4f}")
