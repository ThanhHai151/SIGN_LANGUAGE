import sys
sys.path.insert(0, '.')
import os
import json
import collections
from backend.vision.alphabet_classifier import (
    P_DOWN_D_IW_MIN, P_DOWN_IDX_STRAIGHT_MIN, P_DOWN_IDX_TIP_Y_MIN,
    P_DOWN_IDX_DY_MIN, P_DOWN_IDX_HORIZ_MAX, P_DOWN_D_MW_MIN,
    P_DOWN_MID_STRAIGHT_MIN, P_DOWN_MID_TIP_Y_MIN, P_DOWN_D_TM_MAX,
    P_DOWN_D_TI_MAX, P_DOWN_D_TW_MIN, P_DOWN_ANG_IM_MIN,
    P_DOWN_ANG_IM_MAX, P_DOWN_D_RW_MAX, P_DOWN_D_PW_MAX
)

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

active_neutrals = [r for r in lines if r.get("expected_sign") == "P" and r.get("final_sign") == "NEUTRAL"]

print(f"Total active neutral frames: {len(active_neutrals)}")

def check_p_down_failures(feat):
    fails = []
    if float(feat.get("d_IW", 0.0)) < P_DOWN_D_IW_MIN:
        fails.append(f"d_IW={float(feat.get('d_IW', 0)):.2f} < {P_DOWN_D_IW_MIN}")
    if float(feat.get("idx_straight", 0.0)) < P_DOWN_IDX_STRAIGHT_MIN:
        fails.append(f"idx_straight={float(feat.get('idx_straight', 0)):.2f} < {P_DOWN_IDX_STRAIGHT_MIN}")
    if float(feat.get("idx_tip_y", 0.0)) < P_DOWN_IDX_TIP_Y_MIN:
        fails.append(f"idx_tip_y={float(feat.get('idx_tip_y', 0)):.2f} < {P_DOWN_IDX_TIP_Y_MIN}")
    if float(feat.get("idx_dy", 0.0)) < P_DOWN_IDX_DY_MIN:
        fails.append(f"idx_dy={float(feat.get('idx_dy', 0)):.2f} < {P_DOWN_IDX_DY_MIN}")
    if float(feat.get("idx_horiz_ratio", 0.0)) > P_DOWN_IDX_HORIZ_MAX:
        fails.append(f"idx_horiz={float(feat.get('idx_horiz_ratio', 0)):.2f} > {P_DOWN_IDX_HORIZ_MAX}")
    if float(feat.get("d_MW", 0.0)) < P_DOWN_D_MW_MIN:
        fails.append(f"d_MW={float(feat.get('d_MW', 0)):.2f} < {P_DOWN_D_MW_MIN}")
    if float(feat.get("mid_straight", 0.0)) < P_DOWN_MID_STRAIGHT_MIN:
        fails.append(f"mid_straight={float(feat.get('mid_straight', 0)):.2f} < {P_DOWN_MID_STRAIGHT_MIN}")
    if float(feat.get("mid_tip_y", 0.0)) < P_DOWN_MID_TIP_Y_MIN:
        fails.append(f"mid_tip_y={float(feat.get('mid_tip_y', 0)):.2f} < {P_DOWN_MID_TIP_Y_MIN}")
    if float(feat.get("d_TM", 99.0)) > P_DOWN_D_TM_MAX:
        fails.append(f"d_TM={float(feat.get('d_TM', 0)):.2f} > {P_DOWN_D_TM_MAX}")
    if float(feat.get("d_TI", 99.0)) > P_DOWN_D_TI_MAX:
        fails.append(f"d_TI={float(feat.get('d_TI', 0)):.2f} > {P_DOWN_D_TI_MAX}")
    if float(feat.get("d_TW", 0.0)) < P_DOWN_D_TW_MIN:
        fails.append(f"d_TW={float(feat.get('d_TW', 0)):.2f} < {P_DOWN_D_TW_MIN}")
    ang = float(feat.get("ang_IM_dir", 0.0))
    if ang < P_DOWN_ANG_IM_MIN or ang > P_DOWN_ANG_IM_MAX:
        fails.append(f"ang_IM={ang:.1f} not in [{P_DOWN_ANG_IM_MIN}, {P_DOWN_ANG_IM_MAX}]")
    if float(feat.get("d_RW", 99.0)) > P_DOWN_D_RW_MAX:
        fails.append(f"d_RW={float(feat.get('d_RW', 0)):.2f} > {P_DOWN_D_RW_MAX}")
    if float(feat.get("d_PW", 99.0)) > P_DOWN_D_PW_MAX:
        fails.append(f"d_PW={float(feat.get('d_PW', 0)):.2f} > {P_DOWN_D_PW_MAX}")
    return fails

failure_counts = collections.Counter()
for r in active_neutrals:
    feats = r.get("geometric", {}).get("features", {})
    if not feats:
        failure_counts["NO_FEATURES / INVALID_HAND"] += 1
        continue
    f_list = check_p_down_failures(feats)
    for f in f_list:
        key = f.split("=")[0] if "=" in f else f.split(" ")[0]
        failure_counts[key] += 1

print("\nFrequency of failed P-down conditions across 46 neutral frames:")
for k, cnt in failure_counts.most_common():
    print(f"  {k:15s}: {cnt}/{len(active_neutrals)} ({cnt/len(active_neutrals)*100:.1f}%)")

print("\nDetail of first 10 neutral frames:")
for idx, r in enumerate(active_neutrals[:10]):
    feats = r.get("geometric", {}).get("features", {})
    fl = check_p_down_failures(feats) if feats else ["NO_FEATURES"]
    print(f"  Trial: {r.get('trial_id')}, f_idx: {r.get('frame_idx')}, t: {r.get('timestamp')}: {fl}")

print("\nDetail of rep09 exit frames (23 frames):")
rep09_neutrals = [r for r in active_neutrals if "rep09" in r.get("trial_id", "")]
for r in rep09_neutrals[:5]:
    feats = r.get("geometric", {}).get("features", {})
    fl = check_p_down_failures(feats) if feats else ["NO_FEATURES"]
    print(f"  rep09 f_idx: {r.get('frame_idx')}, t: {r.get('timestamp')}: {fl}")
