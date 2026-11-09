import json

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

r_matched_h_frames = []

with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("expected_sign") == "H" and d.get("final_sign") == "R":
            r_matched_h_frames.append((line_idx, d))

print(f"Total H frames predicted as R: {len(r_matched_h_frames)}")

if r_matched_h_frames:
    line_idx, sample = r_matched_h_frames[0]
    print(f"\n--- Sample Frame Line {line_idx} ---")
    print(f"Trial: {sample.get('trial_id')} | frame_idx: {sample.get('frame_idx')}")
    geom = sample.get("geometric", {})
    print(f"Geom reason: {geom.get('reason')}")
    print(f"Matched candidates: {geom.get('matched_candidates')}")
    feats = geom.get("features", {})
    r_diag = geom.get("r_rule_diagnostics", {})
    h_diag = geom.get("h_rule_diagnostics", {})
    print("\nFeatures:")
    for k in ["idx_straight", "mid_straight", "idx_horiz_ratio", "mid_horiz_ratio", "ang_IM_dir", "d_IW", "d_MW", "d_RW", "d_PW", "cross_indicator", "mcp_width_2d", "thumb_ring_dist", "diff_mw_iw"]:
        print(f"  {k}: {feats.get(k)}")
    print("\nR rule diagnostics:")
    print(json.dumps(r_diag, indent=2))
    print("\nH rule diagnostics (h_webcam):")
    print(json.dumps(h_diag.get("h_webcam", {}), indent=2))

