import os
import sys
import json
import hashlib
from collections import defaultdict, Counter

filepath = os.path.abspath("backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl")

print("=== CALCULATING FILE SHA-256 ===")
h = hashlib.sha256()
with open(filepath, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
file_sha256 = h.hexdigest()
print(f"File SHA-256: {file_sha256}")
file_size = os.path.getsize(filepath)
print(f"File Size: {file_size:,} bytes")

# Pass 1: Parse and validate integrity
total_lines = 0
invalid_json_lines = 0
missing_fields_records = 0

sessions = set()
trials_order = []
trial_set = set()
frames_by_trial = defaultdict(list)
records_by_trial = defaultdict(list)

expected_signs = set()
letters_present = set()

# Metrics collection
per_letter_stats = defaultdict(lambda: {
    "total_frames": 0,
    "correct_frames": 0,
    "incorrect_frames": 0,
    "neutral_frames": 0,
    "correct_confs": [],
    "incorrect_confs": [],
    "neutral_confs": [],
    "confusions": Counter(),
    "sources": Counter(),
    "gatings": Counter(),
    "confirmed_count": 0,
    "trials": set(),
})

mlp_stats = {
    "total_records": 0,
    "has_static_head_true": 0,
    "has_static_head_false": 0,
    "load_status_counts": Counter(),
    "load_errors": Counter(),
    "pytorch_versions": Counter(),
    "mlp_called_count": 0,
    "mlp_not_called_count": 0,
    "mlp_accepted_fallback": 0,
    "mlp_accepted_supplemental": 0,
    "mlp_rejected_gating": 0,
    "mlp_high_conf_wrong": [],
    "mlp_geom_disagreements": [],
}

trial_level_summary = {}

print("\n=== PARSING JSONL RECORDS ===")
with open(filepath, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        line_str = line.strip()
        if not line_str:
            continue
        total_lines += 1
        try:
            r = json.loads(line_str)
        except Exception as e:
            invalid_json_lines += 1
            continue

        # Check required fields
        required_keys = ["timestamp", "session_id", "trial_id", "frame_idx", "expected_sign", "final_sign", "final_confidence", "prediction_source", "confirmed", "has_static_head", "geometric", "static_head", "gating_status"]
        if not all(k in r for k in required_keys):
            missing_fields_records += 1

        sess = r.get("session_id")
        tid = r.get("trial_id")
        fidx = r.get("frame_idx")
        exp = r.get("expected_sign")
        final_sign = r.get("final_sign")
        final_conf = float(r.get("final_confidence") or 0.0)
        source = r.get("prediction_source")
        gating = r.get("gating_status")
        confirmed = bool(r.get("confirmed"))

        if sess:
            sessions.add(sess)
        if tid:
            if tid not in trial_set:
                trial_set.add(tid)
                trials_order.append(tid)
            frames_by_trial[tid].append(fidx)
            # Store lightweight record for anomaly detection
            records_by_trial[tid].append({
                "line": line_idx,
                "frame_idx": fidx,
                "expected": exp,
                "final_sign": final_sign,
                "confidence": final_conf,
                "source": source,
                "gating": gating,
                "confirmed": confirmed,
                "geom_status": r.get("geometric", {}).get("status") if isinstance(r.get("geometric"), dict) else None,
                "geom_letter": r.get("geometric", {}).get("letter") if isinstance(r.get("geometric"), dict) else None,
                "sh_raw": r.get("static_head", {}).get("raw_predicted_letter") if isinstance(r.get("static_head"), dict) else None,
                "sh_conf": r.get("static_head", {}).get("confidence") if isinstance(r.get("static_head"), dict) else None,
                "sh_top_k": r.get("static_head", {}).get("top_k") if isinstance(r.get("static_head"), dict) else None,
                "has_sh": r.get("has_static_head"),
                "sh_load_status": r.get("static_head", {}).get("load_status") if isinstance(r.get("static_head"), dict) else None,
                "sh_error": r.get("static_head", {}).get("load_error") if isinstance(r.get("static_head"), dict) else None,
                "pytorch_v": r.get("static_head", {}).get("pytorch_version") if isinstance(r.get("static_head"), dict) else None,
                "reject_reason": r.get("reject_reason"),
            })

        if exp:
            expected_signs.add(exp)
            st = per_letter_stats[exp]
            st["total_frames"] += 1
            st["trials"].add(tid)
            st["sources"][source] += 1
            st["gatings"][gating] += 1
            if confirmed:
                st["confirmed_count"] += 1

            if final_sign == exp:
                st["correct_frames"] += 1
                st["correct_confs"].append(final_conf)
            elif final_sign == "NEUTRAL":
                st["neutral_frames"] += 1
                st["neutral_confs"].append(final_conf)
            else:
                st["incorrect_frames"] += 1
                st["incorrect_confs"].append(final_conf)
                st["confusions"][final_sign] += 1

        # MLP Telemetry Stats
        mlp_stats["total_records"] += 1
        has_sh = r.get("has_static_head")
        if has_sh is True:
            mlp_stats["has_static_head_true"] += 1
        else:
            mlp_stats["has_static_head_false"] += 1

        sh_data = r.get("static_head", {}) if isinstance(r.get("static_head"), dict) else {}
        mlp_stats["load_status_counts"][sh_data.get("load_status")] += 1
        if sh_data.get("load_error"):
            mlp_stats["load_errors"][sh_data.get("load_error")] += 1
        mlp_stats["pytorch_versions"][sh_data.get("pytorch_version")] += 1

        sh_raw = sh_data.get("raw_predicted_letter")
        sh_conf = float(sh_data.get("confidence") or 0.0)
        if sh_raw is not None:
            mlp_stats["mlp_called_count"] += 1
            if source == "static_head_fallback":
                mlp_stats["mlp_accepted_fallback"] += 1
            elif source == "static_head":
                mlp_stats["mlp_accepted_supplemental"] += 1
            elif gating == "rejected":
                mlp_stats["mlp_rejected_gating"] += 1

            # High confidence wrong
            if sh_raw != exp and sh_conf >= 0.85:
                mlp_stats["mlp_high_conf_wrong"].append({
                    "trial_id": tid,
                    "frame_idx": fidx,
                    "expected": exp,
                    "mlp_pred": sh_raw,
                    "mlp_conf": sh_conf,
                    "source": source,
                })
        else:
            mlp_stats["mlp_not_called_count"] += 1

        geom_data = r.get("geometric", {}) if isinstance(r.get("geometric"), dict) else {}
        geom_stat = geom_data.get("status")
        geom_let = geom_data.get("letter")
        if geom_stat == "MATCH" and sh_raw is not None and geom_let != sh_raw:
            mlp_stats["mlp_geom_disagreements"].append({
                "trial_id": tid,
                "frame_idx": fidx,
                "expected": exp,
                "geom_let": geom_let,
                "mlp_let": sh_raw,
            })

print(f"Total Lines: {total_lines}")
print(f"Invalid JSON Lines: {invalid_json_lines}")
print(f"Records with missing fields: {missing_fields_records}")
print(f"Sessions: {sessions}")
print(f"Total Unique Trials: {len(trial_set)}")
print(f"Total Expected Signs: {len(expected_signs)}: {sorted(list(expected_signs))}")

# Check trial frames integrity
non_monotonic_trials = []
duplicate_frames_trials = []
trial_counts_by_letter = Counter()
frames_counts_by_trial = {}

for tid, f_list in frames_by_trial.items():
    exp = records_by_trial[tid][0]["expected"]
    trial_counts_by_letter[exp] += 1
    frames_counts_by_trial[tid] = len(f_list)
    # Check duplicates
    if len(f_list) != len(set(f_list)):
        duplicate_frames_trials.append((tid, f_list))
    # Check monotonicity
    if f_list != list(range(f_list[0], f_list[0] + len(f_list))):
        non_monotonic_trials.append((tid, f_list))

print(f"\nTrial counts by letter: {dict(trial_counts_by_letter)}")
print(f"Non-monotonic trials: {len(non_monotonic_trials)}")
print(f"Duplicate frames trials: {len(duplicate_frames_trials)}")

out_data = {
    "file_path": filepath,
    "file_size": file_size,
    "file_sha256": file_sha256,
    "total_lines": total_lines,
    "sessions": list(sessions),
    "total_trials": len(trial_set),
    "trials_order": trials_order,
    "trial_counts_by_letter": dict(trial_counts_by_letter),
    "duplicate_frames_trials_count": len(duplicate_frames_trials),
    "non_monotonic_trials_count": len(non_monotonic_trials),
    "mlp_stats": {
        "has_static_head_true": mlp_stats["has_static_head_true"],
        "has_static_head_false": mlp_stats["has_static_head_false"],
        "load_status_counts": dict(mlp_stats["load_status_counts"]),
        "load_errors": dict(mlp_stats["load_errors"]),
        "pytorch_versions": dict(mlp_stats["pytorch_versions"]),
        "mlp_called_count": mlp_stats["mlp_called_count"],
        "mlp_not_called_count": mlp_stats["mlp_not_called_count"],
        "mlp_accepted_fallback": mlp_stats["mlp_accepted_fallback"],
        "mlp_accepted_supplemental": mlp_stats["mlp_accepted_supplemental"],
        "mlp_rejected_gating": mlp_stats["mlp_rejected_gating"],
        "high_conf_wrong_count": len(mlp_stats["mlp_high_conf_wrong"]),
        "geom_disagreements_count": len(mlp_stats["mlp_geom_disagreements"]),
    },
}

with open("scratch/audit_pilot_2141_raw.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print("\nBasic audit saved to scratch/audit_pilot_2141_raw.json")

