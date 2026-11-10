import os
import sys
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np

filepath = os.path.abspath("backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl")

# 1. SHA-256 Verification
h = hashlib.sha256()
with open(filepath, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
file_sha256 = h.hexdigest()
file_size = os.path.getsize(filepath)

print("=== PART A: VERIFYING DATA INTEGRITY ===")
print(f"File: {filepath}")
print(f"Size: {file_size:,} bytes")
print(f"SHA-256: {file_sha256}")
expected_sha = "fb4f725757bcbc73b96748896c4149c21927461314c72e68a49310b7cd76429f"
assert file_sha256 == expected_sha, f"SHA-256 MISMATCH! Got {file_sha256}"
print("SHA-256 MATCH VERIFIED 100%!")

# 2. Parse JSONL with strict separation of trial vs inter-trial frames
total_records = 0
invalid_records = 0
inter_trial_records = 0
trial_records = 0

STATIC_LETTERS_22 = ["A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y"]

# Aggregations
trials_dict = defaultdict(lambda: {
    "expected_sign": None,
    "letter": None,
    "rep": None,
    "attempt": None,
    "frames": [],
    "frame_indices": [],
    "final_signs": [],
    "final_confs": [],
    "sources": Counter(),
    "gatings": Counter(),
    "confirmed_frames": 0,
    "confirmed_signs": set(),
})

letter_frames = defaultdict(lambda: {
    "total": 0,
    "correct": 0,
    "incorrect": 0,
    "neutral": 0,
    "confs_correct": [],
    "confs_incorrect": [],
    "confs_neutral": [],
    "confs_all": [],
    "sources": Counter(),
    "gatings": Counter(),
    "confusions": Counter(),
    "confirmed_frames": 0,
    "att01_frames": 0,
    "att02_frames": 0,
})

# Layer 1: Geometric
geom_eval = {
    "total": 0,
    "status_counts": Counter(),
    "letters_predicted": Counter(),
    "correct": 0,
    "incorrect": 0,
    "no_match": 0,
    "confusions": Counter(),
    "per_letter": defaultdict(lambda: {"MATCH": 0, "NO_MATCH": 0, "UNKNOWN": 0, "correct": 0, "incorrect": 0, "confusions": Counter()}),
}

# Layer 2: Static MLP
mlp_eval = {
    "invoked_count": 0,
    "not_invoked_count": 0,
    "raw_correct": 0,
    "raw_incorrect": 0,
    "raw_confs_correct": [],
    "raw_confs_incorrect": [],
    "raw_preds_distribution": Counter(),
    "raw_confusions": Counter(),
    "top_k_ranks": Counter(), # rank of expected letter in top_k
    "per_letter": defaultdict(lambda: {"invoked": 0, "raw_correct": 0, "raw_incorrect": 0, "raw_confs": [], "top_preds": Counter()}),
}

# Layer 3: Gating
gating_eval = {
    "passed_geometric": 0,
    "passed_fallback": 0,
    "passed_supplemental": 0,
    "rejected": 0,
    "reject_reasons": Counter(),
    "gating_matrix": {
        "true_accept": 0,   # MLP raw correct & accepted
        "false_accept": 0,  # MLP raw incorrect & accepted (LEAK)
        "true_reject": 0,   # MLP raw incorrect & rejected (PROTECTED)
        "false_reject": 0,  # MLP raw correct & rejected (MISSED OPPORTUNITY)
    },
    "per_letter": defaultdict(lambda: Counter()),
}

# Confusion Matrix: 22 rows x 23 cols (22 letters + NEUTRAL)
confusion_matrix = defaultdict(lambda: Counter())

with open(filepath, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        total_records += 1
        try:
            r = json.loads(line)
        except Exception:
            invalid_records += 1
            continue

        tid = r.get("trial_id")
        exp = r.get("expected_sign")
        final_sign = r.get("final_sign")
        final_conf = float(r.get("final_confidence") or 0.0)
        source = r.get("prediction_source")
        gating = r.get("gating_status")
        confirmed = bool(r.get("confirmed"))
        fidx = r.get("frame_idx")

        # Separate inter-trial transition frames
        if tid is None or exp is None:
            inter_trial_records += 1
            continue

        trial_records += 1

        # Parse trial id details: pilot_20261107_2141_A_rep01_att01
        parts = tid.split("_")
        # e.g., ['pilot', '20261107', '2141', 'A', 'rep01', 'att01']
        letter = parts[3] if len(parts) >= 4 else exp
        rep_str = parts[4] if len(parts) >= 5 else "rep01"
        att_str = parts[5] if len(parts) >= 6 else "att01"

        # Populate trial container
        t_obj = trials_dict[tid]
        t_obj["expected_sign"] = exp
        t_obj["letter"] = letter
        t_obj["rep"] = rep_str
        t_obj["attempt"] = att_str
        t_obj["frames"].append(r)
        t_obj["frame_indices"].append(fidx)
        t_obj["final_signs"].append(final_sign)
        t_obj["final_confs"].append(final_conf)
        t_obj["sources"][source] += 1
        t_obj["gatings"][gating] += 1
        if confirmed:
            t_obj["confirmed_frames"] += 1
            t_obj["confirmed_signs"].add(final_sign)

        # Populate letter frame stats
        lst = letter_frames[exp]
        lst["total"] += 1
        lst["confs_all"].append(final_conf)
        lst["sources"][source] += 1
        lst["gatings"][gating] += 1
        if att_str == "att01":
            lst["att01_frames"] += 1
        else:
            lst["att02_frames"] += 1

        if confirmed:
            lst["confirmed_frames"] += 1

        # Frame correctness & confusion matrix
        confusion_matrix[exp][final_sign] += 1

        if final_sign == exp:
            lst["correct"] += 1
            lst["confs_correct"].append(final_conf)
        elif final_sign == "NEUTRAL":
            lst["neutral"] += 1
            lst["confs_neutral"].append(final_conf)
        else:
            lst["incorrect"] += 1
            lst["confs_incorrect"].append(final_conf)
            lst["confusions"][final_sign] += 1

        # Layer 1: Geometric Rule Analysis
        geom = r.get("geometric") or {}
        g_stat = geom.get("status", "UNKNOWN")
        g_let = geom.get("letter")
        geom_eval["total"] += 1
        geom_eval["status_counts"][g_stat] += 1
        geom_eval["per_letter"][exp][g_stat] += 1

        if g_stat == "MATCH":
            geom_eval["letters_predicted"][g_let] += 1
            if g_let == exp:
                geom_eval["correct"] += 1
                geom_eval["per_letter"][exp]["correct"] += 1
            else:
                geom_eval["incorrect"] += 1
                geom_eval["confusions"][f"{exp}->{g_let}"] += 1
                geom_eval["per_letter"][exp]["incorrect"] += 1
                geom_eval["per_letter"][exp]["confusions"][g_let] += 1
        else:
            geom_eval["no_match"] += 1

        # Layer 2: Static MLP Analysis
        sh = r.get("static_head") or {}
        raw_pred = sh.get("raw_predicted_letter")
        raw_conf = float(sh.get("confidence") or 0.0)
        top_k = sh.get("top_k") or []

        if raw_pred is not None:
            mlp_eval["invoked_count"] += 1
            mlp_eval["raw_preds_distribution"][raw_pred] += 1
            mlp_eval["per_letter"][exp]["invoked"] += 1
            mlp_eval["per_letter"][exp]["raw_confs"].append(raw_conf)
            mlp_eval["per_letter"][exp]["top_preds"][raw_pred] += 1

            if raw_pred == exp:
                mlp_eval["raw_correct"] += 1
                mlp_eval["raw_confs_correct"].append(raw_conf)
                mlp_eval["per_letter"][exp]["raw_correct"] += 1
            else:
                mlp_eval["raw_incorrect"] += 1
                mlp_eval["raw_confs_incorrect"].append(raw_conf)
                mlp_eval["raw_confusions"][f"{exp}->{raw_pred}"] += 1
                mlp_eval["per_letter"][exp]["raw_incorrect"] += 1

            # Check top-k rank of expected sign
            rank_found = None
            for idx, cand in enumerate(top_k):
                if cand.get("letter") == exp:
                    rank_found = idx + 1
                    break
            if rank_found is not None:
                mlp_eval["top_k_ranks"][rank_found] += 1
            else:
                mlp_eval["top_k_ranks"]["not_in_top_k"] += 1

            # Layer 3: Gating Analysis
            gating_eval["per_letter"][exp][gating] += 1
            if gating == "passed_geometric":
                gating_eval["passed_geometric"] += 1
            elif gating == "passed_fallback":
                gating_eval["passed_fallback"] += 1
            elif gating == "passed_supplemental":
                gating_eval["passed_supplemental"] += 1
            elif gating == "rejected":
                gating_eval["rejected"] += 1
                rej_r = r.get("reject_reason") or "Unknown"
                gating_eval["reject_reasons"][rej_r] += 1

            # Gating Decision Matrix (True/False Accept/Reject)
            is_accepted = source in ("static_head", "static_head_fallback")
            is_raw_correct = (raw_pred == exp)
            if is_accepted and is_raw_correct:
                gating_eval["gating_matrix"]["true_accept"] += 1
            elif is_accepted and not is_raw_correct:
                gating_eval["gating_matrix"]["false_accept"] += 1
            elif not is_accepted and is_raw_correct:
                gating_eval["gating_matrix"]["false_reject"] += 1
            elif not is_accepted and not is_raw_correct:
                gating_eval["gating_matrix"]["true_reject"] += 1
        else:
            mlp_eval["not_invoked_count"] += 1
            if gating == "passed_geometric":
                gating_eval["passed_geometric"] += 1
            gating_eval["per_letter"][exp][gating] += 1

print(f"\nRecord breakdown:")
print(f"  Total JSONL Lines:      {total_records}")
print(f"  Active Trial Frames:    {trial_records}")
print(f"  Inter-trial Frames:     {inter_trial_records}")
print(f"  Corrupted Lines:        {invalid_records}")

# 3. Adjudicate Trials
# Definition of Trial Outcomes:
# - Rule 1 (Confirmed UX Rule): Emitted confirmed token matching expected sign.
# - Rule 2 (Majority Rule): Non-neutral majority vote matches expected sign.
# - Categories:
#   * CORRECT (Dominant matches expected)
#   * NEUTRAL (100% or dominant NEUTRAL)
#   * INCORRECT (Dominant is a wrong sign)
#   * MIXED (Has both correct and incorrect/neutral without clean majority)

trial_adjudication = {}
trial_summary_by_letter = defaultdict(lambda: {
    "total_trials": 0,
    "att01_trials": 0,
    "att02_trials": 0,
    "correct_trials": 0,
    "incorrect_trials": 0,
    "neutral_trials": 0,
    "mixed_trials": 0,
    "confirmed_trials": 0,
    "att01_correct": 0,
    "att02_correct": 0,
})

for tid, td in trials_dict.items():
    exp = td["expected_sign"]
    att = td["attempt"]
    signs = td["final_signs"]
    non_neutral = [s for s in signs if s != "NEUTRAL"]
    
    # Check confirmation
    has_confirmed_exp = (exp in td["confirmed_signs"])
    
    # Majority sign
    if non_neutral:
        c_non = Counter(non_neutral)
        top_sign, top_cnt = c_non.most_common(1)[0]
        majority_ratio = top_cnt / len(non_neutral)
    else:
        top_sign = "NEUTRAL"
        majority_ratio = 1.0

    # Adjudication
    if top_sign == exp and majority_ratio >= 0.5:
        outcome = "CORRECT"
    elif top_sign == "NEUTRAL" or len(non_neutral) == 0:
        outcome = "NEUTRAL"
    elif top_sign != exp and majority_ratio >= 0.5:
        outcome = "INCORRECT"
    else:
        outcome = "MIXED"

    is_pure = (len(set(signs)) == 1)
    
    trial_adjudication[tid] = {
        "letter": exp,
        "rep": td["rep"],
        "attempt": att,
        "frame_count": len(signs),
        "outcome": outcome,
        "is_pure": is_pure,
        "top_sign": top_sign,
        "has_confirmed_exp": has_confirmed_exp,
        "confirmed_signs": list(td["confirmed_signs"]),
        "sign_distribution": dict(Counter(signs)),
        "sources": dict(td["sources"]),
        "gatings": dict(td["gatings"]),
    }

    st = trial_summary_by_letter[exp]
    st["total_trials"] += 1
    if att == "att01":
        st["att01_trials"] += 1
        if outcome == "CORRECT":
            st["att01_correct"] += 1
    else:
        st["att02_trials"] += 1
        if outcome == "CORRECT":
            st["att02_correct"] += 1

    if outcome == "CORRECT":
        st["correct_trials"] += 1
    elif outcome == "NEUTRAL":
        st["neutral_trials"] += 1
    elif outcome == "INCORRECT":
        st["incorrect_trials"] += 1
    elif outcome == "MIXED":
        st["mixed_trials"] += 1

    if has_confirmed_exp:
        st["confirmed_trials"] += 1

print("\n=== TRIAL ADJUDICATION COMPLETE ===")
total_trials_all = len(trials_dict)
total_correct_trials = sum(st["correct_trials"] for st in trial_summary_by_letter.values())
total_neutral_trials = sum(st["neutral_trials"] for st in trial_summary_by_letter.values())
total_incorrect_trials = sum(st["incorrect_trials"] for st in trial_summary_by_letter.values())
total_mixed_trials = sum(st["mixed_trials"] for st in trial_summary_by_letter.values())

print(f"Total Trials:      {total_trials_all}")
print(f"Correct Trials:    {total_correct_trials} ({total_correct_trials / total_trials_all * 100:.2f}%)")
print(f"Neutral Trials:    {total_neutral_trials} ({total_neutral_trials / total_trials_all * 100:.2f}%)")
print(f"Incorrect Trials:  {total_incorrect_trials} ({total_incorrect_trials / total_trials_all * 100:.2f}%)")
print(f"Mixed Trials:      {total_mixed_trials} ({total_mixed_trials / total_trials_all * 100:.2f}%)")

# Output detailed summary JSON
summary_output = {
    "file_sha256": file_sha256,
    "file_size": file_size,
    "total_records": total_records,
    "active_trial_frames": trial_records,
    "inter_trial_frames": inter_trial_records,
    "total_trials": total_trials_all,
    "trial_summary": {
        "correct": total_correct_trials,
        "neutral": total_neutral_trials,
        "incorrect": total_incorrect_trials,
        "mixed": total_mixed_trials,
        "accuracy_pct": round(total_correct_trials / total_trials_all * 100, 2),
    },
    "trial_summary_by_letter": trial_summary_by_letter,
    "letter_frame_stats": {},
    "geometric_eval": {
        "total": geom_eval["total"],
        "status_counts": dict(geom_eval["status_counts"]),
        "correct": geom_eval["correct"],
        "incorrect": geom_eval["incorrect"],
        "no_match": geom_eval["no_match"],
        "confusions": dict(geom_eval["confusions"].most_common(10)),
    },
    "mlp_eval": {
        "invoked": mlp_eval["invoked_count"],
        "not_invoked": mlp_eval["not_invoked_count"],
        "raw_correct": mlp_eval["raw_correct"],
        "raw_incorrect": mlp_eval["raw_incorrect"],
        "raw_accuracy_pct": round(mlp_eval["raw_correct"] / max(1, mlp_eval["invoked_count"]) * 100, 2),
        "mean_conf_correct": round(float(np.mean(mlp_eval["raw_confs_correct"])), 3) if mlp_eval["raw_confs_correct"] else 0.0,
        "mean_conf_incorrect": round(float(np.mean(mlp_eval["raw_confs_incorrect"])), 3) if mlp_eval["raw_confs_incorrect"] else 0.0,
        "raw_preds_distribution": dict(mlp_eval["raw_preds_distribution"].most_common(15)),
        "raw_confusions": dict(mlp_eval["raw_confusions"].most_common(15)),
        "top_k_ranks": dict(mlp_eval["top_k_ranks"]),
    },
    "gating_eval": {
        "passed_geometric": gating_eval["passed_geometric"],
        "passed_fallback": gating_eval["passed_fallback"],
        "passed_supplemental": gating_eval["passed_supplemental"],
        "rejected": gating_eval["rejected"],
        "gating_matrix": gating_eval["gating_matrix"],
        "reject_reasons": dict(gating_eval["reject_reasons"].most_common(10)),
    },
    "confusion_matrix": {row: dict(col) for row, col in confusion_matrix.items()},
}

# Compile per-letter statistics
for l in STATIC_LETTERS_22:
    st = letter_frames[l]
    tr_st = trial_summary_by_letter[l]
    tot = st["total"]
    cor = st["correct"]
    inc = st["incorrect"]
    neu = st["neutral"]
    confs_c = st["confs_correct"]
    confs_i = st["confs_incorrect"]
    confs_all = st["confs_all"]

    summary_output["letter_frame_stats"][l] = {
        "letter": l,
        "total_frames": tot,
        "correct_frames": cor,
        "incorrect_frames": inc,
        "neutral_frames": neu,
        "frame_accuracy_pct": round(cor / tot * 100, 2) if tot else 0.0,
        "att01_frames": st["att01_frames"],
        "att02_frames": st["att02_frames"],
        "confirmed_frames": st["confirmed_frames"],
        "conf_mean_correct": round(float(np.mean(confs_c)), 3) if confs_c else 0.0,
        "conf_median_correct": round(float(np.median(confs_c)), 3) if confs_c else 0.0,
        "conf_mean_incorrect": round(float(np.mean(confs_i)), 3) if confs_i else 0.0,
        "conf_median_incorrect": round(float(np.median(confs_i)), 3) if confs_i else 0.0,
        "conf_percentiles": {
            "p25": round(float(np.percentile(confs_all, 25)), 3) if confs_all else 0.0,
            "p50": round(float(np.percentile(confs_all, 50)), 3) if confs_all else 0.0,
            "p75": round(float(np.percentile(confs_all, 75)), 3) if confs_all else 0.0,
            "p90": round(float(np.percentile(confs_all, 90)), 3) if confs_all else 0.0,
        },
        "sources": dict(st["sources"]),
        "gatings": dict(st["gatings"]),
        "top_confusions": dict(st["confusions"].most_common(5)),
        "trials_total": tr_st["total_trials"],
        "trials_correct": tr_st["correct_trials"],
        "trials_neutral": tr_st["neutral_trials"],
        "trials_incorrect": tr_st["incorrect_trials"],
        "trial_accuracy_pct": round(tr_st["correct_trials"] / tr_st["total_trials"] * 100, 2) if tr_st["total_trials"] else 0.0,
        "att01_accuracy_pct": round(tr_st["att01_correct"] / tr_st["att01_trials"] * 100, 2) if tr_st["att01_trials"] else 0.0,
        "att02_accuracy_pct": round(tr_st["att02_correct"] / tr_st["att02_trials"] * 100, 2) if tr_st["att02_trials"] else 0.0,
        "confirmed_trials": tr_st["confirmed_trials"],
    }

with open("scratch/task12_deep_recomputation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_output, f, indent=2, ensure_ascii=False)

print("\nSaved full computation to scratch/task12_deep_recomputation_summary.json")

