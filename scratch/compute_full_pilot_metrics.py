import os
import json
from collections import defaultdict, Counter

filepath = os.path.abspath("backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl")

# Data structures
per_letter = defaultdict(lambda: {
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
    "confirmed_frames": 0,
    "trials": set(),
})

trials_data = defaultdict(lambda: {
    "expected": None,
    "frames": 0,
    "frame_signs": [],
    "frame_confs": [],
    "sources": Counter(),
    "gatings": Counter(),
    "confirmed_any": False,
    "confirmed_signs": set(),
    "sh_raws": [],
    "sh_confs": [],
})

mlp_eval = {
    "total": 0,
    "has_static_head_true": 0,
    "has_static_head_false": 0,
    "load_status": Counter(),
    "load_errors": Counter(),
    "pytorch_versions": Counter(),
    "called": 0,
    "not_called": 0,
    "accepted_fallback": 0,
    "accepted_supplemental": 0,
    "rejected_gating": 0,
    "raw_correct": 0,
    "raw_incorrect": 0,
    "raw_confusions": Counter(),
    "raw_confs_correct": [],
    "raw_confs_incorrect": [],
    "sources": Counter(),
}

retest_trials = []
high_conf_wrong_mlp = []
geom_mlp_disagreements = []
sustained_neutral_trials = []
misrecognized_trials = []

with open(filepath, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        tid = r.get("trial_id")
        exp = r.get("expected_sign")
        final_sign = r.get("final_sign")
        final_conf = float(r.get("final_confidence") or 0.0)
        source = r.get("prediction_source")
        gating = r.get("gating_status")
        confirmed = bool(r.get("confirmed"))
        
        # Letter stats
        st = per_letter[exp]
        st["total_frames"] += 1
        st["trials"].add(tid)
        st["sources"][source] += 1
        st["gatings"][gating] += 1
        if confirmed:
            st["confirmed_frames"] += 1
            
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
            
        if tid:
            td = trials_data[tid]
            td["expected"] = exp
            td["frames"] += 1
            td["frame_signs"].append(final_sign)
            td["frame_confs"].append(final_conf)
            td["sources"][source] += 1
            td["gatings"][gating] += 1
            if confirmed:
                td["confirmed_any"] = True
                td["confirmed_signs"].add(final_sign)
            
        # MLP Telemetry
        mlp_eval["total"] += 1
        if r.get("has_static_head") is True:
            mlp_eval["has_static_head_true"] += 1
        else:
            mlp_eval["has_static_head_false"] += 1
            
        sh = r.get("static_head") or {}
        mlp_eval["load_status"][sh.get("load_status")] += 1
        mlp_eval["pytorch_versions"][sh.get("pytorch_version")] += 1
        if sh.get("load_error"):
            mlp_eval["load_errors"][sh.get("load_error")] += 1
            
        mlp_eval["sources"][source] += 1
        sh_raw = sh.get("raw_predicted_letter")
        sh_conf = float(sh.get("confidence") or 0.0)
        
        if sh_raw is not None:
            mlp_eval["called"] += 1
            td["sh_raws"].append(sh_raw)
            td["sh_confs"].append(sh_conf)
            
            if source == "static_head_fallback":
                mlp_eval["accepted_fallback"] += 1
            elif source == "static_head":
                mlp_eval["accepted_supplemental"] += 1
            elif gating == "rejected":
                mlp_eval["rejected_gating"] += 1
                
            if sh_raw == exp:
                mlp_eval["raw_correct"] += 1
                mlp_eval["raw_confs_correct"].append(sh_conf)
            else:
                mlp_eval["raw_incorrect"] += 1
                mlp_eval["raw_confs_incorrect"].append(sh_conf)
                mlp_eval["raw_confusions"][f"{exp}->{sh_raw}"] += 1
                if sh_conf >= 0.80:
                    high_conf_wrong_mlp.append({
                        "line": line_idx,
                        "trial_id": tid,
                        "frame_idx": r.get("frame_idx"),
                        "expected": exp,
                        "mlp_pred": sh_raw,
                        "mlp_conf": sh_conf,
                        "final_sign": final_sign,
                        "source": source,
                        "top_k": sh.get("top_k"),
                    })
        else:
            mlp_eval["not_called"] += 1
            
        geom = r.get("geometric") or {}
        geom_stat = geom.get("status")
        geom_let = geom.get("letter")
        if geom_stat == "MATCH" and sh_raw is not None and geom_let != sh_raw:
            geom_mlp_disagreements.append({
                "line": line_idx,
                "trial_id": tid,
                "frame_idx": r.get("frame_idx"),
                "expected": exp,
                "geom_let": geom_let,
                "mlp_let": sh_raw,
                "mlp_conf": sh_conf,
                "final_sign": final_sign,
            })

# Analyze Trials
trials_summary = {}
trial_correct_count = 0
trial_incorrect_count = 0
trial_neutral_count = 0
trial_accuracy_by_letter = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "neutral": 0})

for tid, td in trials_data.items():
    exp = td["expected"]
    # Determine trial-level prediction:
    # 1. If any confirmed sign was produced, take dominant confirmed sign
    # 2. Else take non-neutral majority vote
    # 3. Else NEUTRAL
    non_neutral = [s for s in td["frame_signs"] if s != "NEUTRAL"]
    if td["confirmed_any"] and td["confirmed_signs"]:
        trial_pred = list(td["confirmed_signs"])[0]
    elif non_neutral:
        trial_pred = Counter(non_neutral).most_common(1)[0][0]
    else:
        trial_pred = "NEUTRAL"
        
    outcome = "CORRECT" if trial_pred == exp else ("NEUTRAL" if trial_pred == "NEUTRAL" else "INCORRECT")
    
    trial_accuracy_by_letter[exp]["total"] += 1
    if outcome == "CORRECT":
        trial_correct_count += 1
        trial_accuracy_by_letter[exp]["correct"] += 1
    elif outcome == "NEUTRAL":
        trial_neutral_count += 1
        trial_accuracy_by_letter[exp]["neutral"] += 1
        sustained_neutral_trials.append((tid, exp, td["frames"], td["sources"]))
    else:
        trial_incorrect_count += 1
        trial_accuracy_by_letter[exp]["incorrect"] += 1
        misrecognized_trials.append((tid, exp, trial_pred, td["frames"], Counter(td["frame_signs"])))
        
    if tid and ("_att02" in str(tid) or "_att03" in str(tid)):
        retest_trials.append(tid)

# Print Summary
print("\n=======================================================")
print(f"OVERALL METRICS (File: {os.path.basename(filepath)})")
print("=======================================================")
print(f"Total Frames: {sum(st['total_frames'] for st in per_letter.values())}")
print(f"Total Trials: {len(trials_data)}")
print(f"Trial Accuracy: {trial_correct_count}/{len(trials_data)} ({trial_correct_count / len(trials_data) * 100:.2f}%)")
print(f"Trial Neutral:  {trial_neutral_count}/{len(trials_data)} ({trial_neutral_count / len(trials_data) * 100:.2f}%)")
print(f"Trial Incorrect:{trial_incorrect_count}/{len(trials_data)} ({trial_incorrect_count / len(trials_data) * 100:.2f}%)")
print(f"Retest Trials ({len(retest_trials)}): {retest_trials}")

print("\n--- PER-LETTER TABLE ---")
print(f"{'Letter':<6}{'Trials':<8}{'TrCorrect':<10}{'TrAcc%':<9}{'Frames':<8}{'FrCorrect':<10}{'FrAcc%':<9}{'Neutral':<9}{'Incorrect':<10}{'AvgConfCor':<12}{'AvgConfInc':<12}{'TopConfusions':<20}")
all_letters_sorted = sorted([k for k in per_letter.keys() if k is not None])

letter_metrics_table = []
for l in all_letters_sorted:
    st = per_letter[l]
    tot_f = st["total_frames"]
    cor_f = st["correct_frames"]
    inc_f = st["incorrect_frames"]
    neu_f = st["neutral_frames"]
    fr_acc = (cor_f / tot_f * 100) if tot_f else 0.0
    
    tr_stat = trial_accuracy_by_letter[l]
    tr_tot = tr_stat["total"]
    tr_cor = tr_stat["correct"]
    tr_acc = (tr_cor / tr_tot * 100) if tr_tot else 0.0
    
    avg_c_cor = round(sum(st["correct_confs"]) / len(st["correct_confs"]), 3) if st["correct_confs"] else 0.0
    avg_c_inc = round(sum(st["incorrect_confs"]) / len(st["incorrect_confs"]), 3) if st["incorrect_confs"] else 0.0
    
    top_conf_str = ", ".join([f"{k}({v})" for k, v in st["confusions"].most_common(2)]) or "None"
    
    print(f"{l:<6}{tr_tot:<8}{tr_cor:<10}{tr_acc:<9.1f}{tot_f:<8}{cor_f:<10}{fr_acc:<9.1f}{neu_f:<9}{inc_f:<10}{avg_c_cor:<12.3f}{avg_c_inc:<12.3f}{top_conf_str:<20}")
    
    letter_metrics_table.append({
        "letter": l,
        "trials_total": tr_tot,
        "trials_correct": tr_cor,
        "trials_incorrect": tr_stat["incorrect"],
        "trials_neutral": tr_stat["neutral"],
        "trial_accuracy_pct": round(tr_acc, 2),
        "frames_total": tot_f,
        "frames_correct": cor_f,
        "frames_incorrect": inc_f,
        "frames_neutral": neu_f,
        "frame_accuracy_pct": round(fr_acc, 2),
        "avg_conf_correct": avg_c_cor,
        "avg_conf_incorrect": avg_c_inc,
        "top_confusions": dict(st["confusions"].most_common(5)),
        "sources": dict(st["sources"]),
        "gatings": dict(st["gatings"]),
        "confirmed_frames": st["confirmed_frames"],
    })

print("\n--- SOURCES BREAKDOWN ---")
all_sources = Counter()
for st in per_letter.values():
    all_sources.update(st["sources"])
for k, v in all_sources.items():
    print(f"  {k:<25}: {v:>6} frames ({v / sum(all_sources.values()) * 100:>5.1f}%)")

print("\n--- STATIC MLP TELEMETRY ---")
print(f"  Total records:           {mlp_eval['total']}")
print(f"  has_static_head == True: {mlp_eval['has_static_head_true']} ({mlp_eval['has_static_head_true'] / mlp_eval['total'] * 100:.1f}%)")
print(f"  load_status:             {dict(mlp_eval['load_status'])}")
print(f"  pytorch_versions:        {dict(mlp_eval['pytorch_versions'])}")
print(f"  load_errors:             {dict(mlp_eval['load_errors'])}")
print(f"  MLP Called:              {mlp_eval['called']} frames ({mlp_eval['called'] / mlp_eval['total'] * 100:.1f}%)")
print(f"  MLP Not Called (Geom):   {mlp_eval['not_called']} frames ({mlp_eval['not_called'] / mlp_eval['total'] * 100:.1f}%)")
print(f"  Accepted Fallback:       {mlp_eval['accepted_fallback']} frames")
print(f"  Accepted Supplemental:   {mlp_eval['accepted_supplemental']} frames")
print(f"  Rejected Gating:         {mlp_eval['rejected_gating']} frames")
raw_total = mlp_eval["raw_correct"] + mlp_eval["raw_incorrect"]
if raw_total > 0:
    print(f"  Raw MLP Accuracy:        {mlp_eval['raw_correct']}/{raw_total} ({mlp_eval['raw_correct'] / raw_total * 100:.1f}%)")
    print(f"  Raw Top Confusions:      {mlp_eval['raw_confusions'].most_common(5)}")

# Save full metrics
results_obj = {
    "file_path": filepath,
    "file_name": os.path.basename(filepath),
    "total_frames": sum(st['total_frames'] for st in per_letter.values()),
    "total_trials": len(trials_data),
    "trial_accuracy_overall": {
        "correct": trial_correct_count,
        "incorrect": trial_incorrect_count,
        "neutral": trial_neutral_count,
        "accuracy_pct": round(trial_correct_count / len(trials_data) * 100, 2),
    },
    "retest_trials": retest_trials,
    "letter_metrics": letter_metrics_table,
    "sources_breakdown": dict(all_sources),
    "mlp_telemetry": {
        "has_static_head_true": mlp_eval["has_static_head_true"],
        "has_static_head_false": mlp_eval["has_static_head_false"],
        "load_status": dict(mlp_eval["load_status"]),
        "pytorch_versions": dict(mlp_eval["pytorch_versions"]),
        "load_errors": dict(mlp_eval["load_errors"]),
        "called": mlp_eval["called"],
        "not_called": mlp_eval["not_called"],
        "accepted_fallback": mlp_eval["accepted_fallback"],
        "accepted_supplemental": mlp_eval["accepted_supplemental"],
        "rejected_gating": mlp_eval["rejected_gating"],
        "raw_correct": mlp_eval["raw_correct"],
        "raw_incorrect": mlp_eval["raw_incorrect"],
        "raw_confusions": dict(mlp_eval["raw_confusions"].most_common(10)),
    },
    "misrecognized_trials": misrecognized_trials,
    "sustained_neutral_trials": sustained_neutral_trials,
    "high_conf_wrong_mlp_count": len(high_conf_wrong_mlp),
    "sample_high_conf_wrong_mlp": high_conf_wrong_mlp[:10],
    "sample_geom_mlp_disagreements": geom_mlp_disagreements[:10],
}

with open("scratch/audit_pilot_2141_full_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results_obj, f, indent=2, ensure_ascii=False)

print("\nFull metrics saved to scratch/audit_pilot_2141_full_metrics.json")
