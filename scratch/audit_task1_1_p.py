import json
import hashlib
import os
import datetime

def audit_file(filepath):
    print("=" * 70)
    print("AUDIT FILE:", filepath)
    print("=" * 70)
    if not os.path.exists(filepath):
        print("FILE NOT FOUND")
        return
    
    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    
    records = []
    corrupted_lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append((idx + 1, rec))
            except Exception as e:
                corrupted_lines.append((idx + 1, str(e)))
                
    print(f"File Size: {file_size} bytes ({file_size/1024:.2f} KB)")
    print(f"SHA256: {file_sha256}")
    print(f"Corrupted JSON lines: {len(corrupted_lines)}")
    if corrupted_lines:
        print("Sample corrupted lines:", corrupted_lines[:5])
    print(f"Total valid records: {len(records)}")
    
    # Session and timestamp
    timestamps = [r.get("timestamp") for _, r in records if r.get("timestamp") is not None]
    if timestamps:
        t_min = min(timestamps)
        t_max = max(timestamps)
        dt_start = datetime.datetime.fromtimestamp(t_min)
        dt_end = datetime.datetime.fromtimestamp(t_max)
        duration_sec = t_max - t_min
        print(f"Start Time: {dt_start} (timestamp {t_min})")
        print(f"End Time:   {dt_end} (timestamp {t_max})")
        print(f"Elapsed:    {duration_sec:.2f}s ({duration_sec/60:.2f} mins)")
    
    # Classify active trial frames vs idle/transition frames
    trial_records = {} # trial_id -> list of records
    idle_records = []
    
    for line_idx, r in records:
        tid = r.get("trial_id")
        if tid:
            trial_records.setdefault(tid, []).append((line_idx, r))
        else:
            idle_records.append((line_idx, r))
            
    print(f"\nActive Trials count: {len(trial_records)}")
    print(f"Active trial frames: {sum(len(v) for v in trial_records.values())}")
    print(f"Idle/Transition frames (trial_id is null): {len(idle_records)}")
    
    # Check sessions and expected signs
    all_sessions = set(r.get("session_id") for _, r in records)
    all_expected = set(r.get("expected_sign") for _, r in records)
    print(f"Session IDs in file: {all_sessions}")
    print(f"Expected signs in file: {all_expected}")

    # Pipeline & Model Info
    has_static_heads = set(r.get("has_static_head") for _, r in records)
    pred_sources = {}
    gating_statuses = {}
    final_signs = {}
    mlp_raw_preds = {}
    mlp_statuses = set()
    mlp_pytorch_versions = set()
    geom_statuses = {}
    p_rule_evals = []
    
    for _, r in records:
        ps = r.get("prediction_source")
        pred_sources[ps] = pred_sources.get(ps, 0) + 1
        gs = r.get("gating_status")
        gating_statuses[gs] = gating_statuses.get(gs, 0) + 1
        fs = r.get("final_sign")
        final_signs[fs] = final_signs.get(fs, 0) + 1
        
        sh = r.get("static_head") or {}
        if isinstance(sh, dict):
            mlp_statuses.add(sh.get("load_status"))
            mlp_pytorch_versions.add(sh.get("pytorch_version"))
            raw_let = sh.get("raw_predicted_letter")
            mlp_raw_preds[raw_let] = mlp_raw_preds.get(raw_let, 0) + 1
            
        geom = r.get("geometric") or {}
        if isinstance(geom, dict):
            gst = geom.get("status")
            geom_statuses[gst] = geom_statuses.get(gst, 0) + 1
            if "p_rule_diagnostics" in geom:
                p_rule_evals.append(geom.get("p_rule_diagnostics"))
                
    print("\n--- PIPELINE TELEMETRY ---")
    print(f"has_static_head flag: {has_static_heads}")
    print(f"MLP load_status: {mlp_statuses}")
    print(f"MLP pytorch_version: {mlp_pytorch_versions}")
    print(f"MLP raw predictions distribution: {mlp_raw_preds}")
    print(f"Geometric status distribution: {geom_statuses}")
    print(f"Prediction sources distribution: {pred_sources}")
    print(f"Final sign distribution: {final_signs}")
    print(f"Gating status distribution: {gating_statuses}")
    print(f"P rule diagnostics present in records: {len([p for p in p_rule_evals if p is not None])}")

    # Detailed Trial-by-Trial Analysis for P
    print("\n--- TRIAL-BY-TRIAL BREAKDOWN ---")
    total_trial_frames = 0
    total_correct_p = 0
    total_wrong = 0
    total_neutral = 0
    wrong_confusions = {}

    for tid in sorted(trial_records.keys()):
        rec_list = trial_records[tid]
        frames_cnt = len(rec_list)
        total_trial_frames += frames_cnt
        f_idxs = [r.get("frame_idx") for _, r in rec_list]
        expected_signs = set(r.get("expected_sign") for _, r in rec_list)
        sess_ids = set(r.get("session_id") for _, r in rec_list)
        timestamps_trial = [r.get("timestamp") for _, r in rec_list if r.get("timestamp") is not None]
        trial_dur = (max(timestamps_trial) - min(timestamps_trial)) if timestamps_trial else 0.0
        
        f_signs = {}
        for _, r in rec_list:
            s = r.get("final_sign")
            f_signs[s] = f_signs.get(s, 0) + 1
            
        correct_p = f_signs.get("P", 0)
        neutral_cnt = f_signs.get("NEUTRAL", 0)
        wrong_cnt = sum(cnt for k, cnt in f_signs.items() if k not in ("P", "NEUTRAL"))
        
        total_correct_p += correct_p
        total_neutral += neutral_cnt
        total_wrong += wrong_cnt
        for k, cnt in f_signs.items():
            if k not in ("P", "NEUTRAL"):
                wrong_confusions[k] = wrong_confusions.get(k, 0) + cnt

        # Continuity check
        f_min = min(f_idxs) if f_idxs else 0
        f_max = max(f_idxs) if f_idxs else 0
        is_continuous = (f_idxs == list(range(f_min, f_max + 1)))
        
        wrong_detail = {k: v for k, v in f_signs.items() if k not in ("P", "NEUTRAL")}
        trial_res = "PASS" if correct_p > 0 and correct_p >= wrong_cnt else ("NEUTRAL" if correct_p == 0 and wrong_cnt == 0 else "FAIL")
        
        print(f"{tid}: {frames_cnt} frames (dur={trial_dur:.2f}s, frame_idx {f_min}..{f_max}, cont={is_continuous})")
        print(f"   Target: {expected_signs} | Session: {sess_ids}")
        print(f"   Correct P: {correct_p} ({correct_p/frames_cnt*100:.1f}%), Neutral: {neutral_cnt} ({neutral_cnt/frames_cnt*100:.1f}%), Wrong: {wrong_cnt} ({wrong_cnt/frames_cnt*100:.1f}%) {wrong_detail}")
        print(f"   Trial Result: {trial_res}")

    print("\n--- OVERALL P TRIAL SUMMARY ---")
    print(f"Total P Trials: {len(trial_records)}")
    print(f"Total P Frames: {total_trial_frames}")
    print(f"Total Correct P Frames: {total_correct_p} ({total_correct_p/total_trial_frames*100:.2f}%)")
    print(f"Total Neutral Frames:   {total_neutral} ({total_neutral/total_trial_frames*100:.2f}%)")
    print(f"Total Wrong Frames:     {total_wrong} ({total_wrong/total_trial_frames*100:.2f}%)")
    print(f"Wrong confusions breakdown: {wrong_confusions}")
    coverage = (total_trial_frames - total_neutral) / total_trial_frames * 100 if total_trial_frames else 0.0
    print(f"Coverage (non-neutral frames): {total_trial_frames - total_neutral}/{total_trial_frames} ({coverage:.2f}%)")
    print()

if __name__ == "__main__":
    audit_file("backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl")
    audit_file("backend/data/diagnostics/pilot_pilot_20261108_042505.jsonl")


def analyze_p_failed_conditions(filepath):
    print("=" * 70)
    print("ANALYSIS OF FAILED CONDITIONS IN P RULE DIAGNOSTICS:", filepath)
    print("=" * 70)
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    p_records = [r for r in records if r.get("trial_id")]
    cond_counts = {}
    failed_cond_counts = {}
    for _, r in enumerate(p_records):
        diag = r.get("geometric", {}).get("p_rule_diagnostics", {})
        conds = diag.get("conditions", {})
        for k, v in conds.items():
            cond_counts.setdefault(k, {"True": 0, "False": 0})
            cond_counts[k]["True" if v else "False"] += 1
        for fc in diag.get("failed_conditions", []):
            failed_cond_counts[fc] = failed_cond_counts.get(fc, 0) + 1

    print(f"Total active frames analyzed: {len(p_records)}")
    print("\nIndividual condition pass/fail distribution:")
    for k, v in cond_counts.items():
        t_cnt = v["True"]
        f_cnt = v["False"]
        print(f"  {k:22s}: PASS={t_cnt:2d}/{len(p_records)} ({t_cnt/len(p_records)*100:5.1f}%) | FAIL={f_cnt:2d}/{len(p_records)} ({f_cnt/len(p_records)*100:5.1f}%)")

    print("\nMost frequent failed conditions (Root causes of NO_MATCH):")
    for k, v in sorted(failed_cond_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:22s}: failed in {v:2d}/{len(p_records)} frames ({v/len(p_records)*100:5.1f}%)")

if __name__ == "__main__":
    analyze_p_failed_conditions("backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl")
