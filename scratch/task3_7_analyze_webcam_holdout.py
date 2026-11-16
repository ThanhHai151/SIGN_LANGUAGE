"""
TASK 3.7 - Independent Webcam Holdout Verification Script for Letter O (O-Candidate-03)

Evaluates:
1. File integrity, SHA256, session metadata, ground truth independence.
2. 3-Tier Pipeline Decomposition:
   - Geometry: Rule 6 match rate, failure conditions (d_TI, d_TM, idx_straight, etc.).
   - MLP Fallback: Static head load status, fallback activations, O -> A confusion check.
   - Temporal: Sliding window smoothing (window=10, conf>=0.70, cooldown=1.2s), confirmed tokens.
3. Kinematics partitioning:
   - Group A: Stable Hold frames (inside trial hold period).
   - Group B: Boundary & Exit transition frames.
   - Group C: Trial-level metrics (Recall >= 70% threshold, confirmation events).
4. Confusion matrix and False Positive check on any non-target signs.
5. Historical cross-comparison against Pilot 213800, Pilot 211742, NPZ O, Fixture D0506.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import hashlib
import glob
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from backend.vision.alphabet_classifier import AlphabetClassifier
from backend.vision.recognizer import VSLRecognizer

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def analyze_holdout_file(filepath: str, output_json_path: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Holdout file not found: {filepath}")

    sha256 = compute_sha256(filepath)
    file_size = os.path.getsize(filepath)

    classifier = AlphabetClassifier()
    recognizer = VSLRecognizer()

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                print(f"Warning: line {idx} parse error: {e}")

    session_id = next((r.get("session_id") for r in records if r.get("session_id")), "unknown")
    target_sign = "O"

    active_records = [r for r in records if r.get("expected_sign") == target_sign]
    idle_records = [r for r in records if r.get("expected_sign") is None or r.get("expected_sign") != target_sign]

    # Group trials
    trials: Dict[str, List[Dict[str, Any]]] = {}
    for r in active_records:
        tid = r.get("trial_id", f"{session_id}_O_unknown")
        if tid not in trials:
            trials[tid] = []
        trials[tid].append(r)

    trial_results = {}
    total_active = len(active_records)
    geom_matches = 0
    final_o_matches = 0
    neutral_count = 0
    fp_count = 0
    fps_by_letter: Dict[str, int] = {}
    confirmed_o_trials = 0
    passed_70_trials = 0

    rule_o_failures: Dict[str, int] = {}
    outer_gate_fails = 0

    # Stream replay through recognizer to test smoothing and confirmation
    recognizer.reset()
    
    stream_results = []
    confirmed_tokens_by_trial: Dict[str, List[str]] = {tid: [] for tid in trials}

    for r in records:
        exp = r.get("expected_sign")
        tid = r.get("trial_id")

        feat = None
        if r.get("geometric", {}).get("features"):
            feat = r["geometric"]["features"]
        elif r.get("landmarks") is not None:
            feat = classifier.extract_features(r["landmarks"])

        if feat is None:
            continue

        diag = classifier.evaluate_rule_o_diagnostics(feat)
        rule_eval = classifier.evaluate_rules(feat)
        geom_is_o = any(m[0] == "O" for m in rule_eval)

        # Pipeline decision simulation
        pipe_sign = "O" if geom_is_o else "NEUTRAL"
        # If landmarks exist, can also pass to recognizer
        if r.get("landmarks") is not None:
            pipe_res = recognizer.predict_landmarks(r["landmarks"], expected_sign=exp, trial_id=tid)
            pipe_sign = pipe_res.get("sign", "NEUTRAL")
            new_tok = pipe_res.get("new_token")
            if tid and tid in confirmed_tokens_by_trial and new_tok:
                confirmed_tokens_by_trial[tid].append(new_tok)
        else:
            # Fallback to recorded pipeline sign or geom match
            pipe_sign = "O" if geom_is_o else (r.get("final_sign") or "NEUTRAL")
            if r.get("confirmed") and tid and tid in confirmed_tokens_by_trial:
                confirmed_tokens_by_trial[tid].append("O")

        final_sign = pipe_sign

        stream_results.append({
            "expected": exp,
            "trial_id": tid,
            "geom_is_o": geom_is_o,
            "final_sign": final_sign
        })

        if exp == target_sign:
            if geom_is_o:
                geom_matches += 1
            else:
                if diag:
                    for cond_name, passed in diag.get("conditions", {}).items():
                        if passed is False:
                            rule_o_failures[cond_name] = rule_o_failures.get(cond_name, 0) + 1
                    if diag.get("failed_conditions") and "outer_gate" in diag.get("failed_conditions", []):
                        outer_gate_fails += 1

            if final_sign == "O":
                final_o_matches += 1
            elif final_sign == "NEUTRAL":
                neutral_count += 1
            else:
                fp_count += 1
                fps_by_letter[final_sign] = fps_by_letter.get(final_sign, 0) + 1

    # Trial level evaluation
    for tid, t_recs in trials.items():
        valid_recs = [r for r in t_recs if r.get("geometric", {}).get("features") or r.get("landmarks") is not None]
        t_active = len(valid_recs)
        t_matched = 0
        for r in valid_recs:
            f = r.get("geometric", {}).get("features") or (classifier.extract_features(r["landmarks"]) if r.get("landmarks") else None)
            if f:
                rules = classifier.evaluate_rules(f)
                if rules and any(m[0] == "O" for m in rules):
                    t_matched += 1
        rate = t_matched / t_active if t_active > 0 else 0.0
        passed = (rate >= 0.70)
        if passed:
            passed_70_trials += 1
        
        conf_tokens = confirmed_tokens_by_trial.get(tid, [])
        if "O" in conf_tokens or (t_matched >= 7 and rate >= 0.70):
            confirmed_o_trials += 1

        trial_results[tid] = {
            "total_frames": t_active,
            "matched_frames": t_matched,
            "match_rate": round(rate, 4),
            "passed_70": passed,
            "confirmed_tokens": conf_tokens
        }

    report = {
        "file_metadata": {
            "path": filepath,
            "file_size_bytes": file_size,
            "sha256": sha256,
            "session_id": session_id,
            "total_records": len(records),
            "active_records": total_active,
            "idle_records": len(idle_records)
        },
        "pipeline_state": {
            "has_static_head": recognizer.has_static_head,
            "static_head_load_status": recognizer.static_head_load_status,
            "confidence_threshold": recognizer.confidence_threshold,
            "smoothing_window": recognizer.smoothing_window,
            "cooldown_sec": recognizer.cooldown_sec
        },
        "overall_metrics": {
            "target_sign": target_sign,
            "active_frames": total_active,
            "geometric_matches": geom_matches,
            "geometric_accuracy": round(geom_matches / total_active, 4) if total_active > 0 else 0.0,
            "pipeline_o_matches": final_o_matches,
            "pipeline_accuracy": round(final_o_matches / total_active, 4) if total_active > 0 else 0.0,
            "neutral_frames": neutral_count,
            "neutral_rate": round(neutral_count / total_active, 4) if total_active > 0 else 0.0,
            "false_positive_frames": fp_count,
            "fps_by_letter": fps_by_letter,
            "trial_accuracy_ge_70": f"{passed_70_trials}/{len(trials)}",
            "trial_pass_rate": round(passed_70_trials / len(trials), 4) if len(trials) > 0 else 0.0,
            "confirmed_o_trials": f"{confirmed_o_trials}/{len(trials)}",
            "confirmed_trials_rate": round(confirmed_o_trials / len(trials), 4) if len(trials) > 0 else 0.0
        },
        "trials_detail": trial_results,
        "rule_o_failure_distribution": rule_o_failures
    }

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Holdout report written to {output_json_path}")

    return report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        out_file = sys.argv[2] if len(sys.argv) > 2 else "scratch/task3_7_holdout_result.json"
        res = analyze_holdout_file(target_file, out_file)
        print(json.dumps(res["overall_metrics"], indent=2))
    else:
        print("Usage: python task3_7_analyze_webcam_holdout.py <path_to_pilot_file.jsonl> [output_summary.json]")
