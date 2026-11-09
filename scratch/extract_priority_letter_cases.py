import os
import json

filepath = os.path.abspath("backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl")

priority_letters = ["H", "X", "P", "N", "M", "O", "E", "Q", "D", "I", "R"]

case_studies = {}

with open(filepath, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        exp = r.get("expected_sign")
        if exp in priority_letters and exp not in case_studies:
            # Look for an informative representative frame:
            # - For H: frame where final_sign == 'V'
            # - For X: frame with x_rule_diagnostics
            # - For P: frame with p_rule_diagnostics
            # - For N: frame where raw MLP predicted N
            # - For M: frame with MLP called
            # - For O: frame where final_sign == 'A'
            # - For E: frame where final_sign == 'C'
            # - For Q: frame with q_rule_diagnostics
            # - For D: frame where final_sign == 'P'
            # - For I: frame where final_sign == 'Y'
            # - For R: frame where final_sign in ('H', 'D')
            final_sign = r.get("final_sign")
            target_hit = False
            if exp == "H" and final_sign == "V":
                target_hit = True
            elif exp == "X" and r.get("frame_idx") == 15:
                target_hit = True
            elif exp == "P" and r.get("frame_idx") == 15:
                target_hit = True
            elif exp == "N" and r.get("static_head", {}).get("raw_predicted_letter") == "N":
                target_hit = True
            elif exp == "M" and r.get("frame_idx") == 15:
                target_hit = True
            elif exp == "O" and final_sign == "A":
                target_hit = True
            elif exp == "E" and final_sign == "C":
                target_hit = True
            elif exp == "Q" and r.get("frame_idx") == 15:
                target_hit = True
            elif exp == "D" and final_sign == "P":
                target_hit = True
            elif exp == "I" and final_sign == "Y":
                target_hit = True
            elif exp == "R" and final_sign == "H":
                target_hit = True

            if target_hit:
                geom = r.get("geometric") or {}
                sh = r.get("static_head") or {}
                case_studies[exp] = {
                    "line": line_idx,
                    "trial_id": r.get("trial_id"),
                    "frame_idx": r.get("frame_idx"),
                    "expected_sign": exp,
                    "final_sign": final_sign,
                    "final_confidence": r.get("final_confidence"),
                    "prediction_source": r.get("prediction_source"),
                    "gating_status": r.get("gating_status"),
                    "threshold_applied": r.get("threshold_applied"),
                    "reject_reason": r.get("reject_reason"),
                    "geometric_status": geom.get("status"),
                    "geometric_letter": geom.get("letter"),
                    "geometric_reason": geom.get("reason"),
                    "matched_candidates": geom.get("matched_candidates"),
                    "static_head_raw": sh.get("raw_predicted_letter"),
                    "static_head_conf": sh.get("confidence"),
                    "static_head_top_k": sh.get("top_k"),
                    "diagnostics_snippet": {
                        k: v for k, v in geom.items() if "diagnostics" in k and v is not None and v.get("matched") is not None
                    }
                }

print(f"Extracted {len(case_studies)} case studies.")
with open("scratch/priority_letter_cases.json", "w", encoding="utf-8") as f:
    json.dump(case_studies, f, indent=2, ensure_ascii=False)

print("Saved to scratch/priority_letter_cases.json")

