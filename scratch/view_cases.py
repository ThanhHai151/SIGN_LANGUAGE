import json

cs = json.load(open("scratch/priority_letter_cases.json", encoding="utf-8"))
for k, v in cs.items():
    print(f"\n=======================================================")
    print(f"CASE STUDY: LETTER {k}")
    print(f"=======================================================")
    print(f"Trial ID:          {v['trial_id']}")
    print(f"Frame Index:       {v['frame_idx']} (Line {v['line']})")
    print(f"Final Prediction:  {v['final_sign']} (Conf: {v['final_confidence']}, Source: {v['prediction_source']})")
    print(f"Geometric Rule:    Status={v['geometric_status']}, Letter={v['geometric_letter']}")
    print(f"                   Reason: {v['geometric_reason']}")
    print(f"Static Head Raw:   Prediction={v['static_head_raw']}, Conf={v['static_head_conf']}")
    print(f"                   Top-K: {v['static_head_top_k']}")
    print(f"Gating:            Status={v['gating_status']}, Reject Reason={v['reject_reason']}")
    print("Diagnostics:")
    for d_name, d_val in v.get("diagnostics_snippet", {}).items():
        if isinstance(d_val, dict):
            failed = d_val.get("failed_conditions", [])
            print(f"  {d_name}: matched={d_val.get('matched')}, failed_conditions={failed[:5]}")

