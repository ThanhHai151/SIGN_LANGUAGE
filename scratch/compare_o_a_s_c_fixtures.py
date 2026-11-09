import json

data = json.load(open("scratch/task1_2_full_audit_metrics.json"))
fixtures = data["fixtures"]

print("=== FIXTURE COMPARISON (STABLE POSE ONLY) ===")
print(f"{'Letter':<6} | {'Total':<6} | {'Stable':<6} | {'Outer Gate Pass':<16} | {'Rule O Match':<14} | {'Classifier MATCH':<20}")
for let in ["O", "A", "S", "C"]:
    f_info = fixtures[let]
    st = f_info["stable_pose_summary"]
    tot = f_info["total_samples"]
    cnt = st.get("count", 0)
    og = f"{st.get('outer_gate_pass', 0)}/{cnt} ({st.get('outer_gate_pass_rate', 0)*100:.1f}%)"
    ro = f"{st.get('rule_o_match', 0)}/{cnt} ({st.get('rule_o_match_rate', 0)*100:.1f}%)"
    cl = str(st.get("classifier_letter_counts", {}))
    print(f"{let:<6} | {tot:<6} | {cnt:<6} | {og:<16} | {ro:<14} | {cl:<20}")

print("\n=== PASS RATES FOR RULE O CONDITIONS ON STABLE POSE ===")
cond_keys = [
    "d_ti_le_max", "d_tm_le_max", "d_iw_ge_min", "d_mw_ge_min",
    "d_rw_ge_min", "d_pw_ge_min", "idx_straight_le_max", "d_tw_ge_min"
]
header = f"{'Condition':<22} | {'O (20)':<10} | {'A (27)':<10} | {'S (21)':<10} | {'C (24)':<10}"
print(header)
print("-" * len(header))
for c in cond_keys:
    o_val = f"{fixtures['O']['stable_pose_summary']['cond_passes'][c]}/20 ({fixtures['O']['stable_pose_summary']['cond_pass_rates'][c]*100:.0f}%)"
    a_val = f"{fixtures['A']['stable_pose_summary']['cond_passes'][c]}/27 ({fixtures['A']['stable_pose_summary']['cond_pass_rates'][c]*100:.0f}%)"
    s_val = f"{fixtures['S']['stable_pose_summary']['cond_passes'][c]}/21 ({fixtures['S']['stable_pose_summary']['cond_pass_rates'][c]*100:.0f}%)"
    c_val = f"{fixtures['C']['stable_pose_summary']['cond_passes'][c]}/24 ({fixtures['C']['stable_pose_summary']['cond_pass_rates'][c]*100:.0f}%)"
    print(f"{c:<22} | {o_val:<10} | {a_val:<10} | {s_val:<10} | {c_val:<10}")

print("\n=== COMPARATIVE FEATURE DISTRIBUTION (MEDIAN [P25, P75]) ===")
feat_names = ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]
f_header = f"{'Feature':<14} | {'Rule O Thresh':<14} | {'O (Stable 20)':<25} | {'A (Stable 27)':<25} | {'S (Stable 21)':<25} | {'C (Stable 24)':<25}"
print(f_header)
print("-" * len(f_header))
thresh_map = {
    "d_TI": "<= 0.22",
    "d_TM": "<= 0.35",
    "d_IW": ">= 0.95",
    "d_MW": ">= 0.90",
    "d_RW": ">= 0.80",
    "d_PW": ">= 0.75",
    "idx_straight": "<= 0.48",
    "d_TW": ">= 0.85"
}
for fn in feat_names:
    def _fmt(st_dict):
        f = st_dict["features_stats"][fn]
        return f"{f['median']:.3f} [{f['p25']:.3f}, {f['p75']:.3f}]"
    o_str = _fmt(fixtures['O']['stable_pose_summary'])
    a_str = _fmt(fixtures['A']['stable_pose_summary'])
    s_str = _fmt(fixtures['S']['stable_pose_summary'])
    c_str = _fmt(fixtures['C']['stable_pose_summary'])
    print(f"{fn:<14} | {thresh_map[fn]:<14} | {o_str:<25} | {a_str:<25} | {s_str:<25} | {c_str:<25}")

