import json
import collections

h_to_v_diag = []

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'H' and r.get('final_sign') == 'V':
            geo = r.get('geometric', {})
            h_diag = geo.get('h_rule_diagnostics', {})
            h_to_v_diag.append({
                'trial_id': r.get('trial_id'),
                'frame_idx': r.get('frame_idx'),
                'features': geo.get('features', {}),
                'h_diag': h_diag
            })

print(f"Total H -> V frames analyzed: {len(h_to_v_diag)}")

# Check rule_v conditions
rule_v_matches = collections.Counter([x['h_diag'].get('rule_v', {}).get('matched') for x in h_to_v_diag])
print("Rule V matched in h_rule_diagnostics:", dict(rule_v_matches))

# Check failed conditions in h_horiz
h_horiz_failed_conds = collections.Counter()
for x in h_to_v_diag:
    fc = x['h_diag'].get('h_horiz', {}).get('failed_conditions', [])
    for c in fc:
        h_horiz_failed_conds[c] += 1
print("\nFailed conditions in h_horiz (out of 140 frames):")
for c, cnt in h_horiz_failed_conds.most_common():
    print(f"  {c:30s}: {cnt} / 140 ({cnt/140*100:.1f}%)")

# Check failed conditions in h_south
h_south_failed_conds = collections.Counter()
for x in h_to_v_diag:
    fc = x['h_diag'].get('h_south', {}).get('failed_conditions', [])
    for c in fc:
        h_south_failed_conds[c] += 1
print("\nFailed conditions in h_south (out of 140 frames):")
for c, cnt in h_south_failed_conds.most_common():
    print(f"  {c:30s}: {cnt} / 140 ({cnt/140*100:.1f}%)")

