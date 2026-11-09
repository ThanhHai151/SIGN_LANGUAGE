import json
import collections

h_neu_records = []

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'H' and r.get('final_sign') == 'NEUTRAL':
            h_neu_records.append(r)

print(f"Total H -> NEUTRAL frames: {len(h_neu_records)}")

# Geometric statuses of H -> NEUTRAL
geo_statuses = collections.Counter([r.get('geometric', {}).get('status') for r in h_neu_records])
print("Geometric statuses in H -> NEUTRAL:", dict(geo_statuses))

# Geometric reasons in H -> NEUTRAL
geo_reasons = collections.Counter([r.get('geometric', {}).get('reason') for r in h_neu_records])
print("\nTop Geometric reasons in H -> NEUTRAL:")
for r_str, cnt in geo_reasons.most_common(10):
    print(f"  {r_str}: {cnt}")

# Check rule_v failed conditions in h_rule_diagnostics for H -> NEUTRAL
v_failed_conds = collections.Counter()
for r in h_neu_records:
    geo = r.get('geometric', {})
    h_diag = geo.get('h_rule_diagnostics', {})
    rule_v = h_diag.get('rule_v', {})
    for fc in rule_v.get('failed_conditions', []):
        v_failed_conds[fc] += 1

print("\nFailed conditions in rule_v for H -> NEUTRAL (out of 291 frames):")
for c, cnt in v_failed_conds.most_common():
    print(f"  {c:30s}: {cnt} / 291 ({cnt/291*100:.1f}%)")

# Check MLP raw predictions for H -> NEUTRAL
sh_preds = collections.Counter([r.get('static_head', {}).get('raw_predicted_letter') for r in h_neu_records])
print("\nMLP raw predictions for H -> NEUTRAL (when geometric is NO_MATCH):")
for p, cnt in sh_preds.most_common():
    print(f"  {p}: {cnt}")

# Check reject reasons for H -> NEUTRAL
rej_reasons = collections.Counter([r.get('reject_reason') for r in h_neu_records])
print("\nReject reasons for H -> NEUTRAL:")
for rr, cnt in rej_reasons.most_common(5):
    print(f"  {rr}: {cnt}")

