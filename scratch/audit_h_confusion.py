import json
import collections
import numpy as np

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if json.loads(line).get('trial_id') and json.loads(line).get('expected_sign')]

h_frames = [r for r in records if r['expected_sign'] == 'H']
v_frames = [r for r in records if r['expected_sign'] == 'V']

print(f"Total H frames: {len(h_frames)}")
print(f"Total V frames: {len(v_frames)}")

# Final sign distribution of H
h_final_signs = collections.Counter([r['final_sign'] for r in h_frames])
print("Final signs for Expected H:", dict(h_final_signs))

# Breakdown of the 164 incorrect frames by final_sign and prediction_source
h_incorrect = [r for r in h_frames if r['final_sign'] not in ('H', 'NEUTRAL')]
print(f"\nTotal H incorrect frames: {len(h_incorrect)}")
inc_breakdown = collections.Counter([(r['final_sign'], r['prediction_source'], r.get('geometric', {}).get('status'), r.get('gating_status')) for r in h_incorrect])
for k, v in inc_breakdown.items():
    print(f"  Final={k[0]}, Source={k[1]}, GeoStatus={k[2]}, Gating={k[3]} -> Count={v}")

# 140 H -> V frames inspection
h_to_v = [r for r in h_frames if r['final_sign'] == 'V']
print(f"\n=== INSPECTING 140 H -> V FRAMES ===")
print("Prediction sources:", collections.Counter([r['prediction_source'] for r in h_to_v]))
print("Geometric statuses:", collections.Counter([r.get('geometric', {}).get('status') for r in h_to_v]))
print("Geometric letters:", collections.Counter([r.get('geometric', {}).get('letter') for r in h_to_v]))
print("Gating statuses:", collections.Counter([r.get('gating_status') for r in h_to_v]))
print("MLP raw preds:", collections.Counter([r.get('static_head', {}).get('raw_predicted_letter') for r in h_to_v]))

# Per trial breakdown of H frames
print("\n=== PER-TRIAL BREAKDOWN FOR LETTER H ===")
trials_h = collections.OrderedDict()
for r in h_frames:
    tid = r['trial_id']
    if tid not in trials_h:
        trials_h[tid] = collections.Counter()
    trials_h[tid][r['final_sign']] += 1
    trials_h[tid]['_total'] += 1

for tid, c in trials_h.items():
    tot = c['_total']
    v_cnt = c.get('V', 0)
    k_cnt = c.get('K', 0)
    r_cnt = c.get('R', 0)
    neu_cnt = c.get('NEUTRAL', 0)
    print(f"{tid}: Total={tot}, V={v_cnt} ({v_cnt/tot*100:.1f}%), K={k_cnt}, R={r_cnt}, Neutral={neu_cnt} ({neu_cnt/tot*100:.1f}%)")

# Save summary to file
with open('scratch/h_trial_breakdown.json', 'w', encoding='utf-8') as f:
    json.dump({tid: dict(c) for tid, c in trials_h.items()}, f, indent=2)

