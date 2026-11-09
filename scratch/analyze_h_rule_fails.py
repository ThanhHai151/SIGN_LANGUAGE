import json
import collections

h_horiz_fails = collections.Counter()
h_north_fails = collections.Counter()
h_south_fails = collections.Counter()

total_h = 0
with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'H':
            total_h += 1
            geo = r.get('geometric', {})
            h_diag = geo.get('h_rule_diagnostics', {})
            for c in h_diag.get('h_horiz', {}).get('failed_conditions', []):
                h_horiz_fails[c] += 1
            for c in h_diag.get('h_north', {}).get('failed_conditions', []):
                h_north_fails[c] += 1
            for c in h_diag.get('h_south', {}).get('failed_conditions', []):
                h_south_fails[c] += 1

print(f"Total H frames: {total_h}")

print("\n=== FAILED CONDITIONS IN h_horiz (out of 455 frames) ===")
for c, cnt in h_horiz_fails.most_common():
    print(f"  {c:30s}: {cnt} / 455 ({cnt/455*100:.1f}%)")

print("\n=== FAILED CONDITIONS IN h_south (out of 455 frames) ===")
for c, cnt in h_south_fails.most_common():
    print(f"  {c:30s}: {cnt} / 455 ({cnt/455*100:.1f}%)")

