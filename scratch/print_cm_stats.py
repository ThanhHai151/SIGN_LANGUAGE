import json

with open('scratch/full_batch_regression_report.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("=" * 60)
print(f"{'Letter':<8} | {'Correct':<10} | {'Acc %':<10} | {'Neutral':<10} | {'Wrong'}")
print("=" * 60)

for l, s in sorted(d['static_summary']['per_letter'].items()):
    print(f"{l:<8} | {s['correct']}/{s['total']:<8} | {s['acc_pct']:<10} | {s['neutral']:<10} | {s['wrong']}")

