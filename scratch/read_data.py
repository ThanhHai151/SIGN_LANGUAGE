import json, sys
sys.stdout.reconfigure(encoding='utf-8')

print("=== BENCHMARK 30 FINAL ===")
with open('scratch/benchmark_30_final.json', encoding='utf-8') as f:
    bench = json.load(f)
print(json.dumps(bench, ensure_ascii=False, indent=2))

print("\n=== PROPOSED EVALUATIONS ===")
with open('scratch/proposed_evaluations.json', encoding='utf-8') as f:
    evals = json.load(f)
print(json.dumps(evals, ensure_ascii=False, indent=2))

