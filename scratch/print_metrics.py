import json

with open('backend/data/alphabet_v2/audit/static_head_v1_clean_baseline.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("=== 22 CLASSES METRICS ===")
for c, m in d['per_class_metrics_22'].items():
    sup = m['support']
    tp = m['tp']
    fn = m['fn']
    fp = m['fp']
    prec = f"{m['precision']*100:.1f}%" if m['precision'] is not None else '—'
    rec = f"{m['recall']*100:.1f}%" if m['recall'] is not None else 'N/A'
    f1 = f"{m['f1_score']:.4f}" if m['f1_score'] is not None else '—'
    confused = m['confused_to']
    print(f"{c:<2} | Sup: {sup:>2} | TP: {tp:>2} | FN: {fn:>2} | FP: {fp:>2} | Prec: {prec:>6} | Rec: {rec:>6} | F1: {f1:>6} | Confused: {confused}")

print("\n=== HIGH CONFIDENCE ERROR ===")
for e in d['confidence_analysis']['high_confidence_errors_samples']:
    print(e)

print("\n=== ALL INCORRECT PAIRS ===")
pairs = {}
with open('backend/data/alphabet_v2/audit/static_head_v1_clean_predictions.json', 'r', encoding='utf-8') as f:
    preds = json.load(f)
for s in preds:
    if not s['is_correct']:
        pair = f"{s['true_label']} -> {s['pred_label']}"
        pairs[pair] = pairs.get(pair, 0) + 1
for p, cnt in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
    print(f"{p}: {cnt} lần")

