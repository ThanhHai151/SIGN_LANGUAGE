import json

with open('backend/data/diagnostics/automated_validation_report_20261112_231703.json', encoding='utf-8') as f:
    d = json.load(f)

print('=== 7 DIACRITIC LETTERS BATCH REPLAY (10 TRIALS EACH) ===')
for k, v in d['diacritic_7_results'].items():
    st = v['overall_status']
    det = v['is_deterministic']
    sample = v['sample_trial']
    toks = [f"f{t[0]}:{t[2]}" for t in sample['emitted_tokens']]
    print(f"{k}: status={st} (Pass: {v['pass_trials']}/10, Blocked: {v['blocked_trials']}/10) | Deterministic={det}")
    print(f"   Video={v['video_id']} | Base={v['expected_base']} | BaseStableFrame=f{sample['base_stable_frame']} | ConfirmedFrame=f{sample['diacritic_confirmed_frame']}")
    print(f"   Tokens=[{', '.join(toks)}] | Confusions={sample['confusions']}")

print('\n=== 22 STATIC UNACCENTED LETTERS REGRESSION ===')
s22 = d['static_22_summary']
print(f"Total Samples: {s22['total_samples']}")
print(f"Correct: {s22['correct_samples']} ({s22['accuracy_pct']}%)")
print(f"Single-Frame False Positives into 7 Diacritics: {s22['false_positives_into_diacritics_1frame']}")
print(f"10-Frame Static Hold False Positives (Adversarial): {s22['false_positives_into_diacritics_10f_hold']}")

print('\n=== PER-LETTER STATIC DETAILS ===')
for letter, stats in sorted(s22['per_letter'].items()):
    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
    print(f"Letter {letter}: {stats['correct']}/{stats['total']} ({acc:.1f}%) | neutral={stats['neutral']} | wrong={stats['wrong']} | wrong_preds={stats['wrong_preds']}")

