import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

clf = VSLAlphabetClassifier()

target_letters = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'O', 'S', 'T', 'V', 'Y']
supported_results = {l: {"total": 0, "match_expected": 0, "match_other": 0, "no_match": 0, "conflict": 0} for l in target_letters}
unsupported_results = {"total": 0, "no_match_unknown": 0, "false_positive": 0, "false_positives_detail": []}

for s in data['samples']:
    if s.get('quality_status') != 'ok':
        continue
    letter = s['letter_upper']
    landmarks = s['landmarks']
    res = clf.classify_detailed(landmarks)
    pred = res['letter']
    status = res['status']
    phase = s.get('phase', 'unknown')

    if letter in target_letters:
        supported_results[letter]["total"] += 1
        if pred == letter:
            supported_results[letter]["match_expected"] += 1
        elif status == "NO_MATCH":
            supported_results[letter]["no_match"] += 1
            print(f"[MISS] {letter} frame {s['frame_idx']} ({phase}): NO_MATCH - {res['reason']}")
        elif status == "CONFLICT":
            supported_results[letter]["conflict"] += 1
            print(f"[CONFLICT] {letter} frame {s['frame_idx']} ({phase}): {res['reason']}")
        else:
            supported_results[letter]["match_other"] += 1
            print(f"[WRONG] {letter} frame {s['frame_idx']} ({phase}): predicted {pred}")
    else:
        unsupported_results["total"] += 1
        if pred == "UNKNOWN":
            unsupported_results["no_match_unknown"] += 1
        else:
            unsupported_results["false_positive"] += 1
            unsupported_results["false_positives_detail"].append({
                "true_letter": letter,
                "pred_letter": pred,
                "frame": s['frame_idx'],
                "phase": phase,
                "reason": res['reason']
            })

print("\n" + "="*50)
print("KET QUA TRANH DANH CHO CAC CHU CO LUAT:")
print("="*50)
for l in target_letters:
    r = supported_results[l]
    print(f"Letter {l}: {r['match_expected']}/{r['total']} correct | NoMatch: {r['no_match']} | Conflict: {r['conflict']} | Wrong: {r['match_other']}")

print("\n" + "="*50)
print(f"CAC CHU CHUA HO TRO (Tong: {unsupported_results['total']} mau):")
print(f"Dung tra ve UNKNOWN: {unsupported_results['no_match_unknown']}/{unsupported_results['total']}")
print(f"False Positive: {unsupported_results['false_positive']}")
for fp in unsupported_results["false_positives_detail"]:
    print(f"  True: {fp['true_letter']} -> Pred: {fp['pred_letter']} (frame {fp['frame']}, {fp['phase']}) | {fp['reason']}")

