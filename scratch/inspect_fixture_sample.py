import json

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for s in d['samples']:
    if s.get('letter') in ['t', 'u', 's']:
        print(f"Letter {s.get('letter')}: keys={list(s.keys())}, has landmarks={s.get('landmarks') is not None}, num landmarks={len(s.get('landmarks', [])) if s.get('landmarks') else 0}")
        break

