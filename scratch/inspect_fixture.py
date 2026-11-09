import json

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

samples = d.get('samples', {})
print("samples type:", type(samples))
if isinstance(samples, dict):
    print("Keys in samples:", list(samples.keys()))
    for letter in ['T', 'U', 'S']:
        if letter in samples:
            item = samples[letter]
            print(f"Letter {letter}: keys={list(item.keys()) if isinstance(item, dict) else len(item)}")
elif isinstance(samples, list):
    letters = {}
    for s in samples:
        let = s.get('letter') or s.get('letter_label')
        letters[let] = letters.get(let, 0) + 1
    print("Letters in samples list:", letters)

