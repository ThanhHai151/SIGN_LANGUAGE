import json

with open('backend/data/sources/qipedc_dictionary.json', 'r', encoding='utf-8') as f:
    q = json.load(f)

for item in q:
    code = item.get('code', '')
    word = item.get('word', '')
    if any(k in code for k in ['D0512', 'D0513', 'D0514', 'D0515', 'D0516']):
        print(f"code: {code}, word: {word}, desc: {item.get('description')}")

